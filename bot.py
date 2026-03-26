import os
import logging
import tempfile
import asyncio
from pathlib import Path

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

from audio_processor import clean_audio_simple, clean_audio_ai

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.environ.get("BOT_TOKEN", "")

# Webhook settings — set these in Render's environment variables
# WEBHOOK_URL  : your full Render URL e.g. https://your-bot.onrender.com
# PORT         : Render sets this automatically (default 8443)
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "")
PORT = int(os.environ.get("PORT", 8443))

# Store pending audio files per user: user_id -> temp_file_path
pending_audio: dict[int, str] = {}


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "🎙️ *Audio Cleaner Bot*\n\n"
        "Send me a voice message or audio file and I'll remove background noise for you.\n\n"
        "Commands:\n"
        "/start – Show this message\n"
        "/help – How to use the bot",
        parse_mode="Markdown",
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "📖 *How to use:*\n\n"
        "1. Send a voice message or audio file\n"
        "2. Choose your cleaning method:\n"
        "   • *Fast* – Quick spectral noise reduction\n"
        "   • *AI (High Quality)* – Deep learning model, slower but better\n"
        "3. Receive your cleaned audio!\n\n"
        "Supported formats: OGG, MP3, WAV, M4A, FLAC",
        parse_mode="Markdown",
    )


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    message = update.message

    # Get the file object (voice or audio)
    file_obj = message.voice or message.audio
    if not file_obj:
        await message.reply_text("❌ Please send a voice message or audio file.")
        return

    await message.reply_text("⏳ Downloading your audio...")

    try:
        # Download the file
        tg_file = await context.bot.get_file(file_obj.file_id)
        suffix = ".ogg" if message.voice else Path(file_obj.file_name or "audio.mp3").suffix
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        await tg_file.download_to_drive(tmp.name)
        tmp.close()

        # Store for later processing
        pending_audio[user_id] = tmp.name

        # Ask for method
        keyboard = [
            [
                InlineKeyboardButton("⚡ Fast (noisereduce)", callback_data="method_fast"),
                InlineKeyboardButton("🧠 AI High Quality", callback_data="method_ai"),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await message.reply_text(
            "✅ Audio received! Choose your cleaning method:",
            reply_markup=reply_markup,
        )

    except Exception as e:
        logger.error(f"Download error: {e}")
        await message.reply_text("❌ Failed to download audio. Please try again.")


async def handle_method_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    method = query.data  # "method_fast" or "method_ai"

    input_path = pending_audio.get(user_id)
    if not input_path or not os.path.exists(input_path):
        await query.edit_message_text("❌ No audio found. Please send your audio again.")
        return

    method_label = "⚡ Fast" if method == "method_fast" else "🧠 AI High Quality"
    await query.edit_message_text(f"🔄 Processing with *{method_label}* method... Please wait.", parse_mode="Markdown")

    try:
        output_path = tempfile.mktemp(suffix="_cleaned.wav")

        if method == "method_fast":
            await asyncio.get_event_loop().run_in_executor(
                None, clean_audio_simple, input_path, output_path
            )
        else:
            await asyncio.get_event_loop().run_in_executor(
                None, clean_audio_ai, input_path, output_path
            )

        # Send back cleaned audio
        with open(output_path, "rb") as f:
            await context.bot.send_audio(
                chat_id=query.message.chat_id,
                audio=f,
                filename="cleaned_audio.wav",
                caption=f"✅ Done! Cleaned with *{method_label}* method.",
                parse_mode="Markdown",
            )

        await query.edit_message_text(f"✅ Cleaned with *{method_label}* — audio sent above!", parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Processing error: {e}")
        await query.edit_message_text("❌ Processing failed. Please try again.")

    finally:
        # Cleanup temp files
        for path in [input_path, output_path if 'output_path' in locals() else None]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
        pending_audio.pop(user_id, None)


def main() -> None:
    if not BOT_TOKEN:
        raise ValueError("BOT_TOKEN environment variable not set!")

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))
    app.add_handler(CallbackQueryHandler(handle_method_choice, pattern="^method_"))

    if WEBHOOK_URL:
        # ── Production (Render) ──────────────────────────────────────────
        # Telegram will POST updates to: https://your-bot.onrender.com/<token>
        logger.info(f"Starting webhook on port {PORT} → {WEBHOOK_URL}")
        app.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path=BOT_TOKEN,                        # secret path
            webhook_url=f"{WEBHOOK_URL}/{BOT_TOKEN}",  # full URL sent to Telegram
            allowed_updates=Update.ALL_TYPES,
        )
    else:
        # ── Local development (Windows / no public URL) ──────────────────
        logger.info("No WEBHOOK_URL set — falling back to long polling (local dev mode)")
        app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
