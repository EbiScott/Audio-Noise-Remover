import os
import re
import logging
import tempfile
import asyncio
from pathlib import Path

import gdown
import requests

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
WEBHOOK_URL = os.environ.get("WEBHOOK_URL", "")
PORT = int(os.environ.get("PORT") or 8443)

TELEGRAM_MAX_BYTES = 20 * 1024 * 1024  # 20 MB

# Store pending audio files per user: user_id -> temp_file_path
pending_audio: dict[int, str] = {}


# ── Google Drive helpers ─────────────────────────────────────────────────────

GDRIVE_PATTERNS = [
    r"https://drive\.google\.com/file/d/([a-zA-Z0-9_-]+)",
    r"https://drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)",
    r"https://drive\.google\.com/uc\?id=([a-zA-Z0-9_-]+)",
]

def extract_gdrive_id(text: str) -> str | None:
    for pattern in GDRIVE_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None

def download_from_gdrive(file_id: str, dest_path: str) -> None:
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest_path, quiet=False, fuzzy=True)


# ── Shared: ask method ───────────────────────────────────────────────────────

async def ask_method(message, user_id: int, file_path: str) -> None:
    pending_audio[user_id] = file_path
    keyboard = [[
        InlineKeyboardButton("⚡ Fast (noisereduce)", callback_data="method_fast"),
        InlineKeyboardButton("🧠 AI High Quality", callback_data="method_ai"),
    ]]
    await message.reply_text(
        "✅ Audio received! Choose your cleaning method:",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


# ── Handlers ─────────────────────────────────────────────────────────────────

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "🎙️ *Audio Cleaner Bot*\n\n"
        "Send me a voice message or audio file (under 20MB) and I'll remove background noise.\n\n"
        "For files *over 20MB*, share a Google Drive link (set to 'Anyone with the link').\n\n"
        "Commands:\n"
        "/start – Show this message\n"
        "/help – How to use the bot",
        parse_mode="Markdown",
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "📖 *How to use:*\n\n"
        "*Small files (under 20MB):*\n"
        "Just send the audio file or voice message directly.\n\n"
        "*Large files (over 20MB):*\n"
        "1. Upload to Google Drive\n"
        "2. Right-click → Share → 'Anyone with the link'\n"
        "3. Paste the link here\n\n"
        "*Then choose your cleaning method:*\n"
        "• ⚡ *Fast* – Quick spectral noise reduction\n"
        "• 🧠 *AI High Quality* – Deep learning, slower but better\n\n"
        "Supported formats: OGG, MP3, WAV, M4A, FLAC",
        parse_mode="Markdown",
    )


async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle direct audio/voice file uploads (Telegram limit: 20MB)."""
    user_id = update.effective_user.id
    message = update.message

    file_obj = message.voice or message.audio
    if not file_obj:
        await message.reply_text("❌ Please send a voice message or audio file.")
        return

    # Check file size before attempting download
    file_size = getattr(file_obj, "file_size", 0) or 0
    if file_size > TELEGRAM_MAX_BYTES:
        await message.reply_text(
            f"⚠️ Your file is *{file_size / 1024 / 1024:.1f}MB* which exceeds Telegram's 20MB bot limit.\n\n"
            "Please upload it to *Google Drive*, set sharing to 'Anyone with the link', and paste the link here.",
            parse_mode="Markdown",
        )
        return

    await message.reply_text("⏳ Downloading your audio...")

    try:
        tg_file = await context.bot.get_file(file_obj.file_id)
        suffix = ".ogg" if message.voice else Path(file_obj.file_name or "audio.mp3").suffix
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        await tg_file.download_to_drive(tmp.name)
        tmp.close()
        await ask_method(message, user_id, tmp.name)

    except Exception as e:
        logger.error(f"Download error: {e}", exc_info=True)
        await message.reply_text(f"❌ Failed to download audio. Error: `{e}`", parse_mode="Markdown")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle Google Drive links sent as text messages."""
    user_id = update.effective_user.id
    message = update.message
    text = message.text or ""

    file_id = extract_gdrive_id(text)
    if not file_id:
        await message.reply_text(
            "💬 I only accept audio files or Google Drive links.\n"
            "Send /help for instructions."
        )
        return

    await message.reply_text("⏳ Downloading from Google Drive... This may take a moment.")

    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".audio")
        tmp.close()

        await asyncio.get_event_loop().run_in_executor(
            None, download_from_gdrive, file_id, tmp.name
        )

        if not os.path.exists(tmp.name) or os.path.getsize(tmp.name) == 0:
            await message.reply_text("❌ Download failed. Make sure the file is shared as 'Anyone with the link'.")
            return

        # Detect real format and rename so ffmpeg can identify it
        import subprocess
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=format_name",
             "-of", "default=noprint_wrappers=1:nokey=1", tmp.name],
            capture_output=True, text=True
        )
        fmt = probe.stdout.strip().split(",")[0]
        ext_map = {"mp3": ".mp3", "m4a": ".m4a", "aac": ".m4a", "ogg": ".ogg",
                   "wav": ".wav", "flac": ".flac", "mov,mp4,m4a,3gp,3g2,mj2": ".m4a"}
        ext = ext_map.get(fmt, ".m4a")
        renamed = tmp.name + ext
        os.rename(tmp.name, renamed)
        logger.info(f"Detected format: {fmt}, renamed to {renamed}")

        size_mb = os.path.getsize(renamed) / 1024 / 1024
        await message.reply_text(f"✅ Downloaded ({size_mb:.1f}MB). Now choose your cleaning method:")
        await ask_method(message, user_id, renamed)

    except Exception as e:
        logger.error(f"Google Drive download error: {e}", exc_info=True)
        await message.reply_text(
            "❌ Failed to download from Google Drive.\n"
            "Make sure sharing is set to *'Anyone with the link'*.",
            parse_mode="Markdown",
        )


async def handle_method_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    method = query.data

    input_path = pending_audio.get(user_id)
    if not input_path or not os.path.exists(input_path):
        await query.edit_message_text("❌ No audio found. Please send your audio again.")
        return

    method_label = "⚡ Fast" if method == "method_fast" else "🧠 AI High Quality"
    ai_warning = (
        "\n\n⚠️ *AI mode can take 5–15 minutes on this server.* Please be patient and don't resend the file."
        if method == "method_ai" else ""
    )
    await query.edit_message_text(
        f"🔄 Processing with *{method_label}* method... Please wait.{ai_warning}",
        parse_mode="Markdown",
    )

    output_path = tempfile.mktemp(suffix="_cleaned.wav")

    try:
        if method == "method_fast":
            await asyncio.get_event_loop().run_in_executor(
                None, clean_audio_simple, input_path, output_path
            )
        else:
            await asyncio.get_event_loop().run_in_executor(
                None, clean_audio_ai, input_path, output_path
            )

        logger.info(f"Processing done. Output exists: {os.path.exists(output_path)}, size: {os.path.getsize(output_path) if os.path.exists(output_path) else 0} bytes")
        await query.edit_message_text("✅ Processing done! Sending audio...", parse_mode="Markdown")

        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            await query.edit_message_text("❌ Processing produced an empty file. Please try again.")
            return

        with open(output_path, "rb") as f:
            await context.bot.send_audio(
                chat_id=query.message.chat_id,
                audio=f,
                filename="cleaned_audio.wav",
                caption=f"✅ Done! Cleaned with *{method_label}* method.",
                parse_mode="Markdown",
            )

        await query.edit_message_text(
            f"✅ Cleaned with *{method_label}* — audio sent above!",
            parse_mode="Markdown",
        )

    except Exception as e:
        logger.error(f"Processing error: {e}", exc_info=True)
        await query.edit_message_text(f"❌ Processing failed. Error: <code>{e}</code>", parse_mode="HTML")

    finally:
        for path in [input_path, output_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
        pending_audio.pop(user_id, None)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if not BOT_TOKEN:
        raise ValueError("BOT_TOKEN environment variable not set!")

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_audio))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(CallbackQueryHandler(handle_method_choice, pattern="^method_"))

    if WEBHOOK_URL:
        logger.info(f"Starting webhook on port {PORT} → {WEBHOOK_URL}")
        app.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path=BOT_TOKEN,
            webhook_url=f"{WEBHOOK_URL}/{BOT_TOKEN}",
            allowed_updates=Update.ALL_TYPES,
        )
    else:
        logger.info("No WEBHOOK_URL — falling back to polling (local dev mode)")
        app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()