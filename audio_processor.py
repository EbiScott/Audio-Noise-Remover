"""
audio_processor.py
Two noise-reduction backends:
  1. clean_audio_simple  – fast spectral gating via noisereduce
  2. clean_audio_ai      – deep learning via speechbrain / demucs
"""

import logging
import subprocess
import tempfile
import os

import numpy as np
import soundfile as sf
import librosa

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _convert_to_wav(input_path: str) -> str:
    """Convert any audio format to a temporary 16-bit PCM WAV using ffmpeg."""
    tmp_wav = tempfile.mktemp(suffix="_input.wav")
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ar", "16000",      # 16 kHz – good for speech
            "-ac", "1",          # mono
            "-sample_fmt", "s16",
            tmp_wav,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr}")
    return tmp_wav


# ──────────────────────────────────────────────
# Method 1 – Fast spectral gating (noisereduce)
# ──────────────────────────────────────────────

def clean_audio_simple(input_path: str, output_path: str) -> None:
    """
    Fast noise reduction using the `noisereduce` library (spectral gating).
    Estimates noise from the first 0.5 s of audio.
    """
    import noisereduce as nr

    logger.info(f"[simple] Converting {input_path}")
    wav_path = _convert_to_wav(input_path)

    try:
        audio, sr = librosa.load(wav_path, sr=None, mono=True)

        # Use first 500 ms as noise profile
        noise_clip_len = int(sr * 0.5)
        noise_clip = audio[:noise_clip_len] if len(audio) > noise_clip_len else audio

        logger.info(f"[simple] Running noisereduce (sr={sr}, duration={len(audio)/sr:.1f}s)")
        reduced = nr.reduce_noise(
            y=audio,
            sr=sr,
            y_noise=noise_clip,
            prop_decrease=0.85,
            stationary=False,
        )

        sf.write(output_path, reduced, sr, subtype="PCM_16")
        logger.info(f"[simple] Saved to {output_path}")

    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


# ──────────────────────────────────────────────
# Method 2 – AI / deep learning (demucs)
# ──────────────────────────────────────────────

def clean_audio_ai(input_path: str, output_path: str) -> None:
    """
    High-quality noise reduction using Facebook's Demucs model
    (htdemucs_ft trained for speech/voice enhancement).

    Falls back to enhanced spectral gating if demucs is unavailable.
    """
    try:
        _clean_with_demucs(input_path, output_path)
    except Exception as e:
        logger.warning(f"[ai] Demucs failed ({e}), falling back to enhanced spectral gating")
        _clean_with_enhanced_spectral(input_path, output_path)


def _clean_with_demucs(input_path: str, output_path: str) -> None:
    """Run demucs htdemucs model to separate vocals from noise."""
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    wav_path = _convert_to_wav(input_path)

    try:
        logger.info("[ai/demucs] Loading model…")
        model = get_model("htdemucs")
        model.eval()

        audio, sr = librosa.load(wav_path, sr=model.samplerate, mono=False)
        if audio.ndim == 1:
            audio = np.stack([audio, audio])  # stereo expected by demucs

        wav_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            sources = apply_model(model, wav_tensor, overlap=0.1, progress=False)

        # sources shape: (batch, stems, channels, time)
        # stem index 3 = "vocals" in htdemucs
        vocals = sources[0, 3].mean(dim=0).numpy()

        sf.write(output_path, vocals, model.samplerate, subtype="PCM_16")
        logger.info(f"[ai/demucs] Saved to {output_path}")

    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


def _clean_with_enhanced_spectral(input_path: str, output_path: str) -> None:
    """
    Fallback: multi-pass spectral gating with Wiener filter smoothing.
    Better than the simple method but no heavy ML dependency needed.
    """
    import noisereduce as nr
    from scipy.signal import wiener

    wav_path = _convert_to_wav(input_path)

    try:
        audio, sr = librosa.load(wav_path, sr=None, mono=True)

        # Pass 1 – stationary noise
        reduced = nr.reduce_noise(y=audio, sr=sr, stationary=True, prop_decrease=0.75)

        # Pass 2 – non-stationary noise
        reduced = nr.reduce_noise(y=reduced, sr=sr, stationary=False, prop_decrease=0.60)

        # Wiener filter for residual smoothing
        reduced = wiener(reduced, mysize=5).astype(np.float32)

        # Normalize
        peak = np.max(np.abs(reduced))
        if peak > 0:
            reduced = reduced / peak * 0.95

        sf.write(output_path, reduced, sr, subtype="PCM_16")
        logger.info(f"[ai/fallback] Saved to {output_path}")

    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)
