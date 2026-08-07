"""Runtime configuration. Everything is overridable by environment variable."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("MIDVOICE_DATA", ROOT / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
JOBS_DIR = DATA_DIR / "jobs"

for _d in (UPLOAD_DIR, JOBS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

HOST = os.environ.get("MIDVOICE_HOST", "127.0.0.1")
PORT = int(os.environ.get("MIDVOICE_PORT", "7860"))

# --- ASR -------------------------------------------------------------------
# large-v3 fits comfortably on a 24GB card. Drop to "medium"/"small" on less.
WHISPER_MODEL = os.environ.get("MIDVOICE_WHISPER_MODEL", "large-v3")
WHISPER_DEVICE = os.environ.get("MIDVOICE_WHISPER_DEVICE", "auto")
WHISPER_COMPUTE = os.environ.get("MIDVOICE_WHISPER_COMPUTE", "float16")

# --- MT --------------------------------------------------------------------
NLLB_MODEL = os.environ.get("MIDVOICE_NLLB_MODEL", "facebook/nllb-200-distilled-1.3B")

# --- TTS -------------------------------------------------------------------
XTTS_MODEL = os.environ.get("MIDVOICE_XTTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
# Coqui's downloader asks for CPML licence agreement on stdin; we run headless.
os.environ.setdefault("COQUI_TOS_AGREED", "1")

# --- Dubbing ---------------------------------------------------------------
# How much we're willing to speed up synthesized speech to fit its time slot.
MAX_SPEEDUP = float(os.environ.get("MIDVOICE_MAX_SPEEDUP", "1.5"))
# Level the original audio is ducked to underneath the dub, in dB.
DUCK_DB = float(os.environ.get("MIDVOICE_DUCK_DB", "-18"))

SAMPLE_RATE = 24000  # XTTS native output rate; the mix runs at this rate.
