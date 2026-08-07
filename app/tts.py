"""Text-to-speech with zero-shot voice cloning (XTTS-v2).

The interesting trick here is the reference clip. Rather than diarizing the
video and building a voice per speaker, we cut a short clip out of the *original
audio at the position of each segment* and use that as the clone reference. The
person talking at 04:12 in the dub is conditioned on whoever was talking at
04:12 in the source, so multi-speaker videos keep their speakers apart without a
diarization model. Conditioning latents are cached per reference clip, so
consecutive segments from the same speaker cost nothing extra.
"""

from __future__ import annotations

import contextlib
import logging
import re
import threading
from pathlib import Path

import numpy as np

from . import config

log = logging.getLogger(__name__)


class TTSError(RuntimeError):
    pass


@contextlib.contextmanager
def _unsafe_torch_load():
    """torch>=2.6 defaults weights_only=True, which rejects XTTS checkpoints.

    We're loading a model we just downloaded from Coqui's own release bucket, so
    full unpickling is acceptable here.
    """
    import torch  # noqa: PLC0415

    original = torch.load

    def patched(*args, **kwargs):
        kwargs["weights_only"] = False
        return original(*args, **kwargs)

    torch.load = patched
    try:
        yield
    finally:
        torch.load = original


_SENTENCE = re.compile(r"(?<=[.!?…])\s+|(?<=[。！？])")


def _chunk(text: str, limit: int = 220) -> list[str]:
    """XTTS degrades past ~250 characters, so feed it sentence-sized pieces."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return [text] if text else []

    chunks, current = [], ""
    for piece in _SENTENCE.split(text):
        piece = (piece or "").strip()
        if not piece:
            continue
        if not current:
            current = piece
        elif len(current) + 1 + len(piece) <= limit:
            current = f"{current} {piece}"
        else:
            chunks.append(current)
            current = piece
    if current:
        chunks.append(current)

    # A single sentence longer than the limit still needs breaking up.
    out: list[str] = []
    for chunk in chunks:
        while len(chunk) > limit:
            cut = chunk.rfind(" ", 0, limit)
            cut = cut if cut > limit // 2 else limit
            out.append(chunk[:cut].strip())
            chunk = chunk[cut:].strip()
        if chunk:
            out.append(chunk)
    return out


class XTTS:
    """Wrapper around Coqui XTTS-v2 with cached speaker conditioning."""

    sample_rate = 24000
    supports_cloning = True

    def __init__(self) -> None:
        self._model = None
        self._api = None
        self._device = "cpu"
        self._latents: dict[str, tuple] = {}
        self._lock = threading.Lock()

    # -- loading ------------------------------------------------------------
    def load(self) -> None:
        if self._model is not None or self._api is not None:
            return
        with self._lock:
            if self._model is not None or self._api is not None:
                return
            try:
                import torch  # noqa: PLC0415
                from TTS.api import TTS as CoquiTTS  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - env dependent
                raise TTSError(
                    "Voice cloning needs coqui-tts. "
                    'Install with: uv pip install -e ".[tts]" — '
                    "or pick the 'Subtitles only' voice option."
                ) from exc

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info("loading XTTS-v2 on %s (first run downloads ~1.8GB)", self._device)
            with _unsafe_torch_load():
                api = CoquiTTS(config.XTTS_MODEL)
            api.to(self._device)
            self._api = api
            # The low-level model lets us cache conditioning latents; the public
            # API recomputes them on every call.
            self._model = getattr(getattr(api, "synthesizer", None), "tts_model", None)

    # -- speaker conditioning ----------------------------------------------
    def _conditioning(self, reference: Path):
        key = str(reference)
        cached = self._latents.get(key)
        if cached is not None:
            return cached
        latents = self._model.get_conditioning_latents(
            audio_path=[str(reference)], gpt_cond_len=6, max_ref_length=10
        )
        # Keep the cache small — long videos would otherwise pin a lot of VRAM.
        if len(self._latents) > 64:
            self._latents.clear()
        self._latents[key] = latents
        return latents

    # -- synthesis ----------------------------------------------------------
    def synth(self, text: str, reference: Path, language: str = "en") -> np.ndarray:
        """Return mono float32 audio at :attr:`sample_rate`."""
        self.load()
        text = " ".join(text.split())
        if not text:
            return np.zeros(0, dtype=np.float32)

        pieces: list[np.ndarray] = []
        for chunk in _chunk(text):
            if self._model is not None:
                latent, embedding = self._conditioning(reference)
                result = self._model.inference(
                    chunk, language, latent, embedding,
                    temperature=0.7, repetition_penalty=5.0, enable_text_splitting=False,
                )
                wav = np.asarray(result["wav"], dtype=np.float32)
            else:  # fallback: public API, slower but always available
                wav = np.asarray(
                    self._api.tts(text=chunk, speaker_wav=str(reference), language=language),
                    dtype=np.float32,
                )
            pieces.append(wav)
            pieces.append(np.zeros(int(0.08 * self.sample_rate), dtype=np.float32))

        if not pieces:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(pieces[:-1])

    def unload(self) -> None:
        with self._lock:
            self._model = self._api = None
            self._latents.clear()
        with contextlib.suppress(ImportError):
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


_xtts = XTTS()


def get_backend(name: str) -> XTTS:
    if name in ("xtts", "clone"):
        return _xtts
    raise TTSError(f"unknown TTS backend {name!r}")


def unload() -> None:
    _xtts.unload()
