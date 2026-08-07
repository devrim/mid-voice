"""Speech recognition via faster-whisper (CTranslate2)."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

from . import config
from .segments import Segment, Transcript

log = logging.getLogger(__name__)

_model = None
_model_key: tuple | None = None
_lock = threading.Lock()


def _resolve_device() -> tuple[str, str]:
    device, compute = config.WHISPER_DEVICE, config.WHISPER_COMPUTE
    if device == "auto":
        device = "cpu"
        try:
            # ctranslate2 is what actually runs Whisper, so ask it directly —
            # this works even when torch isn't installed.
            import ctranslate2  # noqa: PLC0415

            if ctranslate2.get_cuda_device_count() > 0:
                device = "cuda"
        except Exception:  # noqa: BLE001 — fall through to the torch probe
            try:
                import torch  # noqa: PLC0415

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                pass
    if device == "cpu" and compute in ("float16", "bfloat16"):
        compute = "int8"  # float16 is not supported on CPU backends
    return device, compute


_CUDA_LIBS_LOADED = False

# ctranslate2 dlopen()s these by exact soname. They live inside the
# `nvidia-*-cu12` wheels, which nothing adds to the loader path — and a torch
# built for CUDA 13 ships libcublas.so.13, which does *not* satisfy them.
# Ordered so dependencies land before their dependents.
_NEEDED = (
    "libcublasLt.so.12",
    "libcublas.so.12",
    "libcudnn_engines_precompiled.so.9",
    "libcudnn_engines_runtime_compiled.so.9",
    "libcudnn_heuristic.so.9",
    "libcudnn_ops.so.9",
    "libcudnn_adv.so.9",
    "libcudnn_cnn.so.9",
    "libcudnn_graph.so.9",
    "libcudnn.so.9",
)


def _preload_cuda_libs() -> None:
    """Make the CUDA 12 runtime visible to ctranslate2.

    Importing torch registers whatever CUDA libraries *it* was built against,
    which is enough only when torch and ctranslate2 agree on the CUDA major
    version. When they don't, we dlopen the cu12 wheels' shared objects
    ourselves with RTLD_GLOBAL so ctranslate2's own dlopen finds them already
    resident. Silent no-op if they aren't installed — the caller falls back to
    CPU.
    """
    global _CUDA_LIBS_LOADED
    if _CUDA_LIBS_LOADED:
        return
    _CUDA_LIBS_LOADED = True

    try:
        import torch  # noqa: PLC0415

        torch.cuda.is_available()
    except Exception:  # noqa: BLE001 — best effort only
        pass

    import ctypes  # noqa: PLC0415
    import site  # noqa: PLC0415
    import sysconfig  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    roots = {sysconfig.get_paths()["purelib"], *(site.getsitepackages() or [])}
    index: dict[str, Path] = {}
    for root in roots:
        for lib in Path(root, "nvidia").glob("*/lib/*.so.*"):
            index.setdefault(lib.name, lib)

    for name in _NEEDED:
        path = index.get(name)
        if path is None:
            continue
        try:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:  # pragma: no cover - platform dependent
            log.debug("could not preload %s: %s", name, exc)


def get_model():
    """Load (and cache) the Whisper model. Thread-safe; first call downloads."""
    global _model, _model_key
    from faster_whisper import WhisperModel  # noqa: PLC0415 — heavy import

    device, compute = _resolve_device()
    with _lock:
        key = (config.WHISPER_MODEL, device, compute)
        if _model is not None and _model_key == key:
            return _model
        if device == "cuda":
            _preload_cuda_libs()
        log.info("loading whisper %s on %s (%s)", *key)
        try:
            _model = WhisperModel(config.WHISPER_MODEL, device=device, compute_type=compute)
            _warmup(_model)
        except Exception as exc:  # noqa: BLE001 — CUDA setups fail in many ways
            if device == "cpu":
                raise
            log.warning("GPU Whisper unusable (%s); falling back to CPU int8", exc)
            key = (config.WHISPER_MODEL, "cpu", "int8")
            _model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8")
        _model_key = key
    return _model


def _warmup(model) -> None:
    """Force one real encode.

    Construction is lazy, so a broken CUDA install doesn't surface until the
    first chunk of a user's video — far too late to fall back. A second of
    silence makes it fail here instead, where we can still switch to CPU.
    """
    import numpy as np  # noqa: PLC0415

    silence = np.zeros(16000, dtype=np.float32)
    segments, _ = model.transcribe(silence, language="en", vad_filter=False, beam_size=1)
    for _ in segments:
        break


def transcribe(
    audio: Path,
    *,
    task: str = "transcribe",
    language: str | None = None,
    total_duration: float | None = None,
    progress: Callable[[float, str], None] | None = None,
) -> Transcript:
    """Run Whisper over `audio`.

    task="transcribe" keeps the source language; task="translate" makes Whisper
    emit English directly (single pass, timings already match the English text).
    """
    model = get_model()
    seg_iter, info = model.transcribe(
        str(audio),
        task=task,
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
        condition_on_previous_text=False,  # avoids runaway repetition loops
    )

    total = total_duration or info.duration or 0.0
    out: list[Segment] = []
    for s in seg_iter:
        text = s.text.strip()
        if not text:
            continue
        seg = Segment(start=s.start, end=s.end, source_text=text)
        if task == "translate":
            seg.text = text
        out.append(seg)
        if progress and total:
            progress(min(1.0, s.end / total), text)

    return Transcript(
        language=info.language or language or "unknown",
        language_probability=getattr(info, "language_probability", 0.0) or 0.0,
        segments=out,
    )


def unload() -> None:
    global _model, _model_key
    with _lock:
        _model, _model_key = None, None
