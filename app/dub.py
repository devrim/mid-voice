"""Assembling the dub track: fitting speech into its time slot, then mixing.

Two problems that make or break a dub:

1. English is rarely the same length as the source line. We speed the synthesized
   clip up (pitch-preserving, capped) to fit its slot, and let it spill into the
   following silence when speeding up isn't enough.
2. Flatly ducking the original for the whole runtime kills the music and room
   tone. Instead we duck *dynamically* — the original drops only while the dub
   is actually speaking, and comes back up in the gaps.
"""

from __future__ import annotations

import subprocess

import numpy as np

EPS = 1e-9


# --- gain helpers ----------------------------------------------------------
def db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def rms(wav: np.ndarray) -> float:
    if wav.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(wav, dtype=np.float64))))


def normalize_rms(wav: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    level = rms(wav)
    if level < EPS:
        return wav
    gain = db_to_gain(target_db) / level
    return np.clip(wav * gain, -1.0, 1.0).astype(np.float32)


def peak_limit(wav: np.ndarray, ceiling_db: float = -1.0) -> np.ndarray:
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    ceiling = db_to_gain(ceiling_db)
    if peak > ceiling:
        wav = wav * (ceiling / peak)
    return wav.astype(np.float32)


def fade_edges(wav: np.ndarray, sr: int, ms: float = 8.0) -> np.ndarray:
    n = min(int(sr * ms / 1000.0), wav.size // 2)
    if n <= 0:
        return wav
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    wav = wav.copy()
    wav[:n] *= ramp
    wav[-n:] *= ramp[::-1]
    return wav


# --- time stretching -------------------------------------------------------
def _atempo_chain(factor: float) -> list[str]:
    """ffmpeg's atempo is happiest between 0.5 and 2.0 — chain beyond that."""
    factors = []
    remaining = factor
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return [f"atempo={f:.6f}" for f in factors]


def time_stretch(wav: np.ndarray, sr: int, factor: float) -> np.ndarray:
    """Speed up (factor > 1) or slow down without changing pitch."""
    if wav.size == 0 or abs(factor - 1.0) < 0.01:
        return wav
    args = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
        "-filter:a", ",".join(_atempo_chain(factor)),
        "-f", "f32le", "-ar", str(sr), "-ac", "1", "pipe:1",
    ]
    proc = subprocess.run(args, input=wav.astype(np.float32).tobytes(), capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        return wav  # stretching is a nicety; never fail the render over it
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def fit_to_slot(
    wav: np.ndarray,
    sr: int,
    slot: float,
    *,
    headroom: float = 0.0,
    max_speedup: float = 1.5,
) -> tuple[np.ndarray, float]:
    """Squeeze `wav` toward `slot` seconds.

    `headroom` is extra silence available after the slot (the gap before the
    next line) that we're allowed to run into. Returns the fitted audio and the
    overflow in seconds that still didn't fit.
    """
    if wav.size == 0 or slot <= 0:
        return wav, 0.0

    have = wav.size / sr
    budget = slot + max(0.0, headroom)
    if have <= budget:
        return wav, 0.0

    needed = have / budget
    factor = min(needed, max_speedup)
    stretched = time_stretch(wav, sr, factor)
    overflow = max(0.0, stretched.size / sr - budget)
    return stretched, overflow


# --- assembly --------------------------------------------------------------
def place(canvas: np.ndarray, clip: np.ndarray, start_sample: int) -> None:
    """Add `clip` into `canvas` at a sample offset, in place, clipping at the end."""
    if clip.size == 0:
        return
    start = max(0, start_sample)
    end = min(canvas.size, start + clip.size)
    if end <= start:
        return
    canvas[start:end] += clip[: end - start]


def speech_envelope(
    track: np.ndarray,
    sr: int,
    *,
    duck_db: float,
    attack_ms: float = 120.0,
    release_ms: float = 400.0,
    threshold: float = 1e-3,
) -> np.ndarray:
    """Gain curve for the original audio: ducked while the dub speaks, 1.0 elsewhere.

    Computed on ~10ms blocks and then smoothed, so it's cheap even for a
    feature-length file.
    """
    floor = db_to_gain(duck_db)
    block = max(1, int(sr * 0.01))
    blocks = int(np.ceil(track.size / block))
    padded = np.zeros(blocks * block, dtype=np.float32)
    padded[: track.size] = np.abs(track)
    peaks = padded.reshape(blocks, block).max(axis=1)

    target = np.where(peaks > threshold, floor, 1.0).astype(np.float32)

    # One-pole smoothing, asymmetric: duck fast, recover slowly.
    a_att = np.exp(-block / max(1.0, sr * attack_ms / 1000.0))
    a_rel = np.exp(-block / max(1.0, sr * release_ms / 1000.0))
    smooth = np.empty_like(target)
    value = 1.0
    for i, t in enumerate(target):
        coeff = a_att if t < value else a_rel
        value = coeff * value + (1.0 - coeff) * t
        smooth[i] = value

    envelope = np.repeat(smooth, block)[: track.size]
    return envelope.astype(np.float32)


def mix(
    dub: np.ndarray,
    original: np.ndarray,
    sr: int,
    *,
    mode: str = "duck",
    duck_db: float = -18.0,
) -> np.ndarray:
    """Combine the dub with the original audio bed.

    * ``duck``    — original ducks under the dub, full level in the gaps.
    * ``overlay`` — original stays clearly audible underneath (voice-over style).
    * ``replace`` — original removed entirely.
    """
    n = max(dub.size, original.size)
    dub = np.pad(dub, (0, n - dub.size))
    original = np.pad(original, (0, n - original.size))

    if mode == "replace":
        bed = np.zeros_like(original)
    elif mode == "overlay":
        bed = original * speech_envelope(dub, sr, duck_db=-8.0)
    else:
        bed = original * speech_envelope(dub, sr, duck_db=duck_db)

    return peak_limit(dub + bed)
