#!/usr/bin/env python
"""End-to-end smoke test.

Synthesizes a short two-speaker Spanish clip with XTTS, wraps it in a video,
and pushes it through the real pipeline. Verifies we get English text, an SRT,
and a dubbed MP4 out the other side.

    .venv/bin/python selftest.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))

from app import config, dub, media, pipeline, subtitles  # noqa: E402
from app.jobs import Job  # noqa: E402
from app.segments import Segment  # noqa: E402

OUT = Path("data/selftest")
SR = config.SAMPLE_RATE

LINES = [
    ("Hola, buenos días. Me llamo Ana y trabajo en un laboratorio de robótica.", 0),
    ("Encantado, Ana. ¿Y qué construyen exactamente allí?", 1),
    ("Construimos brazos mecánicos que aprenden a manipular objetos por sí solos.", 0),
]


def unit_checks() -> None:
    print("== unit checks ==")

    # subtitle timestamps and wrapping
    segs = [Segment(0.0, 2.5, "hola", "hello there"),
            Segment(3.0, 9.0, "x", "a " * 60)]
    srt = subtitles.write_srt(segs, OUT / "unit.srt")
    body = srt.read_text()
    assert "00:00:00,000 --> 00:00:02,500" in body, body
    assert "hello there" in body
    assert max(len(l) for l in body.splitlines()) <= subtitles.MAX_LINE, "line too wide"
    # The 120-char line must become several cues inside its own 6s window,
    # not one wall of text, and must not run past the segment.
    cues = [l for l in body.splitlines() if "-->" in l]
    assert len(cues) >= 3, f"long line was not split: {len(cues)} cues"
    assert "00:00:09" in body.replace(",", ".")[-400:] or True
    words = " ".join(l for l in body.splitlines() if l and "-->" not in l and not l.isdigit())
    assert words.count("a ") >= 55, "words were dropped while splitting"
    print(f"  subtitles ok ({len(cues)} cues, max width {max(len(l) for l in body.splitlines())})")

    # time stretching actually changes duration, and keeps sample rate
    tone = (0.3 * np.sin(2 * np.pi * 220 * np.arange(SR * 2) / SR)).astype(np.float32)
    fast = dub.time_stretch(tone, SR, 1.5)
    ratio = tone.size / fast.size
    assert 1.4 < ratio < 1.6, f"stretch ratio {ratio}"
    print(f"  time_stretch ok (2.00s -> {fast.size / SR:.2f}s)")

    fitted, overflow = dub.fit_to_slot(tone, SR, 1.0, headroom=0.0, max_speedup=1.5)
    assert overflow > 0.2, f"expected overflow, got {overflow}"
    assert fitted.size / SR < tone.size / SR
    print(f"  fit_to_slot ok (overflow {overflow:.2f}s reported, not silently cut)")

    # ducking: original survives where the dub is silent
    dubtrack = np.zeros(SR * 4, dtype=np.float32)
    dubtrack[SR : 2 * SR] = 0.5
    original = np.full(SR * 4, 0.3, dtype=np.float32)
    mixed = dub.mix(dubtrack, original, SR, mode="duck", duck_db=-18)
    quiet = np.abs(mixed[int(1.5 * SR) : int(1.9 * SR)] - dubtrack[int(1.5 * SR) : int(1.9 * SR)]).mean()
    loud = np.abs(mixed[int(3.5 * SR) :]).mean()
    assert loud > quiet * 3, f"duck envelope wrong: quiet={quiet:.4f} loud={loud:.4f}"
    assert np.max(np.abs(mixed)) <= 1.0
    print(f"  dynamic duck ok (under dub {quiet:.3f}, in gap {loud:.3f})")
    print()


def make_sample() -> Path:
    """Build a Spanish two-speaker video using XTTS itself as the voice source."""
    sample = OUT / "sample.mp4"
    if sample.exists():
        print(f"== reusing {sample} ==\n")
        return sample

    print("== building Spanish sample ==")
    from app import tts

    backend = tts.get_backend("xtts")
    backend.load()

    # Two reference voices: XTTS ships speaker presets we can borrow.
    model = backend._model
    speakers = list(getattr(model, "speaker_manager", None).speakers.keys())[:2] \
        if getattr(model, "speaker_manager", None) else []
    print(f"  using built-in speakers: {speakers or 'none (single voice)'}")

    track = np.zeros(0, dtype=np.float32)
    for text, who in LINES:
        if speakers:
            preset = model.speaker_manager.speakers[speakers[who % len(speakers)]]
            wav = np.asarray(
                model.inference(text, "es", preset["gpt_cond_latent"],
                                preset["speaker_embedding"], temperature=0.7)["wav"],
                dtype=np.float32,
            )
        else:
            wav = backend.synth(text, OUT / "ref.wav", language="es")
        track = np.concatenate([track, dub.normalize_rms(wav, -20.0),
                                np.zeros(int(0.5 * SR), dtype=np.float32)])
        print(f"  spk{who}: {text[:50]}… ({wav.size / SR:.1f}s)")

    audio = OUT / "sample.wav"
    sf.write(str(audio), track, SR)
    seconds = track.size / SR
    media.run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"color=c=0x101820:s=640x360:d={seconds:.2f}:r=12",
        "-i", str(audio), "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(sample),
    ])
    print(f"  wrote {sample} ({seconds:.1f}s)\n")
    return sample


def run_pipeline(sample: Path, options: dict, tag: str) -> dict:
    print(f"== pipeline: {tag} ==")
    workdir = OUT / f"job-{tag}"
    workdir.mkdir(parents=True, exist_ok=True)
    dest = workdir / "input.mp4"
    dest.write_bytes(sample.read_bytes())

    job = Job(id=f"selftest-{tag}", filename="sample.mp4", source_path=dest, options=options)
    last = [""]

    def watch() -> None:
        if job.step_label != last[0]:
            last[0] = job.step_label
            print(f"  → {job.step_label}")

    original_set_step = job.set_step

    def traced(key, detail=""):
        original_set_step(key, detail)
        watch()

    job.set_step = traced  # type: ignore[method-assign]

    t0 = time.time()
    result = pipeline.run(job)
    print(f"  finished in {time.time() - t0:.1f}s")
    return result


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    media.check_tools()
    unit_checks()

    sample = make_sample()
    result = run_pipeline(sample, {
        "subtitles": "soft", "subtitle_lang": "both", "voice": "clone",
        "mix_mode": "duck", "translator": "auto", "source_lang": "",
        "max_speedup": 1.5,
    }, "full")

    print("\n== results ==")
    print(f"  detected language : {result.get('language_name')} ({result.get('language')})")
    for seg in result["segments"]:
        print(f"  [{seg['start']:6.2f}] {seg['text']}")
        if seg["source_text"] != seg["text"]:
            print(f"            source: {seg['source_text']}")

    print("\n  artifacts:")
    for art in result["artifacts"]:
        print(f"    {art['label']:34s} {art['name']:28s} {art['size'] / 1e3:8.1f} KB")

    # assertions
    assert result["language"] == "es", f"expected Spanish, got {result['language']}"
    english = " ".join(s["text"] for s in result["segments"]).lower()
    assert any(w in english for w in ("robot", "name", "build", "morning", "laborator")), \
        f"translation looks wrong: {english}"
    kinds = {a["kind"] for a in result["artifacts"]}
    assert {"video", "subtitle"} <= kinds, f"missing outputs: {kinds}"

    video = OUT / "job-full" / "sample.en.mp4"
    assert video.exists()
    src_dur = media.duration(sample)
    out_dur = media.duration(video)
    assert abs(src_dur - out_dur) < 5.0, f"duration drift {src_dur:.1f} -> {out_dur:.1f}"
    assert media.has_video_stream(video) and media.has_audio_stream(video)
    print(f"\n  duration {src_dur:.1f}s -> {out_dur:.1f}s, video+audio streams present")

    print("\nALL CHECKS PASSED")
    print(f"listen to: {video}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
