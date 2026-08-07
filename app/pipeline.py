"""The translation pipeline: video in, dubbed video (+ subtitles) out."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

from . import asr, config, dub, langs, media, subtitles, translate, tts
from .jobs import Job
from .segments import Segment, Transcript

log = logging.getLogger(__name__)

# Reference clips for voice cloning.
REF_MAX = 8.0     # seconds of reference audio XTTS gets per speaker turn
REF_MIN = 1.2     # below this a clip is too short to condition on
TURN_GAP = 1.2    # a pause longer than this probably means a new speaker turn


def _safe_stem(filename: str) -> str:
    """Output names derived from the upload, minus anything that would need
    escaping in an ffmpeg filtergraph (the subtitle burn-in path) or a URL."""
    stem = Path(filename).stem
    cleaned = "".join(c if (c.isalnum() or c in " ._-") else "_" for c in stem).strip(" .")
    return (cleaned or "output")[:80]


# --- reference clips -------------------------------------------------------
def _group_turns(segments: list[Segment]) -> list[list[int]]:
    """Group consecutive segments into likely speaker turns, by pause length."""
    groups: list[list[int]] = []
    for i, seg in enumerate(segments):
        if groups and seg.start - segments[groups[-1][-1]].end <= TURN_GAP:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def build_reference_clips(
    segments: list[Segment], audio: Path, workdir: Path, total: float
) -> list[Path]:
    """One clone-reference clip per speaker turn, mapped back onto every segment.

    Cutting the reference from the same point in the timeline as the line being
    dubbed means a two-hander keeps two distinct voices without diarization.
    """
    refs_dir = workdir / "refs"
    refs_dir.mkdir(exist_ok=True)
    groups = _group_turns(segments)

    # A global fallback: the longest turn in the file, for turns too short to use.
    longest = max(groups, key=lambda g: segments[g[-1]].end - segments[g[0]].start)
    g_start = segments[longest[0]].start
    g_len = min(REF_MAX, max(REF_MIN, segments[longest[-1]].end - g_start))
    fallback = media.slice_audio(
        audio, refs_dir / "global.wav", g_start, g_len, config.SAMPLE_RATE
    )

    per_segment: list[Path] = [fallback] * len(segments)
    for gi, group in enumerate(groups):
        start = segments[group[0]].start
        end = segments[group[-1]].end
        length = min(REF_MAX, end - start)
        if length < REF_MIN:
            # Too brief on its own — borrow a little of what follows.
            length = min(REF_MAX, min(3.0, total - start))
        if length < REF_MIN:
            continue  # keep the fallback for this turn
        clip = media.slice_audio(
            audio, refs_dir / f"turn{gi:04d}.wav", start, length, config.SAMPLE_RATE
        )
        for idx in group:
            per_segment[idx] = clip
    return per_segment


# --- steps -----------------------------------------------------------------
def _transcribe_and_translate(job: Job, audio: Path, total: float) -> Transcript:
    opts = job.options
    forced = (opts.get("source_lang") or "").strip() or None
    backend = opts.get("translator", "auto")
    want_source_subs = opts.get("subtitles") != "none" and opts.get("subtitle_lang") == "both"

    job.set_step("transcribe", "detecting language")
    transcript = asr.transcribe(
        audio, task="transcribe", language=forced, total_duration=total,
        progress=lambda f, t: job.set_progress(f, t),
    )
    source = transcript.language
    job.log(f"Detected {langs.name(source)} ({transcript.language_probability:.0%} confident), "
            f"{len(transcript.segments)} segments")

    if not transcript.segments:
        raise RuntimeError("No speech found in this file.")

    job.set_step("translate")
    if source == "en":
        job.log("Source is already English — skipping translation")
        for seg in transcript.segments:
            seg.text = seg.source_text
        job.set_progress(1.0)
        return transcript

    if backend == "auto":
        backend = "nllb" if langs.to_nllb(source) else "whisper"

    if backend == "nllb":
        try:
            translate.translate_nllb(
                transcript.segments, source,
                progress=lambda f, t: job.set_progress(f, t),
            )
            job.log(f"Translated {len(transcript.segments)} segments with NLLB-200")
        except translate.TranslationError as exc:
            job.log(f"NLLB unavailable ({exc}) — falling back to Whisper translate")
            backend = "whisper"

    if backend == "whisper":
        english = asr.transcribe(
            audio, task="translate", language=source, total_duration=total,
            progress=lambda f, t: job.set_progress(f, t),
        )
        if want_source_subs:
            # Whisper's translate pass has its own segmentation, so keep the
            # transcribe pass around for the source-language subtitle track.
            source_only = transcript.segments
            transcript = english
            transcript.language = source
            job.result["source_segments"] = [s.to_dict() for s in source_only]
            job._source_segments = source_only  # type: ignore[attr-defined]
        else:
            transcript = english
            transcript.language = source
        job.log(f"Translated with Whisper ({len(transcript.segments)} segments)")

    for seg in transcript.segments:
        seg.text = translate.clean(seg.text)
    job.set_progress(1.0)
    return transcript


def _synthesize(job: Job, transcript: Transcript, ref_audio: Path, total: float) -> np.ndarray:
    sr = config.SAMPLE_RATE
    segments = [s for s in transcript.segments if s.text.strip()]
    job.set_step("synthesize", f"{len(segments)} lines")
    if not segments:
        # Whisper found speech but translation produced nothing usable.
        job.log("Nothing to synthesize — keeping the original audio")
        return np.zeros(int(np.ceil((total + 1.0) * sr)), dtype=np.float32)

    backend = tts.get_backend("xtts")
    backend.load()
    refs = build_reference_clips(segments, ref_audio, job.source_path.parent, total)

    canvas = np.zeros(int(np.ceil((total + 5.0) * sr)), dtype=np.float32)
    max_speedup = float(job.options.get("max_speedup") or config.MAX_SPEEDUP)
    overflow_total = 0.0

    for i, seg in enumerate(segments):
        job.check_cancelled()
        gap = (segments[i + 1].start - seg.end) if i + 1 < len(segments) else 3.0
        headroom = max(0.0, gap - 0.15)  # leave a beat before the next line

        try:
            wav = backend.synth(seg.text, refs[i], language="en")
        except Exception as exc:  # noqa: BLE001 — one bad line shouldn't kill the render
            job.log(f"line {i + 1} failed to synthesize ({exc}); leaving silence")
            continue

        wav = dub.normalize_rms(wav, target_db=-20.0)
        wav, overflow = dub.fit_to_slot(
            wav, sr, seg.duration, headroom=headroom, max_speedup=max_speedup
        )
        overflow_total += overflow
        dub.place(canvas, dub.fade_edges(wav, sr), int(seg.start * sr))

        job.set_progress((i + 1) / len(segments), seg.text[:120])

    if overflow_total > 0.5:
        job.log(f"{overflow_total:.1f}s of speech ran past its slot even after speed-up")
    return canvas


# --- entry point -----------------------------------------------------------
def run(job: Job) -> dict:
    opts = job.options
    workdir = job.source_path.parent
    src = job.source_path
    stem = _safe_stem(job.filename)
    artifacts: list[dict] = []

    def publish(path: Path, kind: str, label: str) -> None:
        artifacts.append({
            "name": path.name, "kind": kind, "label": label,
            "size": path.stat().st_size,
            "url": f"/api/jobs/{job.id}/files/{path.name}",
        })

    # 1. probe
    job.set_step("probe")
    media.check_tools()
    if not media.has_audio_stream(src):
        raise RuntimeError("This file has no audio track.")
    total = media.duration(src)
    has_video = media.has_video_stream(src)
    job.log(f"{total:.1f}s, {'video' if has_video else 'audio only'}")
    job.set_progress(1.0)

    # 2. audio extraction — 16k mono for Whisper, 24k for cloning and the mix
    job.set_step("extract")
    asr_wav = media.extract_audio(src, workdir / "asr.wav", 16000)
    ref_wav = media.extract_audio(src, workdir / "source.wav", config.SAMPLE_RATE)
    job.set_progress(1.0)

    # 3. transcribe + translate
    transcript = _transcribe_and_translate(job, asr_wav, total)
    job.result["language"] = transcript.language
    job.result["language_name"] = langs.name(transcript.language)
    job.result["segments"] = [s.to_dict() for s in transcript.segments]

    # 4. subtitles
    if opts.get("subtitles", "soft") != "none":
        en_srt = subtitles.write_srt(transcript.segments, workdir / f"{stem}.en.srt")
        subtitles.write_vtt(transcript.segments, workdir / f"{stem}.en.vtt")
        publish(en_srt, "subtitle", "English subtitles (.srt)")
        if opts.get("subtitle_lang") == "both" and transcript.language != "en":
            source_segments = getattr(job, "_source_segments", transcript.segments)
            src_srt = subtitles.write_srt(
                source_segments, workdir / f"{stem}.{transcript.language}.srt", use_source=True
            )
            publish(src_srt, "subtitle", f"{langs.name(transcript.language)} subtitles (.srt)")

    # 5. voice + mix
    voice = opts.get("voice", "clone")
    dubbed_path: Path | None = None
    if voice != "none":
        track = _synthesize(job, transcript, ref_wav, total)

        job.set_step("mix")
        original, _ = sf.read(str(ref_wav), dtype="float32", always_2d=False)
        if original.ndim > 1:
            original = original.mean(axis=1)
        mixed = dub.mix(
            track, original, config.SAMPLE_RATE,
            mode=opts.get("mix_mode", "duck"), duck_db=config.DUCK_DB,
        )
        dubbed_path = workdir / "dubbed.wav"
        sf.write(str(dubbed_path), mixed, config.SAMPLE_RATE)
        job.set_progress(1.0)
    else:
        job.log("Voice disabled — subtitles only")

    # 6. mux
    job.set_step("mux")
    sub_for_video = workdir / f"{stem}.en.srt"
    sub_mode = opts.get("subtitles", "soft")
    audio_track = dubbed_path or ref_wav

    if has_video:
        out = workdir / f"{stem}.en.mp4"
        media.mux(
            src, audio_track, out,
            subtitles=sub_for_video if sub_mode in ("soft", "burn") and sub_for_video.exists() else None,
            burn_in=(sub_mode == "burn"),
            sub_style="FontSize=22,OutlineColour=&H80000000,BorderStyle=3",
        )
        publish(out, "video", "Translated video")
    else:
        out = workdir / f"{stem}.en.m4a"
        media.audio_only_output(audio_track, out)
        publish(out, "audio", "Translated audio")

    if dubbed_path:
        audio_out = workdir / f"{stem}.en.m4a"
        if audio_out != out:
            media.audio_only_output(dubbed_path, audio_out)
            publish(audio_out, "audio", "Dubbed audio only")

    job.set_progress(1.0)
    shutil.rmtree(workdir / "refs", ignore_errors=True)
    for tmp in (asr_wav, dubbed_path):
        if tmp and tmp.exists():
            tmp.unlink()

    job.result["artifacts"] = artifacts
    job.result["primary"] = artifacts[0]["url"] if artifacts else None
    return job.result
