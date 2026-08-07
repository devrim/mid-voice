# Installing mid-voice

## Requirements

- **Python 3.10–3.12.** Not 3.13 — coqui-tts doesn't build for it yet.
- **ffmpeg and ffprobe** on `PATH`. All media I/O goes through them.
- **~12 GB of disk** for model weights, plus ~8 GB for the wheels.
- **An NVIDIA GPU is strongly recommended.** 8 GB of VRAM is comfortable with
  the default models. CPU works and is roughly 10–20× slower.

## Quick install

```bash
git clone https://github.com/devrim/mid-voice.git
cd mid-voice

sudo apt install ffmpeg                 # or: brew install ffmpeg

uv venv --python 3.12
uv pip install -e ".[tts,mt,cuda]"

./run.sh                                # → http://127.0.0.1:7860
```

Using plain pip instead of [uv](https://docs.astral.sh/uv/):

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[tts,mt,cuda]"
.venv/bin/python -m app.main
```

The first translation downloads ~10 GB of weights and will look like it's
hanging. Watch the terminal — the downloads print progress there, not in the
browser.

## What each extra pulls in

| extra | gives you | leaving it out |
|---|---|---|
| `tts` | XTTS-v2 voice cloning | subtitles-only mode still works |
| `mt` | NLLB-200 translation | falls back to Whisper's translate task |
| `cuda` | CUDA 12 runtime for Whisper | Whisper drops to CPU |

Subtitles-only, no GPU, smallest footprint:

```bash
uv pip install -e .          # core only: fastapi + faster-whisper + ffmpeg
```

## Verifying the install

```bash
.venv/bin/python selftest.py
```

This synthesizes a two-speaker Spanish clip, runs it through the real pipeline,
and asserts that the output is English, that subtitles are written, and that
the muxed MP4 has both streams. It ends with `ALL CHECKS PASSED`. First run
takes a few minutes because of the downloads.

The capability chips at the top of the web page show the same information at a
glance — green means available, amber means missing.

## Known-good versions

Verified on Ubuntu 24.04, RTX 4090, driver 580.159.03:

```
python 3.12.3        torch 2.13.0+cu130     coqui-tts 0.27.5
ffmpeg 6.1.1         torchaudio 2.11.0      transformers 4.57.6
                     torchcodec 0.15.0      faster-whisper 1.2.1
                     nvidia-cublas-cu12 12.9.2.10    ctranslate2 4.8.1
                     nvidia-cudnn-cu12 9.24.0.43
```

## Troubleshooting

### `Library libcublas.so.12 is not found or cannot be loaded`

ctranslate2, which runs Whisper, is built against CUDA 12 and looks up
`libcublas.so.12` by soname. Current torch wheels ship the CUDA **13** runtime
(`libcublas.so.13`), which does not satisfy that. Installing torch is therefore
not enough:

```bash
uv pip install -e ".[cuda]"     # nvidia-cublas-cu12 + nvidia-cudnn-cu12
```

`app/asr.py` dlopens those libraries at startup so ctranslate2 finds them
already resident. If they're missing, Whisper logs a warning and falls back to
CPU rather than crashing — so check the terminal if transcription is
unexpectedly slow.

### `cannot import name 'isin_mps_friendly' from 'transformers.pytorch_utils'`

transformers 5.x removed that function; coqui-tts still imports it. Pin to the
4.x line:

```bash
uv pip install "transformers>=4.57,<5"
```

The `tts` and `mt` extras both carry this constraint, so a clean install gets
it right. You only hit this if something later upgraded transformers.

### `From Pytorch 2.9, the torchcodec library is required for audio IO`

```bash
uv pip install torchcodec
```

Included in the `tts` extra. torchcodec needs ffmpeg 4–7 present at import time.

### CUDA out of memory

Whisper `large-v3` and NLLB-1.3B together want ~10 GB. Shrink either:

```bash
export MIDVOICE_WHISPER_MODEL=medium                      # or distil-large-v3
export MIDVOICE_NLLB_MODEL=facebook/nllb-200-distilled-600M
```

XTTS is loaded lazily and released between jobs, so it rarely collides with
the other two.

### Running on CPU only

```bash
export MIDVOICE_WHISPER_DEVICE=cpu
export MIDVOICE_WHISPER_COMPUTE=int8
export MIDVOICE_WHISPER_MODEL=small
```

Expect several minutes of processing per minute of video. XTTS on CPU is slow
enough that subtitles-only is usually the better call.

### Port already in use

```bash
MIDVOICE_PORT=8080 ./run.sh
```

### Exposing it beyond localhost

```bash
MIDVOICE_HOST=0.0.0.0 ./run.sh
```

There is **no authentication**. Anyone who can reach the port can upload files,
consume your GPU, and download every job's output. Only do this on a network
you trust, or put a reverse proxy with auth in front of it.

## Uninstalling

```bash
rm -rf .venv data                       # the app and its jobs
rm -rf ~/.local/share/tts               # XTTS weights, ~1.8 GB
rm -rf ~/.cache/huggingface/hub/models--Systran--faster-whisper-large-v3
rm -rf ~/.cache/huggingface/hub/models--facebook--nllb-200-distilled-1.3B
```
