# CosyVoice-CLI

This repository is the upstream **CosyVoice** codebase plus a small, reproducible **command‑line interface** for **CosyVoice3 zero‑shot voice cloning**.

The goal: from an **Anaconda Prompt on Windows**, run a single command with **(text, reference WAV)** and get an output WAV.

```bat
python cosyvoice_cli.py --text "Hello world!" --reference "C:\\path\\to\\reference.wav"
```

## What the CLI does

- Converts the reference audio to **16 kHz mono** (CosyVoice token extraction expects 16k).
- Trims reference audio to **<= 30s** (CosyVoice zero‑shot requirement).
- Auto‑transcribes the processed reference audio into `reference_text.txt` using **faster‑whisper** (default).
- Uses (a short prefix of) that transcript as the **prompt text** so CosyVoice can preserve speaking style.
- Runs CosyVoice **zero‑shot inference** and writes a timestamped output WAV.

Outputs live in a folder next to your reference:

```
<reference_dir>\<reference_stem>\
  reference_text.txt
  reference_text.meta.json
  <reference_stem>__ref_16k_mono.wav
  <reference_stem>__trimmed_29.5s.wav   (only if the ref is longer than 29.5s)
  output_YYMMDD-HHMMSS.wav
```

## Quickstart (Windows, CPU)

1) Clone the repo and open an **Anaconda Prompt** in the repo root.

2) Create the conda environment:

```bat
conda env create -f env\conda-windows-cpu.yml
conda activate cosyvoice-cli
```

3) Run the CLI:

```bat
python cosyvoice_cli.py --text "Hello world!" --reference "C:\\path\\to\\reference.wav"
```

### Notes

- The first run will download models (CosyVoice model weights, Whisper ASR model weights). Later runs are much faster.
- If you want **no VLC auto-play**, add `--no_play`.

## CLI options

Show help:

```bat
python cosyvoice_cli.py -h
```

Commonly useful flags:

- `--model_dir` : path to your CosyVoice model (default `pretrained_models/Fun-CosyVoice3-0.5B`)
- `--max_ref_seconds` : reference trim limit (default `29.5`)
- `--max_transcript_words` / `--prompt_ref_words` : how many transcript words to include in the prompt (default `60`)
- `--force_retranscribe` : ignore cached transcript and re-run ASR
- `--asr_engine faster-whisper|whisper` : ASR backend
- `--asr_model tiny|base|small|medium|large-v3` : Whisper model size (default `small`)
- `--asr_device cpu|cuda` : ASR device preference (default `cpu` for stability)
- `--no_play` : do not launch VLC
- `--vlc "C:\\Program Files\\VideoLAN\\VLC\\vlc.exe"` : explicit VLC path

## Troubleshooting

### “Reference WAV is supposed to match reference text”

Yes — for zero‑shot cloning, the `reference_text` is intended to be the transcript of the **reference audio**. This CLI generates it automatically from the processed (16k/mono/trimmed) reference, then caches it in the output folder.

If you swap to a different reference WAV but keep the same output folder, cached transcripts can get stale. Use:

```bat
python cosyvoice_cli.py ... --force_retranscribe
```

### “Specified provider 'CUDAExecutionProvider' is not in available provider names”

That warning comes from **onnxruntime**. It means your current install is CPU-only (which is totally fine).

If you want GPU on Windows, you typically need an `onnxruntime-gpu` build that matches your CUDA toolkit, plus compatible drivers. This project defaults to CPU ASR (`--asr_device cpu`) specifically to avoid CUDA DLL/version headaches.

### faster-whisper needs ffmpeg

`faster-whisper` uses ffmpeg for decoding many audio formats. The included conda environment installs ffmpeg via conda-forge.

If you install dependencies manually, make sure `ffmpeg` is on PATH.

### Collapse / gibberish / very short output

If CosyVoice collapses to very short audio, the usual culprits are:

- The prompt transcript is too long relative to the target text (prompt dominates).
- The transcript doesn’t match the audio (bad ASR / wrong cache).
- Very short target text (a couple of words) can be unstable.

Fixes:

- Reduce `--max_transcript_words` (try 15–30).
- Re-transcribe with `--force_retranscribe`.
- Try a slightly longer target text to “get it started”, then generate longer content.

## Repo layout additions

This CLI adds:

- `cosyvoice_cli.py` — the runnable script
- `env/conda-windows-cpu.yml` — a reproducible conda environment for Windows CPU
- `requirements-cli.txt` — the pip requirements used by that conda env

