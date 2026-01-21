# CosyVoice-CLI

This repository is the upstream **CosyVoice** codebase plus a small, reproducible **command‑line interface** for **CosyVoice3 zero‑shot voice cloning**.

The goal: from an **Anaconda Prompt on Windows**, run a single command with **(text, reference WAV)** and get an output WAV.

```bat
python cosyvoice_cli.py --text "Hello world!" --reference "C:\\path\\to\\reference.wav"
```

## Two-voice CLI (scripted dialog)

Use `cosyvoice-cli-twovoice.py` to generate a conversation with up to two voices. Provide the first reference via `--reference` and the second via `--reference2`.

If you **omit `--text`**, the script will load `voice_script.txt` from the repo root. The script supports optional prefixes:

- `V1:` lines are spoken by voice 1 (the `--reference` speaker).
- `V2:` lines are spoken by voice 2 (the `--reference2` speaker).
- If no `V1:` marker exists in the script, the entire text is spoken as voice 1.

Example script (`voice_script.txt`):

```
V1: Hey there! It’s good to hear from you.
V2: Hi! I’ve been looking forward to catching up.
V1: Same here. Want to grab coffee later?
V2: Absolutely. Let’s do it.
```

Run with two voices and the default script file:

```bat
python cosyvoice-cli-twovoice.py --reference "C:\\path\\to\\v1.wav" --reference2 "C:\\path\\to\\v2.wav"
```

Run with inline text (no `voice_script.txt` needed):

```bat
python cosyvoice-cli-twovoice.py --text "V1: Hello! V2: Hi there!" --reference "C:\\path\\to\\v1.wav" --reference2 "C:\\path\\to\\v2.wav"
```

Optional: add silence between turns (default 350ms):

```bat
python cosyvoice-cli-twovoice.py --reference "C:\\path\\to\\v1.wav" --reference2 "C:\\path\\to\\v2.wav" --silence_ms 500
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

@@ -90,28 +127,28 @@ If you want GPU on Windows, you typically need an `onnxruntime-gpu` build that m
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
- `cosyvoice-cli-twovoice.py` — two-voice script runner (optional V1/V2)
- `env/conda-windows-cpu.yml` — a reproducible conda environment for Windows CPU
- `requirements-cli.txt` — the pip requirements used by that conda env