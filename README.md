# CosyVoice-CLI

This repository is the upstream **CosyVoice** codebase plus a small, reproducible **command-line interface** for **CosyVoice3 zero-shot voice cloning**.

The goal: from an **Anaconda Prompt on Windows**, run a single command with **(text, reference WAV)** or **(target audio, reference WAV)** and get an output file.

```bat
python cosyvoice-cli.py --text "Hello world!" --reference "C:\\path\\to\\reference.wav"
```

## Two-voice CLI (scripted dialog)

Use `cosyvoice-cli.py` to generate a conversation with up to two voices. Provide the first reference via `--reference` and the second via `--reference2`.

If you **omit `--text`**, the script will load `voice_script.txt` from the repo root. The script supports optional prefixes:

- `V1:` lines are spoken by voice 1 (the `--reference` speaker).
- `V2:` lines are spoken by voice 2 (the `--reference2` speaker).
- If no `V1:` marker exists in the script, the entire text is spoken as voice 1.

Example script (`voice_script.txt`):

```
V1: Hey there! It's good to hear from you.
V2: Hi! I've been looking forward to catching up.
V1: Same here. Want to grab coffee later?
V2: Absolutely. Let's do it.
```

Run with two voices and the default script file:

```bat
python cosyvoice-cli.py --reference "C:\\path\\to\\v1.wav" --reference2 "C:\\path\\to\\v2.wav"
```

Run with inline text (no `voice_script.txt` needed):

```bat
python cosyvoice-cli.py --text "V1: Hello! V2: Hi there!" --reference "C:\\path\\to\\v1.wav" --reference2 "C:\\path\\to\\v2.wav"
```

Optional: add silence between turns (default 350ms):

```bat
python cosyvoice-cli.py --reference "C:\\path\\to\\v1.wav" --reference2 "C:\\path\\to\\v2.wav" --silence_ms 500
```

## What the CLI does

- Converts the reference audio to **16 kHz mono** (CosyVoice token extraction expects 16k).
- Trims reference audio to **<= 30s** (CosyVoice zero-shot requirement).
- Auto-transcribes the processed reference audio into `reference_text.txt` using **faster-whisper** (default).
- Uses (a short prefix of) that transcript as the **prompt text** so CosyVoice can preserve speaking style.
- Runs CosyVoice **zero-shot inference** for text mode, or **voice conversion** for target mode.

Outputs (text mode) live in a folder next to your reference:

```
<reference_dir>\<reference_stem>\
  reference_text.txt
  reference_text.meta.json
  <reference_stem>__ref_16k_mono.wav
  <reference_stem>__trimmed_29.5s.wav   (only if the ref is longer than 29.5s)
  output_YYMMDD-HHMMSS.wav
```

## Voice conversion (--target)

Use `--target` to convert an existing audio file into the voice defined by `--reference`.

Rules:
- `--target` requires `--reference`.
- `--target` cannot be used with `--text` or `--reference2`.

Audio-only target example:

```bat
python cosyvoice-cli.py --reference "C:\\path\\to\\ref.wav" --target "C:\\path\\to\\target.mp3"
```

MP4 target example (video preserved, audio swapped):

```bat
python cosyvoice-cli.py --reference "C:\\path\\to\\ref.wav" --target "C:\\path\\to\\clip.mp4"
```

Target mode details:
- Target audio is decoded (if needed), converted to **16 kHz mono**, and trimmed to **<= 30s**.
- For audio-only targets, output is a WAV in the reference output folder.
- For MP4 targets, the tool extracts audio, runs voice conversion, then writes a new MP4
  named `clip_YYMMDD-HHMMSS.mp4` next to the original, keeping the video unchanged.

## Quickstart (Windows, CPU)

1) Clone the repo and open an **Anaconda Prompt** in the repo root.

2) Create the conda environment:

### faster-whisper needs ffmpeg

`faster-whisper` uses ffmpeg for decoding many audio formats. The included conda environment installs ffmpeg via conda-forge.

If you install dependencies manually, make sure `ffmpeg` is on PATH.

### Collapse / gibberish / very short output

If CosyVoice collapses to very short audio, the usual culprits are:

- The prompt transcript is too long relative to the target text (prompt dominates).
- The transcript doesn't match the audio (bad ASR / wrong cache).
- Very short target text (a couple of words) can be unstable.

Fixes:

- Reduce `--max_transcript_words` (try 15-30).
- Re-transcribe with `--force_retranscribe`.
- Try a slightly longer target text to "get it started", then generate longer content.

## Repo layout additions

This CLI adds:

- `cosyvoice-cli.py` - the runnable script, including two-voice script runner (optional V1/V2)
- `env/conda-windows-cpu.yml` - a reproducible conda environment for Windows CPU
- `requirements-cli.txt` - the pip requirements used by that conda env
