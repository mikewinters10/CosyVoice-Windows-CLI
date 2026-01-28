"""CosyVoice-CLI

A small, reproducible command-line wrapper around CosyVoice3 zero-shot voice cloning.

Goals
- User provides: (text, reference wav) or (target audio, reference wav) for voice conversion
- Script ensures the reference is 16 kHz mono and <= 30s
- Script auto-transcribes the reference (faster-whisper by default) into reference_text.txt
- Script runs CosyVoice zero-shot inference (text mode) or voice conversion (target mode)

Windows / Anaconda Prompt example
  python cosyvoice_cli.py --text "Hello world" --reference "C:\\path\\to\\ref.wav"

Two-voice example
  python cosyvoice-cli-twovoice.py --reference "C:\\path\\to\\v1.wav" --reference2 "C:\\path\\to\\v2.wav"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torchaudio


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


# CosyVoice repo expects this path tweak (per their examples)
_matcha = _repo_root() / "third_party" / "Matcha-TTS"
if _matcha.exists():
    sys.path.append(str(_matcha))

from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402


def timestamp_yyMMdd_hhmmss() -> str:
    return datetime.now().strftime("%y%m%d-%H%M%S")


def output_dir_for_reference(reference_wav: Path) -> Path:
    out_dir = reference_wav.parent / reference_wav.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _ffmpeg_path() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    candidates = [
        r"C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe",
        r"C:\\ffmpeg\\bin\\ffmpeg.exe",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError(
        "ffmpeg not found. Install ffmpeg and make sure ffmpeg.exe is on PATH.\n"
        "Try: winget install Gyan.FFmpeg  (then reopen your terminal)"
    )


def decode_to_wav_with_ffmpeg(src: Path, out_wav: Path) -> Path:
    """Decode arbitrary audio to a WAV file via ffmpeg."""
    ffmpeg = _ffmpeg_path()
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vn",
        "-acodec",
        "pcm_s16le",
        str(out_wav),
    ]
    subprocess.run(cmd, check=True)
    return out_wav


def replace_audio_in_mp4(src_mp4: Path, audio_wav: Path, out_mp4: Path) -> None:
    """Replace audio track in an MP4, preserving video, and write to out_mp4."""
    ffmpeg = _ffmpeg_path()
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src_mp4),
        "-i",
        str(audio_wav),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-map_metadata",
        "0",
        "-map_chapters",
        "0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(out_mp4),
    ]
    subprocess.run(cmd, check=True)


def ensure_wav_16k_mono(wav_path: Path) -> tuple[torch.Tensor, int]:
    """Load audio and return mono 16kHz tensor shaped [1, T]."""
    audio, sr = torchaudio.load(str(wav_path))  # [C, T]

    # mono
    if audio.dim() == 2 and audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # 16 kHz (CosyVoice asserts using 16000 in token extractor)
    target_sr = 16000
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
        sr = target_sr

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    return audio, sr


def ensure_16k_mono_on_disk(src: Path, out: Path, max_seconds: float) -> Path:
    """Normalize audio to 16 kHz mono WAV on disk, trimming to max_seconds."""
    src = src.expanduser().resolve()
    out = out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    tmp_wav = out.with_suffix(".tmp_decode.wav")

    try:
        if src.suffix.lower() == ".wav":
            wav_path = src
        else:
            wav_path = decode_to_wav_with_ffmpeg(src, tmp_wav)

        wav, sr = torchaudio.load(str(wav_path))
        wav = wav.to(torch.float32)

        if wav.dim() == 2 and wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
            sr = 16000

        if max_seconds is not None and max_seconds > 0:
            max_samples = int(sr * float(max_seconds))
            if wav.shape[1] > max_samples:
                wav = wav[:, :max_samples]
                print(f"[INFO] Cropped {src.name} to first {max_seconds:.1f}s for token extraction.")

        torchaudio.save(str(out), wav, sr)
        return out
    finally:
        if tmp_wav.exists():
            try:
                tmp_wav.unlink()
            except OSError:
                pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_reference_audio(
    reference_wav: Path,
    out_dir: Path,
    max_seconds: float = 29.5,
) -> Path:
    """Create a processed reference WAV under out_dir that is 16k mono and <= max_seconds.

    Always returns a path inside out_dir so that:
    - ASR transcript corresponds to the exact audio fed to CosyVoice
    - The original reference file is never modified
    """
    audio, sr = ensure_wav_16k_mono(reference_wav)
    dur_s = audio.shape[1] / sr

    proc_path = out_dir / f"{reference_wav.stem}__ref_16k_mono.wav"
    torchaudio.save(str(proc_path), audio, sr)

    if dur_s <= max_seconds:
        return proc_path

    max_samples = int(max_seconds * sr)
    trimmed = audio[:, :max_samples]
    trimmed_path = out_dir / f"{reference_wav.stem}__trimmed_{max_seconds:.1f}s.wav"
    torchaudio.save(str(trimmed_path), trimmed, sr)
    print(f"[INFO] Reference was {dur_s:.1f}s; trimmed to {max_seconds:.1f}s: {trimmed_path}")
    return trimmed_path


def concat_audio_chunks(chunks: list[torch.Tensor]) -> torch.Tensor:
    if not chunks:
        raise RuntimeError("No audio was generated (empty output stream).")

    fixed: list[torch.Tensor] = []
    for t in chunks:
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        t = t.detach().cpu()
        if t.dim() == 1:
            t = t.unsqueeze(0)  # [T] -> [1, T]
        fixed.append(t)

    return torch.cat(fixed, dim=-1)


def _clean_transcript(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _truncate_words(text: str, max_words: int) -> str:
    words = (text or "").split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).strip()


def transcribe_with_faster_whisper(
    wav_path: Path,
    model_size: str,
    device: str,
    compute_type: str,
) -> str:
    from faster_whisper import WhisperModel

    # If user asked for cuda but it's unavailable, fall back cleanly.
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        compute_type = "int8"

    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, _info = model.transcribe(
        str(wav_path),
        language="en",
        vad_filter=True,
        beam_size=5,
    )
    return _clean_transcript(" ".join(seg.text.strip() for seg in segments))


def transcribe_with_openai_whisper(
    wav_path: Path,
    model_size: str,
    device: str,
) -> str:
    import whisper

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = whisper.load_model(model_size, device=device)
    result = model.transcribe(str(wav_path), language="en")
    return _clean_transcript(result.get("text") or "")


def load_or_create_reference_text(
    out_dir: Path,
    ref_for_model: Path,
    transcript_filename: str,
    auto_transcribe: bool,
    asr_engine: str,
    asr_model: str,
    asr_device: str,
    asr_compute_type: str,
    max_transcript_words: int,
    force_retranscribe: bool,
) -> Optional[str]:
    """Load cached transcript if it matches the current processed reference.

    Writes:
      - reference_text.txt (cleaned transcript)
      - reference_text.meta.json (hash + settings) to avoid stale cache
    """
    transcript_path = out_dir / transcript_filename
    meta_path = transcript_path.with_suffix(transcript_path.suffix + ".meta.json")

    current_hash = sha256_file(ref_for_model)

    if not force_retranscribe and transcript_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("ref_sha256") == current_hash:
                txt = _clean_transcript(transcript_path.read_text(encoding="utf-8"))
                txt = _truncate_words(txt, max_transcript_words)
                if txt:
                    print(f"[INFO] Loaded cached reference_text: {transcript_path}")
                    return txt
        except Exception:
            # If cache is corrupted, fall through to re-transcribe.
            pass

    if transcript_path.exists() and not meta_path.exists() and not force_retranscribe:
        # Back-compat: if user has only reference_text.txt from older runs.
        txt = _clean_transcript(transcript_path.read_text(encoding="utf-8"))
        txt = _truncate_words(txt, max_transcript_words)
        if txt:
            print(f"[INFO] Loaded cached reference_text (no meta): {transcript_path}")
            # Write meta so the cache becomes safe going forward.
            meta_path.write_text(
                json.dumps(
                    {
                        "ref_sha256": current_hash,
                        "asr_engine": asr_engine,
                        "asr_model": asr_model,
                        "asr_device": asr_device,
                        "asr_compute_type": asr_compute_type,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            return txt

    if not auto_transcribe:
        return None

    print(f"[INFO] Transcribing reference for reference_text: {ref_for_model.name}")

    txt = ""
    if asr_engine == "faster-whisper":
        try:
            txt = transcribe_with_faster_whisper(
                ref_for_model,
                model_size=asr_model,
                device=asr_device,
                compute_type=asr_compute_type,
            )
        except ModuleNotFoundError:
            print("[WARN] faster-whisper not installed. Falling back to openai-whisper if available.")
            asr_engine = "whisper"

    if asr_engine == "whisper":
        try:
            txt = transcribe_with_openai_whisper(ref_for_model, model_size=asr_model, device=asr_device)
        except ModuleNotFoundError:
            print("[WARN] openai-whisper not installed. Cannot auto-transcribe.")

    txt = _clean_transcript(txt)

    if not txt:
        print("[WARN] ASR produced empty transcript. Proceeding without reference_text.")
        return None

    # Save the cleaned transcript, then use only the first N words for the prompt.
    transcript_path.write_text(txt + "\n", encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "ref_sha256": current_hash,
                "asr_engine": asr_engine,
                "asr_model": asr_model,
                "asr_device": asr_device,
                "asr_compute_type": asr_compute_type,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[INFO] Saved reference_text to: {transcript_path}")

    return _truncate_words(txt, max_transcript_words)


def play_in_vlc(wav_path: Path, vlc_path: Optional[str] = None) -> None:
    candidates = [
        vlc_path,
        shutil.which("vlc"),
        r"C:\\Program Files\\VideoLAN\\VLC\\vlc.exe",
        r"C:\\Program Files (x86)\\VideoLAN\\VLC\\vlc.exe",
    ]
    vlc_exe = next((c for c in candidates if c and Path(c).exists()), None)
    if not vlc_exe:
        print("VLC not found (vlc.exe not on PATH and not in Program Files). Skipping playback.")
        return

    subprocess.Popen([vlc_exe, str(wav_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CosyVoice3 CLI wrapper: synthesize speech from text + reference wav, or voice conversion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--text", help='Text to speak, e.g. "Hello world!"')
    parser.add_argument("--reference", required=True, help="Path to reference speaker WAV file.")
    parser.add_argument(
        "--reference2",
        help="Optional path to a second reference speaker WAV file for two-voice scripts.",
    )
    parser.add_argument(
        "--target",
        help="Path to target audio for voice conversion (requires --reference, incompatible with --reference2).",
    )

    parser.add_argument(
        "--model_dir",
        default=str(_repo_root() / "pretrained_models" / "Fun-CosyVoice3-0.5B"),
        help="Path to downloaded CosyVoice model directory.",
    )
    parser.add_argument(
        "--max_ref_seconds",
        type=float,
        default=29.5,
        help="CosyVoice zero-shot requires reference audio <= 30s; we auto-trim to this duration.",
    )

    # Auto transcript caching (ON by default)
    parser.add_argument(
        "--no_auto_transcribe",
        action="store_true",
        help="Disable auto speech-to-text transcript generation/caching.",
    )
    parser.add_argument(
        "--force_retranscribe",
        action="store_true",
        help="Ignore cached reference_text and transcribe again.",
    )
    parser.add_argument(
        "--transcript_file",
        default="reference_text.txt",
        help="Cached transcript filename inside the output folder.",
    )
    parser.add_argument(
        "--transcript_file2",
        default="reference2_text.txt",
        help="Cached transcript filename for the second voice inside the output folder.",
    )

    # How much of the transcript to include in the *prompt*.
    parser.add_argument(
        "--max_transcript_words",
        "--prompt_ref_words",
        dest="max_transcript_words",
        type=int,
        default=60,
        help="Use only the first N words of the reference transcript in the prompt.",
    )

    parser.add_argument(
        "--prompt_prefix",
        default="Speak in English (United States).",
        help="Text prepended before <|endofprompt|> in the CosyVoice prompt.",
    )

    # ASR options
    parser.add_argument(
        "--asr_engine",
        choices=["faster-whisper", "whisper"],
        default="faster-whisper",
        help="ASR backend used to create reference_text if missing.",
    )
    parser.add_argument(
        "--asr_model",
        default="small",
        help="Whisper model size (tiny/base/small/medium/large-v3).",
    )
    parser.add_argument(
        "--asr_device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="ASR device preference (default: cpu for stability).",
    )
    parser.add_argument(
        "--asr_compute_type",
        default="int8",
        help="faster-whisper compute type (default: int8 for fast/stable CPU ASR). Ignored by openai-whisper.",
    )

    # Output
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Optional output directory. Default: a folder named after the reference WAV stem, next to the reference.",
    )

    # Playback
    parser.add_argument("--no_play", action="store_true", help="Disable auto-play in VLC.")
    parser.add_argument("--vlc", default=None, help=r"Optional full path to vlc.exe")
    parser.add_argument(
        "--silence_ms",
        type=int,
        default=350,
        help="Silence padding (ms) inserted between utterances when using scripts.",
    )

    return parser


def _load_script_text(text_arg: Optional[str]) -> str:
    if text_arg and text_arg.strip():
        return text_arg.strip()

    script_path = _repo_root() / "voice_script.txt"
    if not script_path.exists():
        raise FileNotFoundError(
            f"No --text provided and default script not found: {script_path}.\n"
            "Create voice_script.txt or pass --text."
        )
    return script_path.read_text(encoding="utf-8").strip()


def _parse_script(text: str, has_voice2: bool) -> list[tuple[int, str]]:
    """Return a list of (voice_index, utterance) tuples."""
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Script text is empty.")

    if not has_voice2:
        return [(1, cleaned)]

    lines = cleaned.splitlines()
    segments: list[tuple[int, str]] = []
    current_voice: Optional[int] = None
    buffer: list[str] = []

    def flush() -> None:
        nonlocal buffer, current_voice
        if current_voice is None:
            return
        merged = " ".join(" ".join(buffer).split()).strip()
        if merged:
            segments.append((current_voice, merged))
        buffer = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("V1:"):
            flush()
            current_voice = 1
            buffer.append(stripped[3:].strip())
            continue
        if stripped.startswith("V2:"):
            flush()
            current_voice = 2
            buffer.append(stripped[3:].strip())
            continue
        buffer.append(stripped)

    flush()

    if not any(seg[0] == 1 for seg in segments):
        return [(1, cleaned)]

    return segments


def _build_prompt(reference_text: Optional[str], prompt_prefix: str, max_words: int) -> str:
    if reference_text:
        print(
            f"[INFO] Using reference_text (truncated): {reference_text[:120]}{'...' if len(reference_text) > 120 else ''}"
        )
        print(f"[INFO] Prompt ref words used: {max_words}")
        return f"{prompt_prefix}<|endofprompt|>{reference_text}"

    print(
        "[WARN] reference_text unavailable. Cloning can work, but speaker similarity may be worse.\n"
        "       (Auto transcription failed or was disabled.)"
    )
    return "<|endofprompt|>"


def _generate_audio(
    cosyvoice: AutoModel,
    text: str,
    prompt_text: str,
    ref_for_model: Path,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for out in cosyvoice.inference_zero_shot(
        text,
        prompt_text,
        str(ref_for_model),
        stream=False,
    ):
        speech = out.get("tts_speech")
        if speech is not None:
            chunks.append(speech)
    return concat_audio_chunks(chunks)


def _silence(sample_rate: int, silence_ms: int) -> torch.Tensor:
    silence_samples = int(sample_rate * max(silence_ms, 0) / 1000.0)
    return torch.zeros(1, silence_samples)


def main() -> None:
    args = build_parser().parse_args()

    reference_wav = Path(args.reference).expanduser().resolve()
    if not reference_wav.exists():
        raise FileNotFoundError(f"Reference WAV not found: {reference_wav}")

    if args.target and args.text:
        raise ValueError("--target cannot be used with --text (voice conversion mode).")

    if args.target and args.reference2:
        raise ValueError("--target cannot be used with --reference2.")

    reference_wav2 = None
    if args.reference2:
        reference_wav2 = Path(args.reference2).expanduser().resolve()
        if not reference_wav2.exists():
            raise FileNotFoundError(f"Reference WAV 2 not found: {reference_wav2}")

    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model dir not found: {model_dir}\n"
            "Did you download / place the CosyVoice model into pretrained_models/?"
        )

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else output_dir_for_reference(reference_wav)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_stamp = timestamp_yyMMdd_hhmmss()
    out_wav = out_dir / f"output_{run_stamp}.wav"
    final_output_path: Optional[Path] = None

    if args.target:
        target_audio = Path(args.target).expanduser().resolve()
        if not target_audio.exists():
            raise FileNotFoundError(f"Target audio not found: {target_audio}")
        target_is_mp4 = target_audio.suffix.lower() == ".mp4"

        ref_for_model = ensure_16k_mono_on_disk(
            reference_wav,
            out_dir / f"{reference_wav.stem}__ref_16k_mono.wav",
            max_seconds=float(args.max_ref_seconds),
        )
        target_for_model = ensure_16k_mono_on_disk(
            target_audio,
            out_dir / f"{target_audio.stem}__target_16k_mono.wav",
            max_seconds=float(args.max_ref_seconds),
        )

        print("[INFO] Voice conversion mode.")
        print(f"[INFO] Reference WAV: {reference_wav}")
        print(f"[INFO] Using ref WAV: {ref_for_model}")
        print(f"[INFO] Target audio:  {target_audio}")
        print(f"[INFO] Using target: {target_for_model}")
        print(f"[INFO] Model dir:    {model_dir}")
        if target_is_mp4:
            final_output_path = target_audio.parent / f"{target_audio.stem}_{run_stamp}.mp4"
        else:
            final_output_path = out_wav
        print(f"[INFO] Output path:  {final_output_path}")

        cosyvoice = AutoModel(model_dir=str(model_dir))
        vc_chunks: list[torch.Tensor] = []
        for out in cosyvoice.inference_vc(str(target_for_model), str(ref_for_model), stream=False):
            speech = out.get("tts_speech")
            if speech is not None:
                vc_chunks.append(speech)

        audio = concat_audio_chunks(vc_chunks)
        if target_is_mp4:
            tmp_processed_wav = out_dir / f"{target_audio.stem}__processed_{run_stamp}.wav"
            torchaudio.save(str(tmp_processed_wav), audio, cosyvoice.sample_rate)
            try:
                replace_audio_in_mp4(target_audio, tmp_processed_wav, final_output_path)
            finally:
                if tmp_processed_wav.exists():
                    try:
                        tmp_processed_wav.unlink()
                    except OSError:
                        pass
        else:
            torchaudio.save(str(final_output_path), audio, cosyvoice.sample_rate)
    else:
        script_text = _load_script_text(args.text)
        script_segments = _parse_script(script_text, has_voice2=reference_wav2 is not None)

        # CosyVoice hard-requires <= 30s for zero-shot token extraction.
        # We also always convert to 16kHz mono.
        ref_for_model = prepare_reference_audio(reference_wav, out_dir, max_seconds=float(args.max_ref_seconds))

        reference_text = load_or_create_reference_text(
            out_dir=out_dir,
            ref_for_model=ref_for_model,
            transcript_filename=args.transcript_file,
            auto_transcribe=(not args.no_auto_transcribe),
            asr_engine=args.asr_engine,
            asr_model=args.asr_model,
            asr_device=args.asr_device,
            asr_compute_type=args.asr_compute_type,
            max_transcript_words=int(args.max_transcript_words),
            force_retranscribe=bool(args.force_retranscribe),
        )

        prompt_text = _build_prompt(reference_text, args.prompt_prefix, args.max_transcript_words)

        ref_for_model2 = None
        prompt_text2 = None
        if reference_wav2:
            ref_for_model2 = prepare_reference_audio(reference_wav2, out_dir, max_seconds=float(args.max_ref_seconds))
            reference_text2 = load_or_create_reference_text(
                out_dir=out_dir,
                ref_for_model=ref_for_model2,
                transcript_filename=args.transcript_file2,
                auto_transcribe=(not args.no_auto_transcribe),
                asr_engine=args.asr_engine,
                asr_model=args.asr_model,
                asr_device=args.asr_device,
                asr_compute_type=args.asr_compute_type,
                max_transcript_words=int(args.max_transcript_words),
                force_retranscribe=bool(args.force_retranscribe),
            )
            prompt_text2 = _build_prompt(reference_text2, args.prompt_prefix, args.max_transcript_words)

        print(f"[INFO] Reference WAV:  {reference_wav}")
        print(f"[INFO] Using ref WAV:  {ref_for_model}")
        if reference_wav2:
            print(f"[INFO] Reference2 WAV: {reference_wav2}")
            print(f"[INFO] Using ref2 WAV: {ref_for_model2}")
        print(f"[INFO] Model dir:      {model_dir}")
        print(f"[INFO] Output path:    {out_wav}")

        cosyvoice = AutoModel(model_dir=str(model_dir))

        silence_chunk = _silence(cosyvoice.sample_rate, args.silence_ms)
        output_chunks: list[torch.Tensor] = []

        for idx, (voice_idx, utterance) in enumerate(script_segments, start=1):
            if voice_idx == 2 and not (ref_for_model2 and prompt_text2):
                raise RuntimeError("Script references V2, but no --reference2 was provided.")

            active_ref = ref_for_model if voice_idx == 1 else ref_for_model2
            active_prompt = prompt_text if voice_idx == 1 else prompt_text2

            print(f"[INFO] Synthesizing turn {idx} (V{voice_idx}): {utterance[:60]}")
            output_chunks.append(_generate_audio(cosyvoice, utterance, active_prompt, active_ref))
            if idx < len(script_segments) and silence_chunk.numel() > 0:
                output_chunks.append(silence_chunk)

        audio = concat_audio_chunks(output_chunks)
        torchaudio.save(str(out_wav), audio, cosyvoice.sample_rate)
        final_output_path = out_wav

    if not final_output_path or not final_output_path.exists():
        raise RuntimeError(f"Output file was not created: {final_output_path}")

    print(f"Created: {final_output_path}")

    if not args.no_play:
        play_in_vlc(final_output_path, vlc_path=args.vlc)


if __name__ == "__main__":
    main()
