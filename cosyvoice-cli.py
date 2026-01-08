"""CosyVoice-CLI

A small, reproducible command-line wrapper around CosyVoice3 zero-shot voice cloning.

Goals
- User provides: (text, reference wav)
- Script ensures the reference is 16 kHz mono and <= 30s
- Script auto-transcribes the reference (faster-whisper by default) into reference_text.txt
- Script runs CosyVoice zero-shot inference and writes an output WAV next to the reference

Windows / Anaconda Prompt example
  python cosyvoice_cli.py --text "Hello world" --reference "C:\\path\\to\\ref.wav"
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
        description="CosyVoice3 CLI wrapper: synthesize speech from text + reference wav.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--text", required=True, help='Text to speak, e.g. "Hello world!"')
    parser.add_argument("--reference", required=True, help="Path to reference speaker WAV file.")

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

    return parser


def main() -> None:
    args = build_parser().parse_args()

    reference_wav = Path(args.reference).expanduser().resolve()
    if not reference_wav.exists():
        raise FileNotFoundError(f"Reference WAV not found: {reference_wav}")

    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model dir not found: {model_dir}\n"
            "Did you download / place the CosyVoice model into pretrained_models/?"
        )

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else output_dir_for_reference(reference_wav)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_wav = out_dir / f"output_{timestamp_yyMMdd_hhmmss()}.wav"

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

    # Keep prompt short unless we have a (short) transcript.
    if reference_text:
        prompt_text = f"{args.prompt_prefix}<|endofprompt|>{reference_text}"
        print(f"[INFO] Using reference_text (truncated): {reference_text[:120]}{'...' if len(reference_text) > 120 else ''}")
        print(f"[INFO] Prompt ref words used: {args.max_transcript_words}")
    else:
        prompt_text = "<|endofprompt|>"
        print(
            "[WARN] reference_text unavailable. Cloning can work, but speaker similarity may be worse.\n"
            "       (Auto transcription failed or was disabled.)"
        )

    print(f"[INFO] Reference WAV:  {reference_wav}")
    print(f"[INFO] Using ref WAV:  {ref_for_model}")
    print(f"[INFO] Model dir:      {model_dir}")
    print(f"[INFO] Output path:    {out_wav}")

    cosyvoice = AutoModel(model_dir=str(model_dir))

    chunks: list[torch.Tensor] = []
    for out in cosyvoice.inference_zero_shot(
        args.text,
        prompt_text,
        str(ref_for_model),
        stream=False,
    ):
        speech = out.get("tts_speech")
        if speech is not None:
            chunks.append(speech)

    audio = concat_audio_chunks(chunks)
    torchaudio.save(str(out_wav), audio, cosyvoice.sample_rate)

    if not out_wav.exists():
        raise RuntimeError(f"Output file was not created: {out_wav}")

    print(f"Created: {out_wav}")

    if not args.no_play:
        play_in_vlc(out_wav, vlc_path=args.vlc)


if __name__ == "__main__":
    main()
