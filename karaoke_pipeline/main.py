"""
Karaoke Pipeline - Gender Voice Remover
========================================
Pipeline:
  1. Separate song into stems with Demucs (htdemucs_6s).
  2. Classify vocal segments as male/female with SpeechBrain.
  3. Silence segments matching selected gender using pydub.
  4. Mix cleaned vocals back with remaining stems.
  5. Write output_gender_removed.wav.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Remove male or female voice from a song using Demucs + SpeechBrain."
    )
    p.add_argument("--input", required=True, help="Path to input MP3 or WAV file.")
    p.add_argument("--output", default=None, help="Path for the output WAV file.")
    p.add_argument("--remove", default="male", choices=["male", "female"],
                   help="Gender to remove: male or female (default: male).")
    p.add_argument("--segment", type=float, default=3.0,
                   help="Segment length in seconds for gender classification (default: 3).")
    p.add_argument("--threshold", type=float, default=0.60,
                   help="Confidence threshold to tag a segment (default: 0.60).")
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU inference even when a GPU is available.")
    return p.parse_args()


def select_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        device = torch.device("cpu")
        print("[device] Forced CPU mode.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[device] GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[device] No GPU found - running on CPU (this may take a few minutes).")
    return device


def separate_stems(input_path: Path, out_dir: Path, device: torch.device) -> dict:
    print("\n[demucs] Running htdemucs_6s stem separation...")
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "demucs",
        "--name", "htdemucs_6s",
        "--out", str(out_dir),
        "--device", str(device),
        str(input_path),
    ]
    print(f"[demucs] Separating stems for: {input_path.name}")
    subprocess.run(cmd, check=True)

    track_dir = out_dir / "htdemucs_6s" / input_path.stem
    if not track_dir.exists():
        raise RuntimeError(f"Demucs output directory not found: {track_dir}.")

    stem_paths = {}
    for stem_file in sorted(track_dir.glob("*.wav")):
        stem_paths[stem_file.stem] = stem_file
        print(f"[demucs]   found {stem_file.name}")

    if "vocals" not in stem_paths:
        raise RuntimeError("Demucs did not produce a 'vocals' stem.")

    return stem_paths


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[error] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    suffix = input_path.suffix.lower()
    if suffix not in {".mp3", ".wav"}:
        print(f"[error] Unsupported format '{suffix}'. Use MP3 or WAV.", file=sys.stderr)
        sys.exit(1)

    if suffix == ".mp3" and shutil.which("ffmpeg") is None:
        print("[error] ffmpeg not found. Install with: winget install Gyan.FFmpeg", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"output_no_{args.remove}.wav"

    device = select_device(args.cpu)

    print(f"\n[main] Input  : {input_path}")
    print(f"[main] Remove : {args.remove} voices")
    print(f"[main] Output : {output_path}\n")

    from gender_filter import classify_vocal_segments, suppress_segments
    from mixer import mix_stems

    with tempfile.TemporaryDirectory(prefix="karaoke_") as tmp:
        tmp_dir = Path(tmp)

        print("[main] Step 1: Separating stems...")
        try:
            stem_paths = separate_stems(input_path, tmp_dir / "stems", device)
        except Exception as exc:
            print(f"[error] Stem separation failed: {exc}", file=sys.stderr)
            sys.exit(1)

        vocals_path = stem_paths["vocals"]

        print(f"\n[main] Step 2: Classifying gender segments (removing '{args.remove}')...")
        try:
            segments = classify_vocal_segments(
                vocals_path,
                segment_sec=args.segment,
                threshold=args.threshold,
                device=device,
                remove=args.remove,
            )
        except Exception as exc:
            print(f"[error] Gender classification failed: {exc}", file=sys.stderr)
            sys.exit(1)

        print("\n[main] Step 3: Suppressing matched segments...")
        clean_vocals = tmp_dir / "vocals_clean.wav"
        try:
            suppress_segments(vocals_path, segments, clean_vocals)
        except Exception as exc:
            print(f"[error] Suppression failed: {exc}", file=sys.stderr)
            sys.exit(1)

        print("\n[main] Step 4: Mixing stems...")
        try:
            mix_stems(clean_vocals, stem_paths, output_path)
        except Exception as exc:
            print(f"[error] Mixing failed: {exc}", file=sys.stderr)
            sys.exit(1)

    print("\n[done] Pipeline complete.")
    print(f"       Output: {output_path.resolve()}")


if __name__ == "__main__":
    main()
