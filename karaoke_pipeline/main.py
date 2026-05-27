"""
Karaoke Pipeline — Gender Voice Remover
Emits JSON: events to stdout so the web UI can parse rich progress data.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


def _jprint(event: dict):
    print("JSON:" + json.dumps(event), flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",     required=True)
    p.add_argument("--output",    default=None)
    p.add_argument("--remove",    default="male", choices=["male","female"])
    p.add_argument("--segment",   type=float, default=3.0)
    p.add_argument("--threshold", type=float, default=0.60)
    p.add_argument("--stems-dir", default=None,
                   help="Directory to persist stem files (for UI visualisation).")
    p.add_argument("--cpu",       action="store_true")
    return p.parse_args()


def select_device(force_cpu):
    if force_cpu or not torch.cuda.is_available():
        device = torch.device("cpu")
        _jprint({"event": "device", "mode": "cpu"})
    else:
        device = torch.device("cuda")
        _jprint({"event": "device", "mode": "cuda",
                 "name": torch.cuda.get_device_name(0)})
    return device


def separate_stems(input_path, out_dir, device):
    _jprint({"event": "stage", "name": "demucs", "status": "start"})
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "demucs",
        "--name", "htdemucs_6s",
        "--out",  str(out_dir),
        "--device", str(device),
        str(input_path),
    ]
    subprocess.run(cmd, check=True)

    track_dir = out_dir / "htdemucs_6s" / input_path.stem
    if not track_dir.exists():
        raise RuntimeError(f"Demucs output not found: {track_dir}")

    stem_paths = {}
    for f in sorted(track_dir.glob("*.wav")):
        stem_paths[f.stem] = f

    if "vocals" not in stem_paths:
        raise RuntimeError("No vocals stem produced by Demucs.")

    _jprint({"event": "stage",  "name": "demucs", "status": "done",
             "stems": {k: str(v) for k, v in stem_paths.items()}})
    return stem_paths


def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[error] File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else \
        input_path.parent / f"output_no_{args.remove}.wav"

    device = select_device(args.cpu)

    _jprint({"event": "info", "input":  str(input_path),
             "remove": args.remove, "output": str(output_path)})

    # Use caller-supplied stems dir so UI can read the files, else temp
    use_temp  = args.stems_dir is None
    stems_dir = Path(args.stems_dir) if args.stems_dir else \
        Path(tempfile.mkdtemp(prefix="karaoke_stems_"))

    try:
        # Step 1 — Demucs
        stem_paths = separate_stems(input_path, stems_dir / "stems", device)

        # Step 2 — Gender classification
        here = Path(__file__).parent
        sys.path.insert(0, str(here))
        from gender_filter import classify_vocal_segments, suppress_segments

        segments = classify_vocal_segments(
            stem_paths["vocals"],
            segment_sec=args.segment,
            threshold=args.threshold,
            device=device,
            remove=args.remove,
        )

        # Step 3 — Suppress
        clean_vocals = stems_dir / "vocals_clean.wav"
        suppress_segments(stem_paths["vocals"], segments, clean_vocals)

        # Step 4 — Mix
        _jprint({"event": "stage", "name": "mix", "status": "start"})
        from mixer import mix_stems
        mix_stems(clean_vocals, stem_paths, output_path)
        _jprint({"event": "stage",  "name": "mix",  "status": "done"})
        _jprint({"event": "done", "output": str(output_path)})

    finally:
        if use_temp:
            shutil.rmtree(stems_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
