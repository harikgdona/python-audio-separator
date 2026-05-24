from pathlib import Path
from pydub import AudioSegment


def mix_stems(
    clean_vocals_path: Path,
    stem_paths: dict,
    output_path: Path,
) -> None:
    print(f"\n[mix] Mixing stems into {output_path.name}...")
    mix = AudioSegment.from_wav(str(clean_vocals_path))
    print(f"[mix]   + vocals (cleaned)  {len(mix)/1000:.1f}s")

    non_vocal_stems = [s for s in stem_paths if s != "vocals"]
    for stem_name in non_vocal_stems:
        stem_file = stem_paths[stem_name]
        part = AudioSegment.from_wav(str(stem_file))
        if len(part) < len(mix):
            part = part + AudioSegment.silent(
                duration=len(mix) - len(part), frame_rate=part.frame_rate
            )
        elif len(part) > len(mix):
            mix = mix + AudioSegment.silent(
                duration=len(part) - len(mix), frame_rate=mix.frame_rate
            )
        mix = mix.overlay(part)
        print(f"[mix]   + {stem_name}  {len(part)/1000:.1f}s")

    mix.export(str(output_path), format="wav")
    print(f"[mix] Final output: {output_path.resolve()}")
