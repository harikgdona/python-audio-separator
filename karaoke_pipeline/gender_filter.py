import torch
import torchaudio
from pathlib import Path
from speechbrain.inference.classifiers import EncoderClassifier


def load_classifier(device: torch.device) -> EncoderClassifier:
    print("[gender] Loading SpeechBrain gender classifier...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/gender-recognition-wav2vec2",
        run_opts={"device": str(device)},
    )
    print("[gender] Classifier loaded.")
    return classifier


def classify_vocal_segments(
    vocals_path: Path,
    segment_sec: float,
    threshold: float,
    device: torch.device,
    remove: str = "male",
) -> list[tuple[float, float]]:
    classifier = load_classifier(device)
    waveform, sr = torchaudio.load(str(vocals_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000

    total_samples = waveform.shape[1]
    seg_samples = int(segment_sec * sr)
    target_label = remove.lower()
    matched_segments: list[tuple[float, float]] = []

    print(f"[gender] Scanning vocals for '{target_label}' segments...")
    for start in range(0, total_samples, seg_samples):
        end = min(start + seg_samples, total_samples)
        chunk = waveform[:, start:end]
        if chunk.shape[1] < sr * 0.5:
            continue
        out_prob, score, index, label = classifier.classify_batch(chunk)
        predicted = label[0].strip().lower()
        confidence = out_prob[0].max().item()
        start_sec = start / sr
        end_sec = end / sr
        if predicted == target_label and confidence >= threshold:
            matched_segments.append((start_sec, end_sec))
            print(f"[gender]   [{start_sec:.1f}s - {end_sec:.1f}s] {predicted} ({confidence:.2f}) -> REMOVE")
        else:
            print(f"[gender]   [{start_sec:.1f}s - {end_sec:.1f}s] {predicted} ({confidence:.2f}) -> keep")

    print(f"[gender] Found {len(matched_segments)} '{target_label}' segments to remove.")
    return matched_segments


def suppress_segments(
    vocals_path: Path,
    segments: list[tuple[float, float]],
    output_path: Path,
) -> None:
    from pydub import AudioSegment
    print(f"[gender] Suppressing {len(segments)} segments...")
    audio = AudioSegment.from_wav(str(vocals_path))
    for start_sec, end_sec in segments:
        start_ms = int(start_sec * 1000)
        end_ms = int(end_sec * 1000)
        silence = AudioSegment.silent(duration=end_ms - start_ms, frame_rate=audio.frame_rate)
        audio = audio[:start_ms] + silence + audio[end_ms:]
    audio.export(str(output_path), format="wav")
    print(f"[gender] Clean vocals saved to {output_path}")
