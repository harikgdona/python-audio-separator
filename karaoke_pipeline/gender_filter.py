"""
Gender Filter — Hybrid pitch + audeering model classifier
Fixes Indian male singer misclassification by combining:
  1. audeering/wav2vec2-large-robust-24-ft-age-gender (model)
  2. Fundamental frequency (F0) pitch analysis
Indian male singers (Armaan Malik, SPB etc.) sing at 165-260 Hz,
well above Western male speech (80-165 Hz), causing model-only
approaches to misclassify them as female.
"""

import json
import sys
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from pathlib import Path
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

MODEL_ID      = "audeering/wav2vec2-large-robust-24-ft-age-gender"
GENDER_LABELS = ["child", "female", "male"]

# Pitch thresholds tuned for Indian film music
F0_MALE_MAX   = 180   # Hz — below this is clearly male
F0_FEMALE_MIN = 290   # Hz — above this is clearly female
# 180–290 Hz is the Indian male singer overlap zone — use weighted blend


def _jprint(event: dict):
    print("JSON:" + json.dumps(event), flush=True)


# ── Custom audeering model architecture ──────────────────────────────────────

class ModelHead(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.dense    = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout  = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    _tied_weights_keys = []

    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age      = ModelHead(config, 1)
        self.gender   = ModelHead(config, 3)
        self.post_init()

    def forward(self, input_values):
        hidden      = self.wav2vec2(input_values)[0]
        hidden_mean = torch.mean(hidden, dim=1)
        logits_age  = self.age(hidden_mean)
        logits_gen  = torch.softmax(self.gender(hidden_mean), dim=-1)
        return hidden_mean, logits_age, logits_gen


# ── Pitch (F0) estimation via autocorrelation ─────────────────────────────────

def estimate_f0(segment_np: np.ndarray, sr: int) -> float:
    """Return median fundamental frequency of a segment in Hz.
       Returns 0 if no clear pitch found (silence / music only)."""
    frame_len = int(0.04 * sr)   # 40 ms
    hop       = int(0.01 * sr)   # 10 ms
    f0s       = []

    for i in range(0, len(segment_np) - frame_len, hop):
        frame = segment_np[i : i + frame_len].copy()
        frame -= frame.mean()
        if frame.std() < 0.005:
            continue
        corr    = np.correlate(frame, frame, mode='full')[frame_len - 1:]
        min_lag = max(1, int(sr / 1000))
        max_lag = int(sr / 65)
        if max_lag >= len(corr):
            continue
        peak = np.argmax(corr[min_lag:max_lag]) + min_lag
        f0   = sr / peak
        if 80 < f0 < 900:
            f0s.append(f0)

    return float(np.median(f0s)) if f0s else 0.0


def pitch_gender_score(f0_hz: float) -> dict:
    """
    Returns {'male': prob, 'female': prob} based purely on pitch.
    Tuned for Indian film music — acknowledges that Indian male
    singers routinely hit 180-260 Hz.
    """
    if f0_hz <= 0:
        return {"male": 0.5, "female": 0.5}   # unknown

    if f0_hz < F0_MALE_MAX:
        male_p = 0.92
    elif f0_hz > F0_FEMALE_MIN:
        male_p = 0.08
    else:
        # Linear blend in overlap zone 180–290 Hz
        t      = (f0_hz - F0_MALE_MAX) / (F0_FEMALE_MIN - F0_MALE_MAX)
        male_p = 0.92 - t * 0.84   # 0.92 → 0.08

    return {"male": male_p, "female": 1.0 - male_p}


# ── Load model ────────────────────────────────────────────────────────────────

def load_classifier(device: torch.device):
    _jprint({"event": "stage", "name": "gender", "status": "loading"})
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model     = AgeGenderModel.from_pretrained(MODEL_ID)
    model.to(device).eval()
    _jprint({"event": "stage", "name": "gender", "status": "ready"})
    return processor, model


# ── Classify segments ─────────────────────────────────────────────────────────

def classify_vocal_segments(
    vocals_path: Path,
    segment_sec: float,
    threshold: float,
    device: torch.device,
    remove: str = "male",
) -> list:
    processor, model = load_classifier(device)

    waveform, sr = torchaudio.load(str(vocals_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000

    total_samples = waveform.shape[1]
    seg_samples   = int(segment_sec * sr)
    segments      = []

    _jprint({"event": "stage", "name": "gender", "status": "scanning",
             "duration": round(total_samples / sr, 2)})

    for start in range(0, total_samples, seg_samples):
        end   = min(start + seg_samples, total_samples)
        chunk = waveform[:, start:end]
        if chunk.shape[1] < sr * 0.5:
            continue

        audio_np = chunk.squeeze().numpy()

        # ── 1. Model prediction ──────────────────────────────────────
        inputs = processor(audio_np, sampling_rate=16000,
                           return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            _, _, logits_gen = model(inputs["input_values"])
        probs     = logits_gen[0].cpu().numpy()
        mdl_male  = float(probs[GENDER_LABELS.index("male")])
        mdl_fem   = float(probs[GENDER_LABELS.index("female")] +
                          probs[GENDER_LABELS.index("child")])
        total_mf  = mdl_male + mdl_fem + 1e-9
        mdl_male /= total_mf
        mdl_fem  /= total_mf

        # ── 2. Pitch estimation ──────────────────────────────────────
        f0       = estimate_f0(audio_np, sr)
        pitch_sc = pitch_gender_score(f0)

        # ── 3. Weighted blend  (60% model, 40% pitch) ────────────────
        # In the Indian male singer zone, pitch gets more weight
        if F0_MALE_MAX <= f0 <= F0_FEMALE_MIN:
            w_model, w_pitch = 0.45, 0.55   # pitch wins in overlap zone
        else:
            w_model, w_pitch = 0.65, 0.35

        final_male = w_model * mdl_male + w_pitch * pitch_sc["male"]
        final_fem  = w_model * mdl_fem  + w_pitch * pitch_sc["female"]

        predicted  = "male" if final_male >= final_fem else "female"
        confidence = final_male if predicted == "male" else final_fem
        will_remove = (predicted == remove.lower() and confidence >= threshold)

        start_sec = round(start / sr, 3)
        end_sec   = round(end   / sr, 3)

        seg = {
            "start":      start_sec,
            "end":        end_sec,
            "gender":     predicted,
            "confidence": round(confidence, 3),
            "f0_hz":      round(f0, 1),
            "remove":     will_remove,
        }
        segments.append(seg)
        _jprint({"event": "segment", **seg})

    removed = sum(1 for s in segments if s["remove"])
    _jprint({"event": "stage", "name": "gender", "status": "done",
             "total": len(segments), "removed": removed})
    return segments


# ── Suppress segments ─────────────────────────────────────────────────────────

def suppress_segments(
    vocals_path: Path,
    segments: list,
    output_path: Path,
) -> None:
    from pydub import AudioSegment
    _jprint({"event": "stage", "name": "suppress", "status": "start"})
    audio     = AudioSegment.from_wav(str(vocals_path))
    to_remove = [s for s in segments if s["remove"]]

    for seg in to_remove:
        start_ms = int(seg["start"] * 1000)
        end_ms   = int(seg["end"]   * 1000)
        silence  = AudioSegment.silent(
            duration=end_ms - start_ms, frame_rate=audio.frame_rate
        )
        audio = audio[:start_ms] + silence + audio[end_ms:]

    audio.export(str(output_path), format="wav")
    _jprint({"event": "stage", "name": "suppress", "status": "done",
             "clean_vocals": str(output_path),
             "removed_count": len(to_remove)})
