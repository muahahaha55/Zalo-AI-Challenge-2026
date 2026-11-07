"""Whisper-family speech recognition (PhoWhisper), greedy and deterministic.

Used by scripts/run_asr.py (whole splits, resumable) and by the container entry point (one sample at a time).
torch / transformers / soundfile are imported only when a transcriber is created, so importing this module is cheap.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

SAMPLE_RATE = 16000
MAX_NEW_TOKENS = 160
RETRY_MAX_NEW_TOKENS = 100


def degenerate(text: str) -> bool:
    """True when the output loops (some 3-gram repeated 3+ times): a known Whisper failure on noisy audio."""
    words = text.split()
    grams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
    return bool(grams) and max(grams.count(g) for g in set(grams)) >= 3


class WhisperTranscriber:
    """Greedy Whisper decoding. `transcribe` takes any number of files and processes them in batches of `batch_size`."""

    def __init__(self, model_path: str | Path, batch_size: int = 8, device: str = "cuda"):
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self._torch = torch
        self._device = device
        self.batch_size = batch_size
        self.processor = WhisperProcessor.from_pretrained(str(model_path))
        self.model = WhisperForConditionalGeneration.from_pretrained(str(model_path), torch_dtype=torch.float16).to(device).eval()

    def param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @staticmethod
    def read_wav(path: str | Path):
        import soundfile as sf
        audio, rate = sf.read(path, dtype="float32")
        assert rate == SAMPLE_RATE, (path, rate)
        return audio if audio.ndim == 1 else audio.mean(1)

    def _generate(self, features, **extra):
        with self._torch.no_grad():
            return self.model.generate(features, language="vi", task="transcribe", num_beams=1, do_sample=False, **extra)

    def transcribe(self, wav_paths: Sequence[str | Path]) -> list[str]:
        texts: list[str] = []
        for start in range(0, len(wav_paths), self.batch_size):
            batch = [self.read_wav(p) for p in wav_paths[start : start + self.batch_size]]
            features = self.processor.feature_extractor(batch, sampling_rate=SAMPLE_RATE, return_tensors="pt")
            features = features.input_features.to(self._device).half()
            decoded = self.processor.batch_decode(self._generate(features, max_new_tokens=MAX_NEW_TOKENS), skip_special_tokens=True)
            for k, text in enumerate(decoded):          # redo a looping item alone with an n-gram block
                if degenerate(text):
                    retry = self._generate(features[k : k + 1], max_new_tokens=RETRY_MAX_NEW_TOKENS, no_repeat_ngram_size=4)
                    decoded[k] = self.processor.batch_decode(retry, skip_special_tokens=True)[0]
            texts.extend(t.strip() for t in decoded)
        return texts
