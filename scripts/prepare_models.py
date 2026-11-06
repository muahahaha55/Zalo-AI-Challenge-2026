"""Prepare the base models for the PRIVATE Kaggle dataset `lifemate-models` (run locally, needs Internet).

  .venv/Scripts/python.exe -I -X utf8 scripts/prepare_models.py

Output (git-ignored): kaggle_upload/lifemate-models/{qwen3-4b,phowhisper-large}/ + manifest.json
- Qwen/Qwen3-4B: official safetensors, copied as is.
- vinai/PhoWhisper-large: the official repo ships only pytorch_model.bin (pickle). It is loaded once here (torch weights_only),
  verified, and re-saved as safetensors, so the Kaggle notebook never loads a pickle file (docs/RULES.md 3.2).
Only public model files are downloaded; no competition data is involved. Parameters are counted with sum(p.numel()).
"""
import hashlib
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoConfig, AutoModelForCausalLM, WhisperForConditionalGeneration, WhisperProcessor

OUT = Path("kaggle_upload/lifemate-models")
TMP = Path("kaggle_upload/_tmp_phowhisper")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def info(repo: str) -> dict:
    i = HfApi().model_info(repo)
    return {"repo": repo, "revision": i.sha, "last_modified": str(i.last_modified),
            "license": [t for t in i.tags if t.startswith("license")]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}

    # --- LLM: safetensors straight from the official repo
    q = info("Qwen/Qwen3-4B")
    d = OUT / "qwen3-4b"
    snapshot_download("Qwen/Qwen3-4B", revision=q["revision"], local_dir=d,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "LICENSE*"])
    with torch.device("meta"):
        q["params_B"] = sum(p.numel() for p in AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(d)).parameters()) / 1e9
    manifest["qwen3-4b"] = q
    print("Qwen3-4B params (B):", q["params_B"], flush=True)

    # --- ASR: bin -> safetensors
    p = info("vinai/PhoWhisper-large")
    snapshot_download("vinai/PhoWhisper-large", revision=p["revision"], local_dir=TMP,
                      allow_patterns=["*.json", "*.txt", "pytorch_model.bin"])
    model = WhisperForConditionalGeneration.from_pretrained(TMP, torch_dtype=torch.float32, use_safetensors=False)
    p["params_B"] = sum(x.numel() for x in model.parameters()) / 1e9
    d = OUT / "phowhisper-large"
    d.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(d, safe_serialization=True)
    WhisperProcessor.from_pretrained(TMP).save_pretrained(d)
    back = WhisperForConditionalGeneration.from_pretrained(d, torch_dtype=torch.float32, use_safetensors=True)
    sd0, sd1 = model.state_dict(), back.state_dict()
    assert sd0.keys() == sd1.keys() and all(torch.equal(sd0[k], sd1[k]) for k in sd0), "safetensors round-trip differs"
    print("PhoWhisper-large params (B):", p["params_B"], "| safetensors round-trip identical", flush=True)
    manifest["phowhisper-large"] = p
    del model, back
    shutil.rmtree(TMP, ignore_errors=True)

    # --- checksums of every weight file (copy into docs/MODELS.md); drop the HF download metadata folders
    for name in manifest:
        shutil.rmtree(OUT / name / ".cache", ignore_errors=True)
        manifest[name]["sha256"] = {f.name: sha256(f) for f in sorted((OUT / name).glob("*.safetensors"))}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
