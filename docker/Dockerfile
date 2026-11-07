# Final-solution image: ASR (PhoWhisper, fine-tuned) + LLM (Qwen3-4B base + LoRA adapter), fully offline at run time.
# UNVERIFIED: never built (no Docker daemon was available, and the organisers' packaging guide has not been received).
# Treat it as a starting point and re-check against their guide. See docs/DOCKER.md.
#
# Models are NOT in git. Prepare a folder `models/` next to this repo before building:
#   models/llm      Qwen3-4B base weights (safetensors) + tokenizer files
#   models/adapter  the trained LoRA adapter (adapter_model.safetensors + adapter_config.json), or leave out and run with --adapter none
#   models/asr      the merged fine-tuned PhoWhisper-large folder
#
#   docker build -f docker/Dockerfile -t lifemate .
#   docker run --rm --gpus all --network none -v <split folder>:/data -v <empty folder>:/app/result lifemate
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    LIFEMATE_DATA_DIR=/data LIFEMATE_LLM=/models/llm LIFEMATE_ADAPTER=/models/adapter LIFEMATE_ASR=/models/asr \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 USE_HUB_KERNELS=NO

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*

# Dependencies are installed while building (needs network then) and never at run time.
# Pin the versions to those of the training environment before the final build (see docs/DOCKER.md).
WORKDIR /app
COPY requirements-inference.txt ./
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124 \
 && pip install --no-cache-dir -r requirements-inference.txt

COPY src ./src
COPY scripts/run_container.py ./scripts/run_container.py
COPY models /models

ENTRYPOINT ["python", "-I", "-X", "utf8", "/app/scripts/run_container.py"]
