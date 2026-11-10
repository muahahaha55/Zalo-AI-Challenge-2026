# Final submission image, built to the organisers' Docker guide (2026):
#   - base image with CUDA 13.2 (their driver 595.58.03 runs CUDA 12.x and 13.x <= 13.2);
#   - code in /code: predict.sh, predict.py, preprocessing.py, predict_notebook.ipynb, start_jupyter.sh, requirements.txt,
#     README.md, saved_models/, training_code/;
#   - they run: docker run --gpus '"device=0"' -v <data>:/data:ro -v <result>:/result <image> /bin/bash /code/predict.sh
#     (no network) and expect /result/submission.jsonl; so there is NO ENTRYPOINT (it would swallow their command).
#
# Models are NOT in git. Before building, fill models/ (checked by scripts/check_models.py):
#   models/llm   Qwen3-4B fine-tuned (LoRA merged, scripts/merge_lora.py) + tokenizer   -> /code/saved_models/llm
#   models/asr   PhoWhisper-large fine-tuned (merged)                                     -> /code/saved_models/asr
#   (optional) models/adapter  only if the LLM is shipped as base + LoRA instead of merged
#
#   docker build -f docker/Dockerfile -t zac2026-lifemate:v1 .
#   docker save zac2026-lifemate:v1 | gzip > zac2026_<TeamName>.tar.gz && md5sum zac2026_<TeamName>.tar.gz
# TORCH_INDEX selects the CUDA build of torch 2.11.0 (training used cu130). cu128 also covers Ampere (A30) and Blackwell
# (RTX 5090) and runs on older local drivers for testing (start it with -e NVIDIA_DISABLE_REQUIRE=1 on such a machine).
FROM nvidia/cuda:13.2.0-runtime-ubuntu22.04

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
ENV DEBIAN_FRONTEND=noninteractive TZ=Asia/Ho_Chi_Minh PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    LIFEMATE_DATA_DIR=/data LIFEMATE_OUT=/result/submission.jsonl \
    LIFEMATE_LLM=/code/saved_models/llm LIFEMATE_ADAPTER=/code/saved_models/adapter LIFEMATE_ASR=/code/saved_models/asr \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 USE_HUB_KERNELS=NO

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip libsndfile1 tzdata \
 && rm -rf /var/lib/apt/lists/* && python3 -m pip install --upgrade pip

# Dependencies are installed while building (network then) and never at run time.
WORKDIR /code
COPY submission/requirements.txt /code/requirements.txt
RUN python3 -m pip install torch==2.11.0 --index-url ${TORCH_INDEX} \
 && python3 -m pip install -r /code/requirements.txt

# inference code + entry points
COPY src /code/src
COPY submission/ /code/
# training code (reproduction by the organisers; base models and competition data are downloaded / provided separately)
COPY src /code/training_code/src
COPY scripts /code/training_code/scripts
COPY configs /code/training_code/configs
COPY pyproject.toml requirements.txt requirements-kaggle.txt requirements-gpu-linux.txt requirements-inference.txt models.lock.json /code/training_code/
COPY docker/COMMIT.txt /code/training_code/COMMIT.txt
# fine-tuned weights used at inference
COPY models /code/saved_models

RUN chmod +x /code/predict.sh /code/start_jupyter.sh
CMD ["/bin/bash", "/code/predict.sh"]
