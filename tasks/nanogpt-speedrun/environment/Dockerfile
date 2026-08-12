FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/usr/local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/data/hf_cache
ENV TORCH_HOME=/data/torch_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python3

RUN python -m pip install --break-system-packages --upgrade pip && \
    python -m pip install --break-system-packages \
        "torch==2.11" \
        "huggingface_hub>=0.24" \
        numpy tqdm

WORKDIR /app/environment
COPY train_gpt_simple.py ./train_gpt_simple.py
COPY cached_fineweb10B.py ./data/cached_fineweb10B.py
COPY track3_README.md ./track3_README.md

RUN mkdir -p /data/hf_cache /data/torch_cache /logs/artifacts/runs /logs/verifier
