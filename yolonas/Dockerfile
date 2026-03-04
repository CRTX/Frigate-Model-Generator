FROM python:3.10-slim-bullseye

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    protobuf-compiler \
    libprotobuf-dev \
    build-essential \
    g++ \
    git \
    libgl1 \
    libglib2.0-0 \
    curl \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# CPU-only torch/torchvision (no CUDA)
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    "torch==2.2.2+cpu" "torchvision==0.17.2+cpu"

# --- Pin to what SG 3.5.0 wants (so it won't try to "fix" at runtime) ---
RUN pip install --no-cache-dir \
    "numpy<=1.23.5" \
    "protobuf==3.20.3" \
    "onnx==1.13.0" \
    "onnxruntime==1.13.1" \
    "tqdm>=4.57.0" \
    "treelib==1.6.1" \
    "hydra-core>=1.2.0" \
    "omegaconf==2.3.0" \
    "PyYAML" \
    "packaging" \
    "rich" \
    "psutil" \
    boto3 botocore s3transfer

# Install SG normally (WITH deps resolution now harmless because we pre-pinned)
RUN pip install --no-cache-dir "super-gradients==3.5.0"

# onnxsim is optional; keep it, but it will pull rich anyway
RUN pip install --no-cache-dir onnxsim

ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
