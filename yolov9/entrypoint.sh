#!/usr/bin/env bash
set -euo pipefail

# Read prefixed env vars (supporting multiple exporters later)
MODEL_SIZE="${YOLOV9_MODEL_SIZE:-e}"
IMG_SIZE="${YOLOV9_IMG_SIZE:-640}"
OPSET="${YOLOV9_OPSET:-12}"

OUT_DIR="/out"
OUT_FILE="yolov9-${MODEL_SIZE}-${IMG_SIZE}-opset${OPSET}.onnx"

cd /yolov9

echo "[info] YOLOV9_MODEL_SIZE=${MODEL_SIZE}"
echo "[info] YOLOV9_IMG_SIZE=${IMG_SIZE}"
echo "[info] YOLOV9_OPSET=${OPSET}"
echo "[info] Output: ${OUT_DIR}/${OUT_FILE}"

# Download weights
WEIGHTS_URL="https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-${MODEL_SIZE}-converted.pt"
WEIGHTS_LOCAL="yolov9-${MODEL_SIZE}.pt"

echo "[info] Downloading weights: ${WEIGHTS_URL}"
curl -fsSL "${WEIGHTS_URL}" -o "${WEIGHTS_LOCAL}"

# Patch torch.load weights_only behavior
echo "[info] Patching models/experimental.py torch.load weights_only=False"
sed -i "s/ckpt = torch.load(attempt_download(w), map_location='cpu')/ckpt = torch.load(attempt_download(w), map_location='cpu', weights_only=False)/g" models/experimental.py

echo "[info] Exporting ONNX..."
python3 - <<PY
import torch
from models.experimental import attempt_load

print("Torch version:", torch.__version__)

weights = "${WEIGHTS_LOCAL}"
imgsz = int("${IMG_SIZE}")
opset = int("${OPSET}")
out = "${OUT_FILE}"

# IMPORTANT: disable fuse/inplace to avoid non-leaf tensor error
model = attempt_load(weights, device="cpu", inplace=False, fuse=False)
model.eval()

# YOLO export convention (harmless if absent)
try:
    if hasattr(model, "model") and hasattr(model.model[-1], "export"):
        model.model[-1].export = True
except Exception:
    pass

x = torch.zeros(1, 3, imgsz, imgsz)

kwargs = dict(
    opset_version=opset,
    do_constant_folding=True,
    input_names=["images"],
    output_names=["output"],
)

# Force legacy exporter (avoid dynamo/pt2 exporter)
try:
    torch.onnx.export(model, x, out, **kwargs, dynamo=False)
except TypeError:
    torch.onnx.export(model, x, out, **kwargs)

import onnx
m = onnx.load(out)
print("ONNX opset_import:", [(o.domain, o.version) for o in m.opset_import])
PY

mkdir -p "${OUT_DIR}"
cp -f "/yolov9/${OUT_FILE}" "${OUT_DIR}/${OUT_FILE}"

echo "[ok] Wrote ${OUT_DIR}/${OUT_FILE}"
ls -lh "${OUT_DIR}/${OUT_FILE}"
