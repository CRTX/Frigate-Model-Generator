#!/usr/bin/env bash
set -euo pipefail

: "${YOLONAS_IMG_SIZE:?YOLONAS_IMG_SIZE not set}"
: "${YOLONAS_SIZE:?YOLONAS_SIZE not set}"   # s|m|l

OUT_DIR="/out"

size="$(echo "${YOLONAS_SIZE}" | tr '[:upper:]' '[:lower:]')"
if [[ "${size}" != "s" && "${size}" != "m" && "${size}" != "l" ]]; then
  echo "ERROR: YOLONAS_SIZE must be one of: s, m, l (got: ${YOLONAS_SIZE})" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"

CKPT_DIR="/root/.cache/torch/hub/checkpoints"
CKPT_FILE="${CKPT_DIR}/yolo_nas_${size}_coco.pth"
mkdir -p "${CKPT_DIR}"

HF_BASE="https://huggingface.co/bdsqlsz/YOLO_NAS/resolve/5a86ddf143600a401ebd3db9b1d3e2c79f3e2d86"

if [[ ! -f "${CKPT_FILE}" ]]; then
  echo "Downloading checkpoint to: ${CKPT_FILE}"
  curl -fL --retry 5 --retry-delay 2 \
    -o "${CKPT_FILE}" \
    "${HF_BASE}/yolo_nas_${size}_coco.pth?download=true"
fi

python - <<'PY'
import os
import torch
from super_gradients.training import models
from super_gradients.common.object_names import Models
from super_gradients.conversion import DetectionOutputFormatMode

size = os.environ["YOLONAS_SIZE"].strip().lower()   # s/m/l
img = int(os.environ["YOLONAS_IMG_SIZE"])                  # 640 etc

size_to_model = {
    "s": Models.YOLO_NAS_S,
    "m": Models.YOLO_NAS_M,
    "l": Models.YOLO_NAS_L,
}
if size not in size_to_model:
    raise SystemExit(f"Bad YOLONAS_SIZE={size}. Use s, m, or l.")

input_model_path = f"/root/.cache/torch/hub/checkpoints/yolo_nas_{size}_coco.pth"
output_model_name = f"/out/yolonas_{size}_{img}.onnx"
input_shape = (img, img)

print("Exporting YOLO-NAS -> ONNX (Frigate FLAT_FORMAT)")
print("weights:", input_model_path)
print("out:", output_model_name)
print("shape:", input_shape)

# 1) Init architecture
model = models.get(size_to_model[size], num_classes=80, checkpoint_num_classes=80)

# 2) Preprocess pipeline (Frigate-friendly)
model.set_dataset_processing_params(
    class_names=["object"] * 80,
    conf=0.25,
    iou=0.45,
    image_processor={
        "ComposeProcessing": {
            "processings": [
                {"NormalizeImage": {"mean": [0.0, 0.0, 0.0], "std": [255.0, 255.0, 255.0]}},
                {"ImagePermute": {"permutation": [2, 0, 1]}}
            ]
        }
    }
)

# 3) Load weights (cpu)
checkpoint = torch.load(input_model_path, map_location="cpu")
if isinstance(checkpoint, dict):
    if "net" in checkpoint:
        model.load_state_dict(checkpoint["net"])
    elif "ema_net" in checkpoint:
        model.load_state_dict(checkpoint["ema_net"])
    else:
        model.load_state_dict(checkpoint)
else:
    model.load_state_dict(checkpoint.state_dict())

model.eval()

# 4) Export with FLAT_FORMAT
model.export(
    output_model_name,
    output_predictions_format=DetectionOutputFormatMode.FLAT_FORMAT,
    input_image_shape=input_shape,
    confidence_threshold=0.25,
    nms_threshold=0.45,
)

print("DONE")
PY
