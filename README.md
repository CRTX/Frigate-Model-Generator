# Frigate model exporter

First change the .env values to the parameters you want to use to create the model onnx file

## To export yolo-nas:

```sh
docker compose run --rm yolonas-export
```

## To export yolo-v9:

```sh
docker compose run --rm yolov9-export
```

## To use yolo-nas in your frigate config add:
### config.yaml
```yaml
model:
  model_type: yolo-generic
  width: 640
  height: 640
  input_tensor: nchw
  input_dtype: float
  path: /config/model_cache/yolov9-e-640-opset12.onnx
  labelmap_path: /labelmap/coco-80.txt
```

## or to use yolo-nas instead
### config.yaml

```yaml
model:
  model_type: yolonas
  width: 640
  height: 640
  input_pixel_format: bgr
  input_tensor: nchw
  path: /config/model_cache/yolonas_l_640.onnx
  labelmap_path: /labelmap/coco-80.txt

```