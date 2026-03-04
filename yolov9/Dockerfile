FROM python:3.11

RUN apt-get update && apt-get install --no-install-recommends -y \
    libgl1 curl \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.10.4 /uv /bin/

RUN mkdir -p /yolov9
WORKDIR /yolov9
ADD https://github.com/WongKinYiu/yolov9.git .

RUN uv pip install --system -r requirements.txt
RUN uv pip install --system onnx==1.18.0 onnxruntime onnx-simplifier==0.4.* onnxscript

RUN sed -i "s/ckpt = torch.load(attempt_download(w), map_location='cpu')/ckpt = torch.load(attempt_download(w), map_location='cpu', weights_only=False)/g" models/experimental.py

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash", "-lc", "true"]