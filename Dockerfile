# Mandated ROCm base image (checked by layer identity).
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app

COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
 && pip install --no-cache-dir --no-deps -U transformers tokenizers safetensors huggingface_hub \
 && python3 -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); assert 'rocm' in torch.__version__, 'pip replaced ROCm torch!'" \
 && python3 -c "import transformers; print('transformers', transformers.__version__)"

# Weights are shipped IN the image (no network at evaluation time).
#   huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir models/vlm
COPY models/ /models/
ENV MC3_MODEL_DIR=/models/vlm \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTHONUNBUFFERED=1

COPY app/ /app/
RUN mkdir -p /app/corpus /app/output /app/index

# The container CMD starts the resident server (model loads NOW, in the startup budget,
# and once), then keeps the container alive. `app.py --index` / `--query` talk to it.
CMD ["sh", "-c", "python3 /app/server.py >/tmp/mc3_server.log 2>&1 & exec sleep infinity"]
