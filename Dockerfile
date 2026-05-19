# FER2013 Emotion Recognition — Streamlit app
# Portable image: runs on a generic VM, Aliyun/Tencent Cloud, Cloud Run,
# Hugging Face Spaces (Docker SDK), Zeabur, Sealos, etc.
FROM python:3.10-slim

WORKDIR /app

# opencv-python-headless still links libGL.so.1 at import time on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install the CPU-only torch build first. This keeps the image ~1 GB instead
# of ~5 GB (no bundled CUDA), which the demo does not need for inference.
# Once satisfied, `pip install -r requirements.txt` will NOT pull the CUDA build.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.1" "torchvision>=0.16" \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

# Hosts inject the port via $PORT (HF Spaces=7860, Cloud Run=8080, …).
# The CLI flag overrides .streamlit/config.toml, so this stays correct anywhere.
ENV PORT=8501
EXPOSE 8501

CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true"]
