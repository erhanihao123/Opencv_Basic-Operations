FROM python:3.9-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn \
    LOG_DIR=/app/logs

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    libice6 \
    libgthread-2.0-0 \
    libgl1 \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/backend/models /app/logs

COPY backend/models/MobileNetSSD_deploy.prototxt /app/backend/models/
COPY backend/models/MobileNetSSD_deploy.caffemodel /app/backend/models/

RUN cd /app/backend/models && \
    wget -q --timeout=30 -O haarcascade_frontalface_default.xml \
        "https://cdn.jsdelivr.net/gh/opencv/opencv@master/data/haarcascades/haarcascade_frontalface_default.xml" || true && \
    wget -q --timeout=30 -O haarcascade_frontalface_alt2.xml \
        "https://cdn.jsdelivr.net/gh/opencv/opencv@master/data/haarcascades/haarcascade_frontalface_alt2.xml" || true && \
    wget -q --timeout=30 -O haarcascade_frontalface_alt.xml \
        "https://cdn.jsdelivr.net/gh/opencv/opencv@master/data/haarcascades/haarcascade_frontalface_alt.xml" || true && \
    wget -q --timeout=30 -O haarcascade_frontalface_alt_tree.xml \
        "https://cdn.jsdelivr.net/gh/opencv/opencv@master/data/haarcascades/haarcascade_frontalface_alt_tree.xml" || true && \
    wget -q --timeout=30 -O haarcascade_eye.xml \
        "https://cdn.jsdelivr.net/gh/opencv/opencv@master/data/haarcascades/haarcascade_eye.xml" || true

COPY . .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
