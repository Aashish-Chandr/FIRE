FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pin numpy<2 before anything else to avoid binary incompatibility
RUN pip install --no-cache-dir "numpy<2"

# Install CPU-only torch (avoids pulling 2GB CUDA build)
RUN pip install --no-cache-dir \
    "torch>=2.4.0" \
    --index-url https://download.pytorch.org/whl/cpu

# Install all other dependencies
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source files
COPY backend/ ./backend/
COPY frontend/ ./frontend/

WORKDIR /app/backend

# Build FAISS index from knowledge base
RUN python rag/ingest.py

# Shrink image
RUN pip cache purge && \
    find / -name "*.pyc" -delete 2>/dev/null || true

EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
