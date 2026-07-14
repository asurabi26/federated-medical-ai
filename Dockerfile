FROM python:3.11-slim

WORKDIR /app

# System deps needed by faiss and sentence-transformers' tokenizers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first, explicitly - this is the single biggest size saver.
# The default `pip install torch` pulls CUDA libraries even when there's no GPU on the
# deploy target, which is most of what pushed this over Vercel's 500MB limit in the
# first place. This wheel is CPU-only and roughly 1/5th the size.
RUN pip install --no-cache-dir torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
