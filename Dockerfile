FROM python:3.11-slim

# System deps needed by pdfplumber / PyMuPDF / sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer-cached until requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Data directories (populated at runtime via volume mount)
RUN mkdir -p data/raw data/processed data/chroma_db data/eval

# FastAPI runs on 8000, Streamlit on 8501
EXPOSE 8000 8501
