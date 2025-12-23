FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

RUN pip install -e .

RUN useradd -m -u 1000 appuser
USER appuser

CMD ["python", "scripts/run_pipeline.py"]