FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment code
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1

# Entrypoint for running inference
ENTRYPOINT ["python", "inference.py"]
