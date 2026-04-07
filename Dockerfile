FROM python:3.10-slim

WORKDIR /app

# Install system build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Runtime environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Force matplotlib into headless/non-interactive mode (no display needed)
ENV MPLBACKEND=Agg

# Expose the HuggingFace Spaces standard port
EXPOSE 7860

# Start the OpenEnv server directly via uvicorn
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
