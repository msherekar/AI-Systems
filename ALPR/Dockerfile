# Use multi-stage build to reduce image size
FROM python:3.11-slim AS builder

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install build dependencies and Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

# Second stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r alpr && useradd -r -g alpr alpr

# Copy application code
COPY . .

# Set proper permissions
RUN chown -R alpr:alpr /app

# Create directory for results
RUN mkdir -p /app/results && chown -R alpr:alpr /app/results

# Switch to non-root user
USER alpr

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8080/health') or exit(1)"

# Set environment variables
ENV PYTHONPATH=/app

# Expose API port
EXPOSE 8080
EXPOSE 23002/udp

# Set entrypoint
ENTRYPOINT ["python", "-m", "src.app"]

# Default command (can be overridden)
CMD ["--mode", "stream"]
