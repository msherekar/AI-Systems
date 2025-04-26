FROM python:3.11.3-slim

# Install system dependencies (curl for healthcheck)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy configuration
COPY config/ ./config/

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY models/ ./models/

# Make scripts executable
RUN chmod +x ./scripts/*.py

# Create a non-root user to run the application
RUN useradd -m appuser && \
    chown -R appuser:appuser /app
USER appuser

# Set environment variables
ENV PORT=5000
ENV PYTHONUNBUFFERED=1

# Add health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose the port
EXPOSE ${PORT}

# Run the application
ENTRYPOINT ["python3"]
CMD ["scripts/deploy.py"]