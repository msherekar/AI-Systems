# Use the official Python image from the Docker Hub with multi-platform support
FROM --platform=$TARGETPLATFORM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install build dependencies for packages that might need compilation on ARM64
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
# Use special handling for PyTorch on ARM64
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["uvicorn", "src.api.service:app", "--host", "0.0.0.0", "--port", "80"]