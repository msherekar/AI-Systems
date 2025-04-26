FROM python:3.11.3-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"

# Copy the rest of the application
COPY . .

# Create directories if they don't exist
RUN mkdir -p logs models/

# Expose the port the app runs on
EXPOSE 8786

# Set environment variables
ENV FLASK_PORT=8786
ENV FLASK_HOST=0.0.0.0
ENV FLASK_DEBUG=False

# Command to run the application
CMD ["python", "src/deployment/deployment.py"]

