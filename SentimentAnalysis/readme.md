# Sentiment Analysis System

A robust, production-ready sentiment analysis system that analyzes movie reviews and predicts sentiment ratings.

## Overview

This system analyzes text from movie reviews to predict sentiment ratings. It uses natural language processing (NLP) techniques to process text data and machine learning models to classify sentiment.

### Problem Statement

To create a sentiment analysis system for analyzing movie reviews that provides a more nuanced and reliable rating system than traditional star ratings.

### Value Proposition

An accurate sentiment analysis system enables better prediction of movie ratings, capturing nuanced feedback that might be missed in simple star ratings.

## Project Structure

```
SentimentAnalysis/
├── data/                           # All data files
│   ├── raw/                        # Original data
│   └── processed/                  # Processed data
├── src/                            # Source code
│   ├── data/                       # Data handling modules
│   │   ├── data_pipeline.py        # Data processing pipeline
│   │   ├── dataset.py              # Dataset partitioning
│   │   ├── preprocess.py           # Text preprocessing
│   │   └── encode.py               # Text encoding
│   ├── models/                     # Model implementations
│   │   └── model.py                # Sentiment analysis model
│   ├── metrics/                    # Performance metrics
│   │   └── metrics.py              # Model evaluation metrics
│   └── deployment/                 # Deployment code
│       └── deployment.py           # Flask API service
├── tests/                          # Test suite
│   ├── test_model.py               # Tests for model
│   └── test_preprocess.py          # Tests for preprocessing
├── notebooks/                      # Jupyter notebooks
│   ├── EDA.ipynb                   # Exploratory data analysis
│   └── text_data_processing_mod-11.ipynb  # Data processing
├── config/                         # Configuration files
│   └── config.yaml                 # Configuration parameters
├── models/                         # Saved models
│   ├── rf_basic.pkl                # Random Forest model
│   └── model_wv.bin                # Word2Vec model
├── scripts/                        # Utility scripts
│   └── train.py                    # Script to train the model
├── Dockerfile                      # Container configuration
├── docker-compose.yml              # Docker Compose configuration
├── requirements.txt                # Project dependencies
├── System.md                       # System design documentation
└── README.md                       # Project overview and usage instructions
```

## Features

- Text preprocessing including normalization, tokenization, and lemmatization
- Word embedding using Word2Vec
- Sentiment classification using Random Forest algorithm
- Performance evaluation with accuracy, precision, recall, and F1 score
- REST API for prediction serving
- Containerized deployment with Docker
- Comprehensive test suite

## Requirements

- Python 3.11.3 or higher
- Required packages are listed in `requirements.txt`

## Installation

### Local Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd SentimentAnalysis
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download NLTK resources:
   ```bash
   python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
   ```

### Docker Installation

1. Build the Docker image:
   ```bash
   docker build -t sentiment-analysis .
   ```

2. Or use Docker Compose:
   ```bash
   docker-compose up -d
   ```

## Usage

### Training the Model

To train the sentiment analysis model:

```bash
python scripts/train.py --data-file your_data.csv
```

The script will:
1. Load and preprocess the data
2. Train a Word2Vec model for embeddings
3. Train a Random Forest classifier
4. Evaluate the model
5. Save the model and embeddings

### Running the API

Start the API service:

```bash
python src/deployment/deployment.py
```

The API will be available at `http://localhost:8786`.

### Using the API

Send a CSV file with a text column to the `/predict` endpoint:

```bash
curl -X POST -F "text=@your_file.csv" http://localhost:8786/predict
```

### Running Tests

To run the test suite:

```bash
pytest
```

For specific tests:

```bash
pytest tests/test_preprocess.py
pytest tests/test_model.py
```

## Docker Deployment

The system is containerized for easy deployment:

```bash
# Pull the image
docker pull msherekar/705.603spring24:SentimentAnalysis

# Run the container
docker run -it -p 8786:8786 msherekar/705.603spring24:SentimentAnalysis
```

Use POST requests to send CSV files to the container at `http://localhost:8786/predict` with the key name `text`.

## Development

### Adding New Features

1. Implement the feature in the appropriate module
2. Add tests for the feature
3. Update the documentation
4. Submit a pull request

### Running in Debug Mode

Set the environment variable:

```bash
export FLASK_DEBUG=True
python src/deployment/deployment.py
```

## Acknowledgements

This project was developed as part of the Creating AI Enabled Systems course (EN.705.603.81).

