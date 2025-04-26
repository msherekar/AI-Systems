# TimeSeries Project

This project aims to forecast overall transactions and fraudulent transactions using time series models.

## Overview

The TimeSeries Forecasting project is designed to predict total and fraudulent transactions based on historical data. It employs both traditional statistical models (SARIMAX) and deep learning models (LSTM) for comparison.

## Directory Structure

- `data/`: Contains raw and processed data files.
  - `raw/`: Raw data files (e.g., credit card transactions).
  - `processed/`: Processed data files ready for model training.
- `src/`: Source code for data handling, models, training, and API.
  - `data/`: Data preprocessing modules.
  - `models/`: Model implementations (SARIMAX and LSTM).
  - `training/`: Training orchestration modules.
  - `api/`: API service implementations.
  - `utils/`: Utility functions and configuration management.
- `tests/`: Test suite for the project.
  - `test_preprocessor.py`: Tests for data preprocessing.
  - `test_sarimax.py`: Tests for SARIMAX models.
  - `test_lstm.py`: Tests for LSTM models.
- `notebooks/`: Jupyter notebooks for EDA and model development.
- `config/`: Configuration files.
  - `config.yaml`: Main configuration parameters.
- `models/`: Saved trained models.
- `docs/`: Documentation.
  - `system_design.md`: System design documentation.
- `scripts/`: Utility scripts.
  - `generate_synthetic_data.py`: Generate synthetic data for testing.
  - `run_pipeline.py`: Run the entire forecasting pipeline.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Docker (for containerized deployment)

### Installation

1. Clone the repository:
   ```bash
   git clone [repository_url]
   cd TimeSeries
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Pipeline

1. Generate synthetic data (optional):
   ```bash
   python scripts/generate_synthetic_data.py
   ```

2. Run the entire pipeline:
   ```bash
   python scripts/run_pipeline.py
   ```

3. To run the pipeline without generating new data:
   ```bash
   python scripts/run_pipeline.py --no-generate-data
   ```

### Running the API

1. Start the API locally:
   ```bash
   uvicorn src.api.service:app --reload
   ```

2. Access the API documentation:
   Open `http://localhost:8000/docs` in your web browser.

### Using Docker

1. Build the Docker image:
   ```bash
   docker build -t timeseries-forecasting .
   ```

2. Run the container:
   ```bash
   docker run -p 80:80 timeseries-forecasting
   ```

3. Alternatively, use Docker Compose:
   ```bash
   docker-compose up
   ```

### Testing

Run the test suite:
```bash
pytest
```

## Model Details

### SARIMAX Models
- **Total Transactions**: SARIMAX model for forecasting overall transaction volume.
- **Fraudulent Transactions**: SARIMAX model for forecasting fraudulent transaction volume.

### LSTM Models
- **Total Transactions**: LSTM model for forecasting overall transaction volume.
- **Fraudulent Transactions**: LSTM model for forecasting fraudulent transaction volume.

## API Endpoints

- `GET /`: Root endpoint, returns a welcome message.
- `GET /forecast/sarimax/total`: Forecasts total transactions using the SARIMAX model.
- `GET /forecast/sarimax/fraud`: Forecasts fraudulent transactions using the SARIMAX model.
- `GET /forecast/lstm/total`: Forecasts total transactions using the LSTM model.
- `GET /forecast/lstm/fraud`: Forecasts fraudulent transactions using the LSTM model.
- `POST /forecast/upload`: Accepts a CSV file upload and returns forecasts based on the data.

# FORECASTING WITH TIME SERIES DATA
## CLASS PROJECT FOR CREATING AI ENABLED SYSTEMS (EN.705.603.81)
### An AI system to predict total & fraudulent transactions

**Problem Statement**: To forecast overall transactions and fraudulent transactions. 
**Value Proposition**: To provide insights into business growth, theft & impact of special activities

# CONTENTS OF THIS REPOSITORY

## Readme 

## Deliverable: A Systems Planning & Requirements

## Deliverable: B Jupyter Notebooks

- ** TimeSeriesDataPrep.ipynb **: Notebook that explores pandas time methods
- ** Sarimax notebooks **: Notebooks to explore tranditional time series models
- ** LSTM notebooks **: Notebook to explore neural network time series models.


