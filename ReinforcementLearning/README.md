# Email Marketing Reinforcement Learning System

A machine learning system that uses reinforcement learning to optimize email marketing campaigns by dynamically recommending subject lines based on customer attributes and response history.

## Project Overview

This system leverages Q-learning to learn optimal email subject line selection strategies based on customer demographics and historical engagement data. By treating the email marketing process as a reinforcement learning problem, the system:

1. Represents customer attributes as states
2. Treats subject line selections as actions
3. Uses customer responses as rewards
4. Continuously improves recommendations through experience

## Directory Structure

```
ReinforcementLearning/
├── config/            # Configuration files
├── data/              # Data directory
│   ├── raw/           # Raw data files
│   └── processed/     # Processed data files
├── models/            # Trained model files
├── scripts/           # Executable scripts
│   ├── train.py       # Training script
│   └── deploy.py      # Deployment script
└── src/               # Source code
    ├── preprocessing/ # Data preprocessing modules
    ├── agents/        # RL agent implementations
    ├── api/           # API service code
    └── utils/         # Utility functions
```

## Data Description

The system uses the following data files:

### Sample Data Files

1. `userbase_sample.csv` - Customer demographic data:
   - `Customer_ID` - Unique identifier for each customer
   - `Gender` - Customer gender
   - `Type` - Customer type (Premium, Basic)
   - `Age` - Customer age in years
   - `Tenure` - Customer tenure in months

2. `sent_emails_sample.csv` - Record of sent emails:
   - `Customer_ID` - Customer identifier
   - `Sent_Date` - Date when the email was sent
   - `SubjectLine_ID` - Identifier for the subject line used

3. `responded_emails_sample.csv` - Record of email responses:
   - `Customer_ID` - Customer identifier
   - `Sent_Date` - Date when the email was sent
   - `SubjectLine_ID` - Subject line that received a response
   - `Response_Date` - Date when the customer responded
   - `Response_Type` - Type of response (positive, neutral, negative)

4. `click_data_sample.csv` - Record of link clicks in emails:
   - `Customer_ID` - Customer identifier
   - `Sent_Date` - Date when the email was sent
   - `SubjectLine_ID` - Subject line of the clicked email
   - `Click_Date` - Date when the customer clicked
   - `Link_ID` - Identifier for the clicked link

5. `test_input.csv` - Sample input for the API service:
   - `Gender` - Customer gender
   - `Type` - Customer type
   - `Age` - Customer age
   - `Tenure` - Customer tenure

## Installation

### Prerequisites

- Python 3.11+
- Docker (optional for containerized deployment)

### Setting up the environment

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd ReinforcementLearning
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the model

To train the reinforcement learning model:

```bash
python scripts/train.py
```

You can customize the training with configuration options:

```bash
python scripts/train.py --userbase_file data/raw/userbase_sample.csv --sent_file data/raw/sent_emails_sample.csv --responded_file data/raw/responded_emails_sample.csv
```

### Deploying the API service

To deploy the API service:

```bash
python scripts/deploy.py
```

By default, the service runs on port 5000.

### Using Docker

Build and run the Docker container:

```bash
docker build -t email-rl-system .
docker run -p 5000:5000 email-rl-system
```

### API Endpoints

- `POST /suggest_subject_lines` - Get subject line recommendations for new customers
  ```bash
  curl -X POST -F "new_state=@data/raw/test_input.csv" http://localhost:5000/suggest_subject_lines
  ```

- `GET /health` - Health check endpoint
  ```bash
  curl http://localhost:5000/health
  ```

## Testing

Run the test suite:

```bash
pytest
```

## License

[Open License]

## Contributors

[Mukul Sherekar]