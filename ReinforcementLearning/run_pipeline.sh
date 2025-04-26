#!/bin/bash

# Run Pipeline Script for Reinforcement Learning Email Marketing System
# This script runs the entire pipeline including training, testing, and visualization

echo "Starting Reinforcement Learning Pipeline"
echo "----------------------------------------"

# Define paths
USERBASE_PATH="data/raw/userbase.csv"
SENT_PATH="data/raw/sent_emails.csv"
RESPONDED_PATH="data/raw/responded.csv"
CONFIG_PATH="config/config.yaml"
MODEL_PATH="models/q_table.pkl"
TEST_INPUT="data/raw/test_input.csv"

# Activate conda environment if needed (uncomment and modify as needed)
# source /Users/mukulsherekar/yes/etc/profile.d/conda.sh
# conda activate craisys

# Create necessary directories
mkdir -p models data/processed logs

echo "Step 1: Training the Q-Learning Model"
echo "------------------------------------"
python scripts/train.py \
    --config ${CONFIG_PATH} \
    --userbase_file ${USERBASE_PATH} \
    --sent_file ${SENT_PATH} \
    --responded_file ${RESPONDED_PATH} \
    --episodes 1000

# Check if training was successful
if [ ! -f "$MODEL_PATH" ]; then
    echo "Error: Training failed, model not created"
    exit 1
fi

echo "Step 2: Starting API Service"
echo "---------------------------"
# Start the API service in the background
python scripts/deploy.py --config ${CONFIG_PATH} --debug &
API_PID=$!

# Wait for API to start
echo "Waiting for API service to start..."
sleep 5

echo "Step 3: Testing the API with sample data"
echo "---------------------------------------"
# Test the API with sample data
curl -X POST -F "new_state=@${TEST_INPUT}" http://localhost:5000/suggest_subject_lines

# Check the health endpoint
echo -e "\nChecking API health:"
curl http://localhost:5000/health

# Terminate the API service
echo -e "\nTerminating API service"
kill $API_PID

echo -e "\nPipeline execution complete!" 