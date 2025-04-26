# *********** This code is for deploying the model and making predictions ***********
# This code will accept new text review(as .csv file) via post return the sentiment of the review.

import os
import sys
import logging
from pathlib import Path
from flask import Flask, request, jsonify
import pandas as pd
import tempfile

# Add project root to path for relative imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data.data_pipeline import Pipeline
from src.models.model import Model

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load model once at startup
model = None
try:
    model = Model(model_path=os.path.join(project_root, 'models/rf_basic.pkl'))
    logger.info("Model loaded successfully at startup")
except Exception as e:
    logger.error(f"Error loading model at startup: {str(e)}")

@app.route('/', methods=['GET'])
def home():
    """
    Home endpoint that provides basic info about the API.
    """
    return jsonify({
        "status": "success",
        "message": "Sentiment Analysis API is running",
        "endpoints": {
            "/predict": "POST - Send a CSV file with text for sentiment analysis",
            "/healthcheck": "GET - Check if the API is healthy"
        }
    })

@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    """
    Health check endpoint to ensure the API is running.
    """
    return jsonify({"status": "success", "message": "API is healthy"})

@app.route('/predict', methods=['POST'])
def predict_post():
    """
    Prediction endpoint that accepts a CSV file with text and returns sentiment predictions.
    """
    try:
        if 'text' not in request.files:
            return jsonify({"status": "error", "message": "No file part named 'text' in the request"}), 400

        # Get the text review from the request
        file = request.files['text']
        
        # Check if the file is empty
        if file.filename == '':
            return jsonify({"status": "error", "message": "No file selected"}), 400
        
        # Check if we have a model loaded
        if model is None:
            return jsonify({"status": "error", "message": "Model not loaded"}), 500

        # Save the file to a temporary location
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        file.save(temp_file.name)
        temp_file.close()
        
        logger.info(f"Processing file: {file.filename}")
        
        # Process the data
        try:
            pipeline = Pipeline(temp_file.name)
            processed_data = pipeline.get_processed_data()
            
            # Make predictions
            predictions = model.predict_sentiment(processed_data)
            
            # Add predictions to the original data
            original_data = pd.read_csv(temp_file.name)
            predictions_df = pd.DataFrame(predictions, columns=['sentiment'])
            result_df = pd.concat([original_data, predictions_df], axis=1)
            
            # Convert result to JSON
            result = result_df.to_dict(orient='records')
            
            # Clean up the temporary file
            os.unlink(temp_file.name)
            
            logger.info(f"Successfully processed file: {file.filename}")
            return jsonify({
                "status": "success",
                "predictions": result,
                "prediction_count": len(predictions)
            })
        except Exception as e:
            # Clean up the temporary file
            os.unlink(temp_file.name)
            logger.error(f"Error processing file: {str(e)}")
            return jsonify({"status": "error", "message": str(e)}), 500
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    flask_port = int(os.environ.get("FLASK_PORT", 8786))
    flask_host = os.environ.get("FLASK_HOST", "0.0.0.0")
    debug_mode = os.environ.get("FLASK_DEBUG", "True").lower() == "true"
    
    logger.info(f"Starting server on {flask_host}:{flask_port}, debug mode: {debug_mode}")
    app.run(host=flask_host, port=flask_port, debug=debug_mode)
