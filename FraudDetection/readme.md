# FRAUD DETECTION SYSTEM
## CLASS PROJECT FOR CREATING AI ENABLED SYSTEMS (EN.705.603.81)
### An AI system to detect fraudulant transactions

**Problem Statement**: To improve performance of model for credit card fraud detection. 
**Value Proposition**: A better model will prevent fraudulent transactions thereby improving company profits.

# CONTENTS OF THIS REPOSITORY

## Readme

## Deliverable: A Systems Planning & Requirements

## Deliverable: B Jupyter Notebooks

- ** Exploratory Data Analysis**
- ** Model Performance Analysis**


## Python Files

## Deliverable: C Data Engineering Pipeline
- **data_pipeline.py** : An ETL pipeline to transform the data using the knowledge discovered in your Exploratory Data Analysis

## Deliverable: D Data Partitioning Pipeline
- **dataset.py**: A pipeline to partition data for k-fold cross validation 

## Deliverable: E Metrics Pipeline
- **metrics.py**: For calculating various metrics

## Deliverable: F Model Pipeline
- **model.py** : For model training

## Deliverable: G A wrapper for accepting new data
- **fraud_detection_service.py**: 

## Pickle Files
- Three pickle files for each of three models - random forest, logistic regression and an ensemble model.

## Text Files
-Files have model metrics that were generated while training

## Miscellaneous Files
- Various .csv files for testing
- original data file (transactions.csv)

## Docker Container: To run the model as a service 
- [Docker Image] (https://hub.docker.com/layers/msherekar/705.603spring24/FraudDetection/images/sha256-4688d93a2e62ccba44598334fff72f448778a1b2dd657ff563a400eaf61a7b25?context=repo)
- Download the container using this command on command line *docker pull msherekar/705.603spring24:FraudDetection*
- Run the container on command line using this command line *docker run -it -p 8786:8786 msherekar/705.603spring24:FraudDetection*

