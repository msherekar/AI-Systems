# **Motivation (WHY's)**

## **Why are we solving this problem?**

**Problem Statement**: Forecast total and fraudulent transactions  upto a month based on historical data.

 **Value Proposition**: Forecasting will shed light on business growth and impact of various factors on increasing sales and decreasing theft.

 **Why is our solution a viable one**?

We are proposing a machine learning based solution. It is a viable solution because of its accurate prediction capability, fully automated nature and ability to course correct if anything happens. Also, we have access to tons of data which will enable construction of a robust model. We can tolerate some degree of risk in the sense it will be after all a forecast. Ideally, we aim to build a model that predicts within1-2% errors. Forecasting is just one helpful tool for taking future business decisions.This solution is a feasible one because we have the data powerful model and compute power. 

# **Requirements (WHAT's)**

## **Scope**

### What are our goals?
- **Organizational Goals:** To improve forecasting
- **System Goals:** Develop better forecasting models
- **User Goals:**
- **Customer:** Prevent loss of money, time and related stress
- **Company:** Prevent resources being deployed and wasted to recover lost money and provide a smooth experience to customers
- **Model Goals:** Bring down the prediction	to less than 5%

### What are the success criteria?
Our success criteria is to increase the accuracy to 95%

## **Requirements**

### **What are our(system) Assumptions?**
We assume we have the compute power (hardware), cyber security set up (hardware and software), data storage capabilities, availability of state of the art ML softwares, visualization products, project management solutions and talented hardworking man power.

### **What are our (system) Requirements?**
First and foremost - clean data ready to be fed into models. Secondly, ability to create and analyze new models. Thirdly, ability to quickly act on detected fraudulent transactions. Finally, ability to change or update model with new incoming data.

# **Implementation (HOW's)**

### **Methodology:** 

- **Feature Engineering**: Data was grouped by months for traditional SARIMAX model and by days for LSTM model. The total number of transactions per month and total number of fraudulent transactions per day were determined to be forecasted.

- **Dataset Partitioning:** Models were trained on first 48 months for SARIMAX and 1795 days for LSTM.

- **Metrics and Advanced Metrics:** Mean Squared Error and Root Mean Squared Errors were used.

- **Model Selection:** SARIMAX and LSTM

- ** Deployment Strategy:**The deployment strategy for the forecasting system involves packaging the LSTM model into a Dockerized system architecture. The system will utilize a class named Fraud_Forecasting_Model defined in the model.py module to construct and handle the model logic. The system will be capable of receiving input data via requests sent through Postman and GET, which should be in the format of a .csv file. The final system will be packaged into a Docker image and published on DockerHub for ease of deployment.


