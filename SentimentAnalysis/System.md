# **Motivation (WHY's)**

## **Why are we solving this problem?**

**Problem Statement**: To have a better rating system based on reviews written by reviewers

 **Value Proposition**: Capturing sentiments derived ratings will help in a more nuanced and reliable customer feedback and product quality.

 **Why is our solution a viable one**?

We are proposing a NLP based solution. Ideally, such a solution will fix the unreliable star based rating. NLP based solution is viable because of large number of reviews and powerful algorithms that can analyze such a large amount of text.

# **Requirements (WHAT's)**

## **Scope**

### What are our goals?
- **Organizational Goals:** To improve rating system to capture customer feedback and product quality
- **System Goals:** Develop a NLP. based sentiment analysis model

- **User Goals:**
- **Customer:** Provide an accurate and reliable system to choose movies
- **Company:** Obtain accurate feedback. based on sentiment analysis to improve product quality
- **Model Goals:** Predict ratings based on sentiments with 95% accuracy

### What are the success criteria?
Our success criteria is to increase the accuracy to 95% for each rating category

## **Requirements**

### **What are our(system) Assumptions?**
We assume we have the compute power (hardware), cyber security set up (hardware and software), data storage capabilities, availability of state of the art ML softwares, visualization products, project management solutions and talented hardworking man power.

### **What are our (system) Requirements?**
First and foremost - clean data ready to be fed into models. Secondly, ability to create and analyze new models. Thirdly, ability to change or update model with new incoming data.

# **Implementation (HOW's)**

### **Methodology:** 

- **Feature Engineering**: Selecting text and ratings, normalizing the text, removing rows where there is no normalized text, tokenizing the normalized text, creating embeddings based on tokenized text, upsampling/downsampling based on ratings.

- **Dataset Partitioning:**: A simple 80:20 split for training and testing.

- **Metrics and Advanced Metrics:** Simple metrics like precision and recall and overall accuracy.

- **Model Selection:** A traditional random forest model and a deep learning based GRU model.

- ** Deployment Strategy:**The deployment strategy for the forecasting system involves packaging the model into a Dockerized system architecture. The system will utilize a class named Ratings_Prediction_Model defined in the model.py module to construct and handle the model logic. The system will be capable of receiving input data via requests sent through Postman  which should be in the format of a .csv. The final system will be packaged into a Docker image and published on DockerHub for ease of deployment.


