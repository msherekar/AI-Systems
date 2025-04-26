# **Motivation (WHY's)**

## **Why are we solving this problem?**

**Problem Statement**: Performance of model for fraud detection has deteriorated (Precision = 70% & Recall = 40%). Hence, we are solving this problem to improve model performance by improving both the metrics

 **Value Proposition**: A better model will prevent fraudulent transactions thereby improving company profits.

 **Why is our solution a viable one**?

 We are proposing a machine learning based solution. It is a viable solution because of its accurate prediction capability, fully automated nature and ability to course correct if anything happens. Also, we have access to tons of data which will enable construction of a robust model.

We can tolerate some degree of risk in the sense that blocking non-fradulent transactions at the expense of preventing loss of money for customers. This can be taken care of temporary blocking. This will help us prevent spending time and resources to get the money back for customers, track down fraudsters and litigate the case(s). On the same note, we should be able to tolerate mistakes because we will have other avenues to get the money back. These avenues include litigation and immediate refunds to the customer.

This solution is a feasible one because we have the data and compute power. 

# **Requirements (WHAT's)**

## **Scope**

### What are our goals?
- **Organizational Goals:** Increase profits and user experience (expand)
- **System Goals:** Catch fraudulent transactions from happening by predicting them beforehand
- **User Goals:**
- **Customer:** Prevent loss of money, time and related stress
- **Company:** Prevent resources being deployed and wasted to recover lost money and provide a smooth experience to customers
- **Model Goals:** Improve Precision and Recall compared to current model and ideally take it to more than 95%

### What are the success criteria?
Our success criteria is to improve the Precision=99% and Recall = 99% (change later)

## **Requirements**

### **What are our(system) Assumptions?**
We assume we have the compute power (hardware), cyber security set up (hardware and software), data storage capabilities, availability of state of the art ML softwares, visualization products, project management solutions and talented hardworking man power.

### **What are our (system) Requirements?**
First and foremost - clean data ready to be fed into models. Secondly, ability to create and analyze new models. Thirdly, ability to quickly act on detected fraudulent transactions. Finally, ability to change or update model with new incoming data.

#### **identify Stakeholders**: Regulatory Bodies, Financial Institutions, Customers, AI systems designers, Legal & Compliance Teams

#### **Collect Information (data)**
- **Transaction Data:** Collect historical transaction data including timestamps, transaction amounts, locations, and transaction types.
- **Customer Information:** Gather customer profiles, including demographics, account activity, and transaction behavior.
- **Fraudulent Activity Data:** Acquire data on known fraudulent transactions or patterns obtained from past incidents or external sources.

#### **Negotiate Conflicts** 
- **Risk Tolerance vs. False Positives:** Balancing the need to detect fraud with the risk of flagging legitimate transactions as false positives.
- **Resource Allocation:** Address conflicts regarding budget allocation and resource prioritization for AI model development and deployment.
- **Interpretation of Results:** Resolve disagreements on the interpretation of AI model outputs and the appropriate action to take in response to flagged transactions.
- **Ethical Considerations:** Navigate ethical dilemmas related to the use of AI in fraud detection, such as bias in algorithms or unintended consequences for certain demographic groups.

#### **Document Requirements**
- **Functional Requirements:** Define the features and capabilities of the fraud detection AI model, including real-time monitoring, anomaly detection, ####and predictive analytics.
- **Non-functional Requirements:** Specify performance criteria such as detection accuracy, latency, scalability, and reliability.
- **Compliance Requirements:** Document regulatory requirements for fraud detection systems, ensuring adherence to anti-money laundering (AML) ####and Know Your Customer (KYC) regulations.
- **Data Handling Policies:** Establish policies for data acquisition, storage, and processing, outlining procedures for data encryption, access controls, ####and audit trails.
- **Model Validation Procedures:** Document methodologies for validating the AI model, including testing procedures, validation metrics, and ####acceptance criteria.

#### **Evaluate**
- **Detection Accuracy:** Measure the AI model's effectiveness in accurately identifying fraudulent transactions compared to legitimate ones.
- **False Positive Rate:** Evaluate the frequency of false positives and their impact on customer experience and operational efficiency.
- **Model Performance Over Time:** Monitor the AI model's performance and adaptability to evolving fraud patterns and attack techniques.
- **Feedback from Stakeholders:** Gather feedback from financial institutions, regulatory bodies, and customers to assess the AI model's ####effectiveness and identify areas for improvement.
- **Continuous Monitoring and Improvement:** Implement mechanisms for ongoing evaluation and refinement of the fraud detection AI model based on performance metrics and feedback.

#### **Iterate**
- **Feedback Loop Integration:** Incorporate feedback from stakeholders and evaluation results into the model development process to enhance detection capabilities and reduce false positives.
- **Feature Engineering:** Iterate on feature selection and engineering techniques to improve the AI model's ability to detect subtle patterns indicative of fraudulent activity.
- **Algorithm Optimization:** Explore and experiment with different machine learning algorithms and techniques to optimize the fraud detection model's performance.
- **Adaptive Learning:** Implement mechanisms for the AI model to learn from new data and adapt its detection strategies to emerging fraud trends.
- **Regular Updates and Maintenance:** Schedule regular updates and maintenance activities to address software vulnerabilities, data drift, and changing regulatory requirements.

## **Risk & Uncertainties**

### **What are possible harms?**
Possible harms for a ML based fraud detection system include - faulty detections (change words), inconvenience to the customers if their legitimate transactions get blocked, super inconvenience if a time-dependent transaction is blocked. Then, secondly missing on fraud detection all together causing monetary losses to the company as well as customer and downstream hassle. Overall, this can add an overhead expenditure to recover and rectify transactions missed by the ML systems.

### **What are causes of mistakes?**
Mistakes can arise from stale data or incorrect data. Also, faulty pre-processing can lead to mistakes. Absence of variety in data will not able the model to generalize or learn hollistically. 

# **Implementation (HOW's)**

## **Development**

### **Methodology:** 

- **ETL Pipeline:** Develop an Extract, Transform, Load (ETL) pipeline to preprocess and clean the data, including steps such as data normalization, feature engineering, and outlier detection to improve model performance. 

- **Feature Engineering**: Four new features ('is_night', 'is_holidays', 'is_post_holidays', 'is_summer') were engineered based on exploratory data analysis. 
1. 'is_holidays', 'is_post_holidays', 'is_summer' are categorical features (one-hot-encoded). It was designed this way because data showed most of the fraudulent transactions belonged to summer, holidays and post-holiday period
2. 'is_night' is a binary feature having 0 for transactions happening between 4am to 10pm and 1 for transactions between 10pm-4am. It was designed this way because most of the fraudulent transactions happened in the night between 10pm-4am
3. Overall, my data transformation includes grouping transactions by unique victims and adding new features.

- **Dataset Partitioning:** Split the dataset into training, validation, and test sets to train and evaluate the model's performance. To ensure representative samples for each class, considering the imbalanced nature of fraud detection datasets, following steps were taken-
1. Stratified sampling 
2. 5-fold cross validation
3. Oversampling of minority class so as to have 1:1 ratio prior to splitting
4. While splitting, used group fold to ensure transactions of each customer stayed in one of test, validation or training set

- **Metrics and Advanced Metrics:** Define evaluation metrics such as precision, recall, F1-score to assess model performance. 
1. Recall: Prioritize recall because the cost of false negatives (failing to detect actual fraudulent transactions) is high. Missing fraudulent activities can lead to financial losses and legal consequences.
2. Precision: Precision can be compromised because the cost of false positives (misclassifying a non-fraudulent transaction as fraudulent) is not as high as failing to detect a fraudulent transaction. Blocking legitimate transactions as fraud might result in customer dissatisfaction but it might be as bad as compared to missing a actual fraudulent transaction.
3. F1 score: Use this metric to compare different modeling strategies.

- **Model Selection:** 
1. I trained and tested three models - logistic regression, random forest and ensenble model. 
2. Tree based random forest and ensemble model prioritize recall scores hence should be the models of choice. 
3. Also, their recall scores (0.68 and 0.75) are much higher than logistic regression (0.03 !)
4. But for this submission, I chose logistic regression for the containerization.
5. I was having troubles with dependencies related to tree based models. These models were working fine in jupyter notebook.


1. Based on this decision to prioritize Recall over Precision, Random Forest, Gradient Boosting Machines will be preferred models of choice. Also, ensemble models using Stacking Classifier (Linear Discriminant Analysis & Decision Tree) can also be used.
2. Because of dependencies issues inside docker, I am submitting docker container with logistic regression model.
 
- **Deployment Strategy:**

The deployment strategy for the fraud detection system involves packaging the machine learning model into a Dockerized system architecture. The system will utilize a class named Fraud_Detector_Model defined in the model.py module to construct and handle the model logic. The system will be capable of receiving input data via requests sent through Postman and GET, which should be in the format of a .csv file. Additionally, an ETL pipeline will be utilized to transform the input data into usable features. The final system will be packaged into a Docker image and published on DockerHub for ease of deployment.

### **High-level System Design**
- **Scalability:** Design the system to handle large volumes of real-time transaction data efficiently, ensuring scalability to accommodate increasing transaction volumes over time.
- **Real-time Processing:** Implement real-time processing capabilities to analyze transactions as they occur, enabling immediate detection and response to fraudulent activities.
- **Integration with Existing Systems:** Integrate the fraud detection model with the credit card company's existing IT infrastructure and systems, ensuring seamless interoperability and data exchange.

### **Development Workflow**
- **Agile Development:** Adopt an agile development methodology to facilitate iterative development, allowing for frequent feedback cycles and rapid iteration based on stakeholder inputs and evolving requirements.
- **Version Control:** Utilize version control systems such as Git to manage codebase changes and facilitate collaboration among development team members.
- **Continuous Integration and Testing:** Implement continuous integration and automated testing processes to ensure code quality, reliability, and consistency across different development stages.

## **Policy**

### **Human-Machine Interfaces**
- **User Interface Design:** Design intuitive user interfaces for analysts and investigators to interact with the fraud detection system, providing them with actionable insights and decision support tools to efficiently investigate flagged transactions.
- **Training and Education:** Provide training programs and educational resources to equip employees with the necessary skills and knowledge to effectively utilize the AI-powered fraud detection system.

### **Regulations:**
- **Compliance:** Ensure that the fraud detection system complies with relevant regulations and industry standards governing data privacy, security, and financial transactions, such as GDPR, PCI DSS, and regulations set forth by financial regulatory bodies.

## **Operations**

### **Continuous Deployment**
- **Automated Deployment Pipelines:** Implement automated deployment pipelines to streamline the deployment process, enabling rapid and consistent deployment of model updates and enhancements.
- **Rollback Mechanisms:** Establish rollback mechanisms to revert to previous model versions in case of deployment failures or performance issues, minimizing downtime and service disruptions.

### **Post-deployment Monitoring**
- **Performance Monitoring:** Continuously monitor the performance of the deployed fraud detection model in production, tracking key metrics such as detection accuracy, false positive rates, and response times.
- **Anomaly Detection:** Implement anomaly detection techniques to identify deviations from expected model behavior or performance indicators, enabling proactive intervention to address potential issues or anomalies.

### **Maintenance**
- **Model Retraining:** Schedule periodic model retraining cycles using updated data to ensure that the fraud detection model remains effective and adaptive to evolving fraud patterns and trends.
- **System Updates and Patches:** Apply regular updates and patches to the fraud detection system to address security vulnerabilities, software bugs, and compatibility issues, minimizing risks of system downtime or breaches.

### **Quality Assurance**
- **Testing and Validation:** Conduct comprehensive testing and validation procedures across different stages of the development lifecycle, including unit testing, integration testing, and end-to-end testing, to verify the correctness, robustness, and reliability of the fraud detection system.
- **Auditing and Compliance Checks:** Perform regular audits and compliance checks to ensure that the fraud detection system adheres to established quality standards, regulatory requirements, and internal policies.

