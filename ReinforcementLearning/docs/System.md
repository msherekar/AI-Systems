# **Motivation (WHY's)**

## **Why are we solving this problem?**

**Problem Statement**: To improve response rates to campaign marketing emails. 

 **Value Proposition**: Changing subject lines dynamically might result in better response rates.

 **Why is our solution a viable one**?

We are proposing a reinforcement learning based solution. Idea is to capture dynamic nature of responses to emails. Then, change the subject line based on responses.

# **Requirements (WHAT's)**

## **Scope**

### What are our goals?
- **Organizational Goals:** To improve response rates to emails in order to derive actions from potenial customers
- **System Goals:** Develop a Reinforcement Learning based system to change email subject lines to improve response rates

- **User Goals:**
- **Customer:** Provide a meaningful email subject line to respond.
- **Company:** Obtain better response rates from customers
- **Model Goals:** Build a Q Table based upon the conversion rate given the state and actions.

### What are the success criteria?
Our success criteria is to double the response rate/success rate

## **Requirements**

### **What are our(system) Assumptions?**
We assume we have the compute power (hardware), cyber security set up (hardware and software), data storage capabilities, availability of state of the art ML softwares, visualization products, project management solutions and talented hardworking man power.

### **What are our (system) Requirements?**
First and foremost - clean data ready to be fed into models. Secondly, ability to create and analyze new models. Thirdly, ability to change or update model with new incoming data.

# **Implementation (HOW's)**

### **Methodology:** 

- ** Design Implementation:** The over all design is to have a container to which user can send new data  in form of a .csv file via postman. The container will return new subject lines for each state. Within the container, interface is the deployment.py file. This file will accept the .csv file from the user via post. Then, preprocessing will take place and finally the saved reinforcement model to predict the subject lines for each state. The entire code for model training is modularized into different files. The preprocess.py file is used to load the data, merge the data, calculate rewards and preprocess the data. The sar.py file is used to generate states, generate actions, calculate rewards and get rewards. The qagent.py file is used to initialize the Q-learning agent and train the agent. The metrics.py file is used to calculate the metrics. The train.py file is used to train the agent. The report.py file is used to save the report.

- ** Data Cleansing:** Idea is to merge the datasets to set up rewards. Rewards feature will have two unique values 1 and 0. 1 will be assigned to the state where the user has responded to the email on the same data and 0 will be assigned to the state where the user has not responded to the email onm same day. The states will be generated based on the unique values of the features. Features selected were Age, Gender, Tenure and Type. These features were selected because they are the most important features that can be used to describe the user and predict the response of the user. The features discarded were emailid, customerid, dates because they were not deemed important. Also, their inclusion would have increased the state size and made the model more complex.The actions will be generated based on SubLine_Sent (subject line sent) values from the given data.

- ** Model Selection:** A Q learning model was used to predict the subject lines for each state. The Q learning model was selected because it is a model-free reinforcement learning algorithm to learn the value of an action in a particular state

- ** Model Configuration:** The Q learning model was configured with a learning rate of 0.1, discount factor of 0.9 and epsilon of 0.1. The learning rate is the rate at which the agent learns from the rewards. The discount factor is the rate at which the agent discounts future rewards. The epsilon is the rate at which the agent explores the environment.

- ** Model Optimization:** The model was optimized by training the agent on the states and rewards. The agent was trained for 1000 episodes. The training rewards were calculated for each episode. The discounted rewards were calculated for each episode. The cumulative rewards were calculated for each episode. The average reward was calculated for each episode. The episodes to convergence were calculated for each episode. The Q table was saved for each episode.

- ** Model Hyperparameters:** The hyperparameters of the model were the learning rate, discount factor and epsilon. The learning rate was set to 0.1. The discount factor was set to 0.9. The epsilon was set to 0.1. The learning rate is the rate at which the agent learns from the rewards. The discount factor is the rate at which the agent discounts future rewards. The epsilon is the rate at which the agent explores the environment.

- ** Model Metrics:** The metrics of the model were the cumulative rewards, average reward, discounted reward and episodes to convergence. The cumulative rewards were the rewards obtained by the agent for each episode. The average reward was the average reward obtained by the agent for each episode. The discounted reward was the reward obtained by the agent for each episode. The episodes to convergence were the number of episodes taken by the agent to converge.

- ** Deployment Strategy:** The deployment strategy for the Email Marketing System involves packaging the model into a Dockerized system architecture. The system will utilize a class named QLearningAgent defined in the qagent.py module to construct and train the agent . The system will be capable of receiving input data  via requests sent through Postman  which should be in the format of a .csv. The output can be seen on the Postman. The final system will be packaged into a Docker image and published on DockerHub for ease of deployment.


