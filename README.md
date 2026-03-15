# Customer Churn Prediction using ANN

## Project Overview

This project implements an **Artificial Neural Network (ANN)** to predict **customer churn**. By analyzing customer data, the model identifies customers at risk of leaving, helping businesses take proactive retention measures.

The model is **deployed using Streamlit**, providing a user-friendly interface for interactive predictions. Users can input customer details and get real-time churn predictions.

## Dataset

* **Source:** Churn dataset (Kaggle dataset)
* **Features:** Customer demographics, account information, usage patterns, etc.
* **Target:** `Churn` (Yes/No)

## Technologies & Libraries

* **Python** – Programming language
* **TensorFlow / Keras** – Building and training the ANN
* **scikit-learn** – Data preprocessing, scaling, and evaluation
* **Pandas & NumPy** – Data handling
* **Matplotlib & Seaborn** – Visualization
* **Streamlit** – Deployment of the interactive web app

## Data Preprocessing

* Handled categorical variables using **Label Encoding / One-Hot Encoding**
* Scaled numerical features using **StandardScaler**
* Split the dataset into **training and testing sets**

## Model Architecture

* Input Layer: Number of neurons = number of features
* Hidden Layers: 2–3 layers with **ReLU activation**
* Output Layer: 1 neuron with **Sigmoid activation** (binary classification)
* Optimizer: **Adam**
* Loss Function: **Binary Crossentropy**
* Metrics: **Accuracy, Precision, Recall, F1-score**

## Training

* Early stopping to prevent overfitting
* Model evaluation on the test set
* Confusion matrix and accuracy metrics visualized

## Deployment with Streamlit

* **Interactive Web App:** Users can enter customer details and receive churn predictions instantly
* **Real-Time Predictions:** Model outputs probability and classification (Yes/No)
* **Visual Feedback:** Prediction results are displayed clearly in the app

  

### How to Run the App


1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```
3. Open the provided local URL in a browser to interact with the deployed model

## Results

* Model Accuracy: **XX%** (replace with your actual results)
* Confusion matrix, precision, recall, and F1-score demonstrate model performance
* The deployed app enables **instant and practical predictions** for new customer data

## Future Improvements

* Hyperparameter tuning to improve accuracy
* Experiment with more advanced ANN architectures
* Add visualizations for overall churn trends and customer insights.
