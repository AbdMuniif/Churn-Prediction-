Machine Learning Pipeline Design to Predict Churning in 2 years 
==============================
This project is a churn prediction pipeline that includes a prototype for user testing. 
It leverages three machine learning models—Logistic Regression, Random Forest, and Neural Network. The best-performing model is integrated into the prototype for user testing. The prototype, built with Gradio, provides an interactive interface for testing and improving the model's performance.

Directory Structure
-------------------
- `Dataset` Dataset used in the project.
- `logistic_regression_model.pkl` The best-performing model selected for integration into the prototype.
- `training_columns.pkl` The list of feature columns used for training the model.
- `ChurnPrediction` Python script of data training, model development, and experimentation.
- `app.py` Python script for the Gradio user interface or testing.

Pipeline Overview:
-------------------
Pipeline Overview:
1. Data Loading and Preprocessing:

Data Loading:
The dataset is loaded from a CSV file and specific columns are dropped (Unnamed: 0 and customerID).

Data Cleaning and Transformation:
Categorical values are replaced with numerical values, including the encoding of InternetService, Contract, PaymentMethod, and other categorical features.
Features are selected and numerical values are scaled using StandardScaler.
One-hot encoding is applied to categorical features (InternetService, Contract, PaymentMethod).

Feature and Target Split:
The dataset is split into features (x) and target (y), followed by splitting into training and testing sets.

2. Model Training and Evaluation:

Logistic Regression:

  Model Training:
A Logistic Regression model is trained on the training data (x_train, y_train).

  Evaluation:
Model predictions are evaluated using accuracy, classification report, and ROC curve.
ROC curve and AUC are plotted to visualize the model's performance.

Random Forest:

  Model Training:
A Random Forest Classifier is trained on the training data.

  Evaluation:
Predictions are evaluated using accuracy, classification report, and ROC curve.
ROC curve and AUC are plotted for visualization.

Neural Network:

Model Architecture:
A Sequential Neural Network model is created with Dense and Dropout layers.

  Training:
The model is compiled and trained with early stopping based on validation loss.

  Evaluation:
Predictions are evaluated using classification report and ROC curve.
ROC curve and AUC are plotted to show performance.

3. Churn Count Calculation:

Churn Counting:
The function count_churn calculates the number of churns and non-churns for each model's predictions.

4. Model and Metadata Saving:

Model Saving:
The Logistic Regression model is saved using joblib.

Training Columns Saving:
The list of training columns is saved for future reference.

5. Visualization:

Histograms and ROC Curves:
Histograms of numerical features and ROC curves for each model (Logistic Regression, Random Forest, Neural Network) are plotted to visualize performance metrics.


Installation (Terminal)
------------
pip install pandas numpy matplotlib scikit-learn keras tensorflow joblib gradio

Make sure to install the necessary libraries and modules before running the project.

==============================

The Gradio Prototype
![image](https://github.com/user-attachments/assets/36092ddf-e25d-499f-88b4-f1b2095aea59)

