import gradio as gr
from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib


model = joblib.load('C:/Users/USER/Desktop/ChurnPrediction/logistic_regression_model.pkl')
training_columns = joblib.load('C:/Users/USER/Desktop/ChurnPrediction/training_columns.pkl')

def preprocess_input_data(input_data):
    df = pd.DataFrame(input_data, index=[0])
    df = df.replace({'No': 0, 'Yes': 1, 'No internet service': 0})
    df['InternetService'] = df['InternetService'].replace({'Fiber optic': 2, 'DSL': 1})
    df['Contract'] = df['Contract'].replace({'Two year': 24, 'One year': 12, 'Month-to-month': 1})
    df['PaymentMethod'] = df['PaymentMethod'].replace({
        'Electronic check': 1,
        'Mailed check': 2,
        'Bank transfer (automatic)': 3,
        'Credit card (automatic)': 4
    })
    df['MonthlyCharges'] = df['MonthlyCharges'].astype(int)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0).astype(int)

    df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod'])

    scaler = StandardScaler()
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    
    for col in training_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[training_columns]

    return df

def churn_prediction_interface(
    SeniorCitizen, Partner, Dependents, tenure, InternetService,
    OnlineSecurity, TechSupport, Contract, PaperlessBilling,
    PaymentMethod, MonthlyCharges, TotalCharges
):
    input_data = {
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'TechSupport': TechSupport,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }
    input_df = preprocess_input_data(input_data)
    prediction = model.predict(input_df)
    prediction_probability = model.predict_proba(input_df)

    if prediction[0] == 1:
        churn_status = 'Churn'
    else:
        churn_status = 'Not Churn'

    return {
        "Prediction": churn_status,
        "Probability of Not Churning": prediction_probability[0][0],
        "Probability of Churning": prediction_probability[0][1]
    }

inputs = [
    gr.Radio(['No', 'Yes'], label="Senior Citizen"),
    gr.Radio(['No', 'Yes'], label="Partner"),
    gr.Radio(['No', 'Yes'], label="Dependents"),
    gr.Slider(minimum=0, maximum=100, label="Tenure (months)"),
    gr.Radio(['DSL', 'Fiber optic'], label="Internet Service"),
    gr.Radio(['No', 'Yes'], label="Online Security"),
    gr.Radio(['No', 'Yes'], label="Tech Support"),
    gr.Radio(['Two year', 'One year', 'Month-to-month'], label="Contract"),
    gr.Radio(['No', 'Yes'], label="Paperless Billing"),
    gr.Radio(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], label="Payment Method"),
    gr.Slider(minimum=0, maximum=200, label="Monthly Charges"),
    gr.Slider(minimum=0, maximum=10000, label="Total Charges"),
]

iface = gr.Interface(
    fn=churn_prediction_interface,
    inputs=inputs,
    outputs="json",
    title="Churn Prediction Model",
    description="Predict if a customer will churn using Logistic Regression model."
)

if __name__ == "__main__":
    iface.launch()
