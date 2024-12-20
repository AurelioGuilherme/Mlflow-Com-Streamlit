
import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import subprocess
import os

# Load the dataset
data = pd.read_csv('Data/IMDb movies.csv', low_memory=False)

# Preprocessing: Select relevant columns and drop NaN values
data = data[['duration', 'avg_vote', 'votes', 'budget', 'worlwide_gross_income']]
data = data.dropna()
data['budget'] = data['budget'].replace('[^0-9]', '', regex=True).astype(float)
data['worlwide_gross_income'] = data['worlwide_gross_income'].replace('[^0-9]', '', regex=True).astype(float)

# Split data into features and target
X = data[['duration', 'avg_vote', 'votes', 'budget']]
y = data['worlwide_gross_income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Streamlit UI
st.title('IMDb Movies Box Office Prediction with MLflow')

if st.button('Train Model'):
    with mlflow.start_run():
        # Train a Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Log model and parameters
        mlflow.sklearn.log_model(model, 'model')
        mlflow.log_param('n_estimators', 100)

        # Evaluate the model
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mlflow.log_metric('mse', mse)

        st.write('Model trained and logged with MSE:', mse)

# Button to launch MLflow UI
if st.button('Launch MLflow UI'):
    st.write('Launching MLflow UI...')
    subprocess.Popen(['mlflow',"ui", '--host', '0.0.0.0', '--port', '5000'])
    st.write('MLflow UI is running at http://localhost:5000')

