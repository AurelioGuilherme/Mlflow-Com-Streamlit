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
from mlflow.tracking import MlflowClient
import time
import requests
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

def check_mlflow_server(port=5000):
    """Verifica se o servidor MLflow está rodando"""
    try:
        response = requests.get(f"http://localhost:{port}")
        return True
    except:
        return False

def start_mlflow_server():
    """Inicia o servidor MLflow se não estiver rodando"""
    if not check_mlflow_server():
        try:
            subprocess.Popen(['mlflow', 'ui', '--host', '0.0.0.0', '--port', '5000'])
            # Aguarda o servidor iniciar
            time.sleep(5)
            return True
        except Exception as e:
            st.error(f"Erro ao iniciar MLflow server: {str(e)}")
            return False
    return True

def load_and_preprocess_data():
    """Carrega e pré-processa os dados"""
    try:
        data = pd.read_csv('Data/IMDb movies.csv', low_memory=False)
        data = data[['duration', 'avg_vote', 'votes', 'budget', 'worlwide_gross_income']]
        data = data.dropna()
        data['budget'] = data['budget'].replace('[^0-9]', '', regex=True).astype(float)
        data['worlwide_gross_income'] = data['worlwide_gross_income'].replace('[^0-9]', '', regex=True).astype(float)
        return data
    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        return None

def main():
    st.title('IMDb Movies Box Office Prediction with MLflow')

    # Configurar MLflow
    mlflow.set_tracking_uri('http://localhost:5000')
    
    # Carregar dados
    data = load_and_preprocess_data()
    if data is None:
        return

    # Preparar dados
    X = data[['duration', 'avg_vote', 'votes', 'budget']]
    y = data['worlwide_gross_income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Interface do usuário
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button('Start MLflow Server'):
            if start_mlflow_server():
                st.success('MLflow server started successfully!')
                st.write('MLflow UI is running at http://localhost:5000')
            else:
                st.error('Failed to start MLflow server')

    with col2:
        if st.button('Train Model'):
            if not check_mlflow_server():
                st.warning('MLflow server is not running. Please start it first.')
                return
            
            try:
                with mlflow.start_run():
                    # Treinar modelo
                    st.info('Training model...')
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    
                    # Registrar modelo e parâmetros
                    mlflow.sklearn.log_model(model, 'model')
                    mlflow.log_param('n_estimators', 100)
                    
                    # Avaliar modelo
                    predictions = model.predict(X_test)
                    mse = mean_squared_error(y_test, predictions)
                    mlflow.log_metric('mse', mse)
                    
                    st.success(f'Model trained and logged with MSE: {mse:.2f}')
                    
                    # Mostrar algumas predições
                    st.subheader('Sample Predictions')
                    sample_df = pd.DataFrame({
                        'Actual': y_test[:5],
                        'Predicted': predictions[:5]
                    })
                    st.dataframe(sample_df)
                    
            except Exception as e:
                st.error(f'Error during model training: {str(e)}')

    # Mostrar status do MLflow
    st.sidebar.subheader('MLflow Status')
    server_status = "Running" if check_mlflow_server() else "Stopped"
    st.sidebar.write(f"Server Status: {server_status}")
    
    if server_status == "Running":
        try:
            # Mostrar experimentos existentes
            client = MlflowClient()
            experiments = mlflow.search_experiments()
            st.sidebar.subheader('Existing Experiments')
            for exp in experiments:
                st.sidebar.write(f"- {exp.name}")
        except Exception as e:
            st.sidebar.error(f"Error fetching experiments: {str(e)}")

if __name__ == '__main__':
    main()