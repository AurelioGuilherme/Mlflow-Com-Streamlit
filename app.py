import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os
from dotenv import load_dotenv

import dagshub
dagshub.init(repo_owner='aurelioguilherme',
             repo_name='Mlflow-Com-Streamlit',
             mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)

# Carregar variáveis de ambiente
load_dotenv()

def init_mlflow():
    """Initialize MLflow with DagsHub"""
    try:
        # Configurar MLflow - DagsHub
        MLFLOW_TRACKING_URI = os.getenv('https://dagshub.com/api/v1/repo-buckets/s3/AurelioGuilherme')
        MLFLOW_TRACKING_USERNAME = os.getenv('MLFLOW_TRACKING_USERNAME')
        MLFLOW_TRACKING_PASSWORD = os.getenv('MLFLOW_TRACKING_PASSWORD')

        # Configurar URI com autenticação básica
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Set experiment
        experiment_name = "imdb_box_office_prediction"
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            st.error(f"Erro ao configurar experimento MLflow: {str(e)}")
            return False
            
        return True
        
    except Exception as e:
        st.error(f"Erro ao inicializar MLflow: {str(e)}")
        return False

@st.cache_data
def load_and_preprocess_data():
    """Carrega e pré-processa o dataset do IMDb"""
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

def train_model(X_train, X_test, y_train, y_test, params):
    """Treina e registra o modelo com MLflow"""
    try:
        with mlflow.start_run() as run:
            # Registrar parâmetros
            mlflow.log_params(params)
            
            # Treinar modelo
            model = RandomForestRegressor(**params)
            model.fit(X_train, y_train)
            
            # Fazer predições e calcular métricas
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            
            # Registrar métricas
            mlflow.log_metrics({
                "mse": mse,
                "rmse": rmse
            })
            
            # Registrar modelo
            mlflow.sklearn.log_model(model, "model")
            
            return model, predictions, mse, run.info.run_id
            
    except Exception as e:
        st.error(f"Erro no treinamento do modelo: {str(e)}")
        return None, None, None, None
def get_mlflow_ui_url():
    """Retorna a URL do MLflow UI no DagsHub"""
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI', '')
    # Remove '.mlflow' do final da URI se existir
    if tracking_uri.endswith('.mlflow'):
        tracking_uri = tracking_uri[:-7]
    return tracking_uri

def main():
    st.title('IMDb Movies Box Office Prediction com MLflow')
    
    # Inicializar MLflow
    if not init_mlflow():
        st.error("Falha ao inicializar MLflow. Verifique sua configuração.")
        return
    
        # Adicionar botão para MLflow UI no topo
    _, col2 = st.columns([3, 1])
    with col2:
        mlflow_ui_url = get_mlflow_ui_url()
        if st.button('Abrir MLflow UI'):
            # Usar HTML para abrir em nova aba
            js = f"window.open('{mlflow_ui_url}')"
            html = f'<script language="javascript">{js}</script>'
            st.components.v1.html(html, height=0)
            st.success(f"MLflow UI aberto em nova aba!")
    
    # Carregar dados
    data = load_and_preprocess_data()
    if data is None:
        return
        
    # Preparar dados
    X = data[['duration', 'avg_vote', 'votes', 'budget']]
    y = data['worlwide_gross_income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Parâmetros do modelo
    st.sidebar.subheader("Parâmetros do Modelo")
    n_estimators = st.sidebar.slider("Número de árvores", 10, 200, 100)
    max_depth = st.sidebar.slider("Profundidade máxima", 1, 30, 10)
    min_samples_split = st.sidebar.slider("Mínimo de amostras para divisão", 2, 10, 2)
    
    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "random_state": 42
    }
    
    # Botão de treinamento
    if st.button('Treinar Modelo'):
        with st.spinner('Treinando modelo...'):
            model, predictions, mse, run_id = train_model(
                X_train, X_test, y_train, y_test, params
            )
            
            if model is not None:
                st.success(f'Modelo treinado com sucesso! MSE: {mse:.2f}')
                st.write(f'MLflow Run ID: {run_id}')
                
                # Mostrar predições de exemplo
                st.subheader('Amostra de Predições')
                results_df = pd.DataFrame({
                    'Real': y_test[:5],
                    'Previsto': predictions[:5],
                    'Diferença': abs(y_test[:5] - predictions[:5])
                })
                st.dataframe(results_df)
                
                # Importância das features
                st.subheader('Importância das Features')
                importance_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Importância': model.feature_importances_
                }).sort_values('Importância', ascending=False)
                st.dataframe(importance_df)

if __name__ == '__main__':
    main()