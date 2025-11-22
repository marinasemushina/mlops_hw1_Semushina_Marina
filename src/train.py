import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import mlflow
import mlflow.sklearn
import os

def train_model():
    """Обучение модели с логированием в MLflow"""
    
    # Загружаем параметры
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Настройки MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("wine-classification")
    
    # Загружаем данные
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
    y_test = pd.read_csv('data/processed/y_test.csv').squeeze()
    
    # Начинаем эксперимент MLflow
    with mlflow.start_run():
        # Логируем параметры
        mlflow.log_params({
            "model_type": params['train']['model_type'],
            "n_estimators": params['train']['n_estimators'],
            "max_depth": params['train']['max_depth'],
            "random_state": params['train']['random_state'],
            "test_size": params['prepare']['test_size']
        })
        
        # Выбираем и обучаем модель
        if params['train']['model_type'] == "RandomForest":
            model = RandomForestClassifier(
                n_estimators=params['train']['n_estimators'],
                max_depth=params['train']['max_depth'],
                random_state=params['train']['random_state']
            )
        elif params['train']['model_type'] == "LogisticRegression":
            model = LogisticRegression(random_state=params['train']['random_state'])
        else:
            raise ValueError(f"Unknown model type: {params['train']['model_type']}")
        
        model.fit(X_train, y_train)
        
        # Предсказания и метрики
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Логируем метрики
        mlflow.log_metric("accuracy", accuracy)
        
        # Сохраняем модель
        mlflow.sklearn.log_model(model, "model")
        
        # Создаем и сохраняем confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        
        # Логируем артефакты
        mlflow.log_artifact('confusion_matrix.png')
        mlflow.log_artifact('params.yaml')
        
        # Сохраняем модель в файл
        import joblib
        joblib.dump(model, 'model.pkl')
        mlflow.log_artifact('model.pkl')
        
        print(f"Модель обучена. Точность: {accuracy:.4f}")
        print("Результаты сохранены в MLflow")
        
        # Выводим отчет классификации
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_model()
