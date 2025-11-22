import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
import yaml
import os

def load_data():
    """Загрузка датасета Wine"""
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    return df

def prepare_data():
    """Подготовка данных и разделение на train/test"""
    
    # Загружаем параметры
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    test_size = params['prepare']['test_size']
    random_state = params['prepare']['random_state']
    
    # Загружаем данные
    df = load_data()
    
    # Разделяем на признаки и целевую переменную
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Разделяем на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Сохраняем обработанные данные
    os.makedirs('data/processed', exist_ok=True)
    
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    print("Данные успешно подготовлены и сохранены:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_test: {y_test.shape}")

if __name__ == "__main__":
    prepare_data()
