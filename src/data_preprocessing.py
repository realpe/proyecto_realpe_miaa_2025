"""
Módulo de Preprocesamiento de Datos
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Any
import joblib
import os

class DataPreprocessor:
    """
    Clase para preprocesamiento de datos con funcionalidades comunes.
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Maneja valores faltantes en el DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame con posibles valores faltantes
            strategy (str): Estrategia para imputación ('mean', 'median', 'mode', 'constant')
            
        Returns:
            pd.DataFrame: DataFrame con valores faltantes imputados
        """
        df_processed = df.copy()
        
        # Separar columnas numéricas y categóricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Imputar valores numéricos
        if len(numeric_cols) > 0:
            numeric_strategy = strategy if strategy in ['mean', 'median'] else 'mean'
            if 'numeric_imputer' not in self.imputers:
                self.imputers['numeric_imputer'] = SimpleImputer(strategy=numeric_strategy)
                df_processed[numeric_cols] = self.imputers['numeric_imputer'].fit_transform(df_processed[numeric_cols])
            else:
                df_processed[numeric_cols] = self.imputers['numeric_imputer'].transform(df_processed[numeric_cols])
        
        # Imputar valores categóricos
        if len(categorical_cols) > 0:
            if 'categorical_imputer' not in self.imputers:
                self.imputers['categorical_imputer'] = SimpleImputer(strategy='most_frequent')
                df_processed[categorical_cols] = self.imputers['categorical_imputer'].fit_transform(df_processed[categorical_cols])
            else:
                df_processed[categorical_cols] = self.imputers['categorical_imputer'].transform(df_processed[categorical_cols])
        
        return df_processed
    
    def encode_categorical_variables(self, df: pd.DataFrame, columns: List[str] = None, 
                                   method: str = 'onehot') -> pd.DataFrame:
        """
        Codifica variables categóricas.
        
        Args:
            df (pd.DataFrame): DataFrame con variables categóricas
            columns (List[str]): Lista de columnas a codificar
            method (str): Método de codificación ('onehot', 'label')
            
        Returns:
            pd.DataFrame: DataFrame con variables codificadas
        """
        df_processed = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=['object']).columns.tolist()
        
        for col in columns:
            if col in df_processed.columns:
                if method == 'onehot':
                    if col not in self.encoders:
                        self.encoders[col] = OneHotEncoder(sparse_output=False, drop='first')
                        encoded_cols = self.encoders[col].fit_transform(df_processed[[col]])
                        feature_names = [f"{col}_{cat}" for cat in self.encoders[col].categories_[0][1:]]
                    else:
                        encoded_cols = self.encoders[col].transform(df_processed[[col]])
                        feature_names = [f"{col}_{cat}" for cat in self.encoders[col].categories_[0][1:]]
                    
                    # Crear DataFrame con las nuevas columnas
                    encoded_df = pd.DataFrame(encoded_cols, columns=feature_names, index=df_processed.index)
                    df_processed = pd.concat([df_processed.drop(col, axis=1), encoded_df], axis=1)
                    
                elif method == 'label':
                    if col not in self.encoders:
                        self.encoders[col] = LabelEncoder()
                        df_processed[col] = self.encoders[col].fit_transform(df_processed[col])
                    else:
                        df_processed[col] = self.encoders[col].transform(df_processed[col])
        
        return df_processed
    
    def scale_features(self, df: pd.DataFrame, columns: List[str] = None, 
                      method: str = 'standard') -> pd.DataFrame:
        """
        Escala características numéricas.
        
        Args:
            df (pd.DataFrame): DataFrame con características numéricas
            columns (List[str]): Lista de columnas a escalar
            method (str): Método de escalado ('standard', 'minmax', 'robust')
            
        Returns:
            pd.DataFrame: DataFrame con características escaladas
        """
        df_processed = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if method == 'standard':
            scaler_key = 'standard_scaler'
            if scaler_key not in self.scalers:
                self.scalers[scaler_key] = StandardScaler()
                df_processed[columns] = self.scalers[scaler_key].fit_transform(df_processed[columns])
            else:
                df_processed[columns] = self.scalers[scaler_key].transform(df_processed[columns])
        
        return df_processed
    
    def remove_outliers(self, df: pd.DataFrame, columns: List[str] = None, 
                       method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Remueve outliers del DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame original
            columns (List[str]): Columnas a analizar para outliers
            method (str): Método para detección ('iqr', 'zscore')
            threshold (float): Umbral para considerar outlier
            
        Returns:
            pd.DataFrame: DataFrame sin outliers
        """
        df_processed = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if method == 'iqr':
                Q1 = df_processed[col].quantile(0.25)
                Q3 = df_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_processed = df_processed[(df_processed[col] >= lower_bound) & 
                                          (df_processed[col] <= upper_bound)]
            
            elif method == 'zscore':
                z_scores = np.abs((df_processed[col] - df_processed[col].mean()) / df_processed[col].std())
                df_processed = df_processed[z_scores <= threshold]
        
        return df_processed
    
    def create_train_test_split(self, X: pd.DataFrame, y: pd.Series, 
                              test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Crea división de entrenamiento y prueba.
        
        Args:
            X (pd.DataFrame): Características
            y (pd.Series): Variable objetivo
            test_size (float): Proporción del conjunto de prueba
            random_state (int): Semilla para reproducibilidad
            
        Returns:
            Tuple: (X_train, X_test, y_train, y_test)
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state, 
                               stratify=y if y.nunique() < 20 else None)
    
    def save_preprocessors(self, filepath: str = 'models/preprocessors.joblib') -> None:
        """
        Guarda los preprocesadores entrenados.
        
        Args:
            filepath (str): Ruta donde guardar los preprocesadores
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        preprocessors = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'imputers': self.imputers
        }
        
        joblib.dump(preprocessors, filepath)
        print(f"Preprocesadores guardados en: {filepath}")
    
    def load_preprocessors(self, filepath: str = 'models/preprocessors.joblib') -> None:
        """
        Carga preprocesadores previamente entrenados.
        
        Args:
            filepath (str): Ruta de los preprocesadores guardados
        """
        if os.path.exists(filepath):
            preprocessors = joblib.load(filepath)
            self.scalers = preprocessors.get('scalers', {})
            self.encoders = preprocessors.get('encoders', {})
            self.imputers = preprocessors.get('imputers', {})
            print(f"Preprocesadores cargados desde: {filepath}")
        else:
            print(f"Archivo no encontrado: {filepath}")

def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Genera datos de ejemplo para demostración.
    
    Args:
        n_samples (int): Número de muestras a generar
        
    Returns:
        pd.DataFrame: DataFrame con datos de ejemplo
    """
    np.random.seed(42)
    
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 15000, n_samples),
        'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'experience_years': np.random.randint(0, 40, n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples),
        'score': np.random.uniform(0, 100, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Introducir algunos valores faltantes
    df.loc[np.random.choice(df.index, 50), 'income'] = np.nan
    df.loc[np.random.choice(df.index, 30), 'education_level'] = np.nan
    
    return df