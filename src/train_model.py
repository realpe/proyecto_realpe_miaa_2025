"""
Módulo de Entrenamiento de Modelos de Machine Learning
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           mean_squared_error, mean_absolute_error, r2_score,
                           confusion_matrix, classification_report)
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, List
import os

class MLTrainer:
    """
    Clase para entrenamiento y evaluación de modelos de Machine Learning.
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.task_type = None
        
    def initialize_models(self, task_type: str = 'classification') -> None:
        """
        Inicializa modelos según el tipo de tarea.
        
        Args:
            task_type (str): Tipo de tarea ('classification' o 'regression')
        """
        self.task_type = task_type
        
        if task_type == 'classification':
            self.models = {
                'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
                'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
                'SVM': SVC(random_state=42),
                'DecisionTree': DecisionTreeClassifier(random_state=42),
                'KNeighbors': KNeighborsClassifier()
            }
        elif task_type == 'regression':
            self.models = {
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
                'LinearRegression': LinearRegression(),
                'SVM': SVR(),
                'DecisionTree': DecisionTreeRegressor(random_state=42),
                'KNeighbors': KNeighborsRegressor()
            }
        else:
            raise ValueError("task_type debe ser 'classification' o 'regression'")
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                    cv_folds: int = 5) -> Dict[str, Dict]:
        """
        Entrena todos los modelos y evalúa con validación cruzada.
        
        Args:
            X_train (pd.DataFrame): Datos de entrenamiento
            y_train (pd.Series): Etiquetas de entrenamiento
            cv_folds (int): Número de folds para validación cruzada
            
        Returns:
            Dict: Resultados de validación cruzada para cada modelo
        """
        print("Iniciando entrenamiento de modelos...")
        
        for name, model in self.models.items():
            print(f"\nEntrenando {name}...")
            
            # Entrenamiento
            model.fit(X_train, y_train)
            
            # Validación cruzada
            if self.task_type == 'classification':
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='accuracy')
                metric_name = 'accuracy'
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2')
                metric_name = 'r2'
            
            self.results[name] = {
                'model': model,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores,
                'metric': metric_name
            }
            
            print(f"{name} - {metric_name}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Encontrar el mejor modelo
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['cv_mean'])
        self.best_model = self.results[best_model_name]['model']
        print(f"\nMejor modelo: {best_model_name}")
        
        return self.results
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                      model_name: str = "Modelo") -> Dict[str, Any]:
        """
        Evalúa un modelo en el conjunto de prueba.
        
        Args:
            model: Modelo entrenado
            X_test (pd.DataFrame): Datos de prueba
            y_test (pd.Series): Etiquetas verdaderas
            model_name (str): Nombre del modelo
            
        Returns:
            Dict: Métricas de evaluación
        """
        predictions = model.predict(X_test)
        
        if self.task_type == 'classification':
            metrics = {
                'accuracy': accuracy_score(y_test, predictions),
                'precision': precision_score(y_test, predictions, average='weighted', zero_division=0),
                'recall': recall_score(y_test, predictions, average='weighted', zero_division=0),
                'f1': f1_score(y_test, predictions, average='weighted', zero_division=0)
            }
            
            print(f"\n=== EVALUACIÓN - {model_name} ===")
            for metric, value in metrics.items():
                print(f"{metric.capitalize()}: {value:.4f}")
                
            # Matriz de confusión
            cm = confusion_matrix(y_test, predictions)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Matriz de Confusión - {model_name}')
            plt.ylabel('Etiqueta Real')
            plt.xlabel('Etiqueta Predicha')
            plt.show()
            
            # Reporte de clasificación
            print(f"\nReporte de Clasificación - {model_name}:")
            print(classification_report(y_test, predictions))
            
        else:  # regression
            metrics = {
                'mse': mean_squared_error(y_test, predictions),
                'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                'mae': mean_absolute_error(y_test, predictions),
                'r2': r2_score(y_test, predictions)
            }
            
            print(f"\n=== EVALUACIÓN - {model_name} ===")
            for metric, value in metrics.items():
                print(f"{metric.upper()}: {value:.4f}")
                
            # Gráfico de predicciones vs valores reales
            plt.figure(figsize=(10, 6))
            plt.subplot(1, 2, 1)
            plt.scatter(y_test, predictions, alpha=0.7)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Valores Reales')
            plt.ylabel('Predicciones')
            plt.title(f'Predicciones vs Reales - {model_name}')
            
            # Residuos
            plt.subplot(1, 2, 2)
            residuals = y_test - predictions
            plt.scatter(predictions, residuals, alpha=0.7)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicciones')
            plt.ylabel('Residuos')
            plt.title(f'Gráfico de Residuos - {model_name}')
            
            plt.tight_layout()
            plt.show()
        
        return metrics
    
    def compare_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """
        Compara todos los modelos entrenados en el conjunto de prueba.
        
        Args:
            X_test (pd.DataFrame): Datos de prueba
            y_test (pd.Series): Etiquetas de prueba
            
        Returns:
            pd.DataFrame: Comparación de métricas
        """
        comparison_results = []
        
        for name, result in self.results.items():
            model = result['model']
            metrics = self.evaluate_model(model, X_test, y_test, name)
            
            row = {'Model': name, 'CV_Mean': result['cv_mean'], 'CV_Std': result['cv_std']}
            row.update(metrics)
            comparison_results.append(row)
        
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df = comparison_df.sort_values(by='CV_Mean', ascending=False)
        
        print("\n=== COMPARACIÓN DE MODELOS ===")
        print(comparison_df.to_string(index=False))
        
        return comparison_df
    
    def optimize_hyperparameters(self, model_name: str, param_grid: Dict, 
                                X_train: pd.DataFrame, y_train: pd.Series,
                                cv_folds: int = 5) -> Dict[str, Any]:
        """
        Optimiza hiperparámetros usando GridSearchCV.
        
        Args:
            model_name (str): Nombre del modelo a optimizar
            param_grid (Dict): Grilla de parámetros
            X_train (pd.DataFrame): Datos de entrenamiento
            y_train (pd.Series): Etiquetas de entrenamiento
            cv_folds (int): Número de folds para validación cruzada
            
        Returns:
            Dict: Resultados de la optimización
        """
        if model_name not in self.models:
            raise ValueError(f"Modelo {model_name} no encontrado")
        
        model = self.models[model_name]
        scoring = 'accuracy' if self.task_type == 'classification' else 'r2'
        
        print(f"Optimizando hiperparámetros para {model_name}...")
        
        grid_search = GridSearchCV(
            model, param_grid, cv=cv_folds, 
            scoring=scoring, n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Actualizar el modelo con los mejores parámetros
        self.models[model_name] = grid_search.best_estimator_
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        print(f"Mejores parámetros: {grid_search.best_params_}")
        print(f"Mejor puntuación: {grid_search.best_score_:.4f}")
        
        return results
    
    def get_feature_importance(self, model_name: str = None) -> pd.DataFrame:
        """
        Obtiene la importancia de características para modelos que la soportan.
        
        Args:
            model_name (str): Nombre del modelo (usa el mejor si no se especifica)
            
        Returns:
            pd.DataFrame: Importancia de características
        """
        if model_name is None:
            model = self.best_model
            name = "Mejor Modelo"
        else:
            model = self.results[model_name]['model']
            name = model_name
        
        if hasattr(model, 'feature_importances_'):
            # Para modelos basados en árboles
            feature_names = [f'Feature_{i}' for i in range(len(model.feature_importances_))]
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            # Gráfico de importancia
            plt.figure(figsize=(10, 6))
            sns.barplot(data=importance_df.head(15), x='Importance', y='Feature')
            plt.title(f'Importancia de Características - {name}')
            plt.tight_layout()
            plt.show()
            
            return importance_df
        
        elif hasattr(model, 'coef_'):
            # Para modelos lineales
            feature_names = [f'Feature_{i}' for i in range(len(model.coef_))]
            if len(model.coef_.shape) > 1:
                # Clasificación multiclase
                importance = np.mean(np.abs(model.coef_), axis=0)
            else:
                importance = np.abs(model.coef_)
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            return importance_df
        
        else:
            print(f"El modelo {name} no soporta importancia de características")
            return pd.DataFrame()
    
    def save_model(self, model_name: str = None, filepath: str = None) -> None:
        """
        Guarda un modelo entrenado.
        
        Args:
            model_name (str): Nombre del modelo a guardar
            filepath (str): Ruta donde guardar el modelo
        """
        if model_name is None:
            model = self.best_model
            name = "best_model"
        else:
            model = self.results[model_name]['model']
            name = model_name.lower()
        
        if filepath is None:
            filepath = f'models/{name}.joblib'
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model, filepath)
        print(f"Modelo guardado en: {filepath}")
    
    def load_model(self, filepath: str):
        """
        Carga un modelo previamente guardado.
        
        Args:
            filepath (str): Ruta del modelo guardado
            
        Returns:
            Modelo cargado
        """
        if os.path.exists(filepath):
            model = joblib.load(filepath)
            print(f"Modelo cargado desde: {filepath}")
            return model
        else:
            raise FileNotFoundError(f"Archivo no encontrado: {filepath}")

def get_default_param_grids() -> Dict[str, Dict]:
    """
    Retorna grillas de parámetros por defecto para optimización.
    
    Returns:
        Dict: Grillas de parámetros para cada algoritmo
    """
    return {
        'RandomForest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        },
        'KNeighbors': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance']
        }
    }