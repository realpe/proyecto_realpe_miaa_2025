"""
Módulo de Utilidades para el Proyecto MIAA 2025
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Any
import os

def load_data(filepath: str) -> pd.DataFrame:
    """
    Carga datos desde un archivo CSV, Excel o Parquet.
    
    Args:
        filepath (str): Ruta al archivo de datos
        
    Returns:
        pd.DataFrame: DataFrame con los datos cargados
    """
    file_extension = os.path.splitext(filepath)[1].lower()
    
    try:
        if file_extension == '.csv':
            return pd.read_csv(filepath)
        elif file_extension in ['.xlsx', '.xls']:
            return pd.read_excel(filepath)
        elif file_extension == '.parquet':
            return pd.read_parquet(filepath)
        else:
            raise ValueError(f"Formato de archivo no soportado: {file_extension}")
    except Exception as e:
        print(f"Error al cargar el archivo {filepath}: {e}")
        return pd.DataFrame()

def basic_info(df: pd.DataFrame) -> None:
    """
    Muestra información básica sobre el DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame a analizar
    """
    print("=== INFORMACIÓN BÁSICA DEL DATASET ===")
    print(f"Forma del dataset: {df.shape}")
    print(f"Memoria utilizada: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("\n=== TIPOS DE DATOS ===")
    print(df.dtypes)
    print("\n=== VALORES FALTANTES ===")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_info = pd.DataFrame({
        'Faltantes': missing,
        'Porcentaje': missing_percent
    })
    print(missing_info[missing_info['Faltantes'] > 0])

def plot_distributions(df: pd.DataFrame, numeric_cols: List[str] = None) -> None:
    """
    Grafica las distribuciones de variables numéricas.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos
        numeric_cols (List[str]): Lista de columnas numéricas a graficar
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, col in enumerate(numeric_cols):
        row, col_idx = divmod(i, n_cols)
        ax = axes[row, col_idx] if n_rows > 1 else axes[col_idx]
        
        df[col].hist(ax=ax, bins=30, alpha=0.7)
        ax.set_title(f'Distribución de {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frecuencia')
    
    # Ocultar axes vacíos
    for i in range(len(numeric_cols), n_rows * n_cols):
        row, col_idx = divmod(i, n_cols)
        axes[row, col_idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def correlation_heatmap(df: pd.DataFrame, figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Crea un mapa de calor de correlaciones.
    
    Args:
        df (pd.DataFrame): DataFrame con datos numéricos
        figsize (Tuple[int, int]): Tamaño de la figura
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Matriz de Correlaciones')
    plt.tight_layout()
    plt.show()

def save_results(data: Any, filename: str, results_dir: str = 'results') -> None:
    """
    Guarda resultados en el directorio de resultados.
    
    Args:
        data: Datos a guardar
        filename (str): Nombre del archivo
        results_dir (str): Directorio de resultados
    """
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)
    
    if isinstance(data, pd.DataFrame):
        data.to_csv(filepath, index=False)
    elif isinstance(data, dict):
        pd.Series(data).to_csv(filepath)
    else:
        # Para otros tipos, guardamos como texto
        with open(filepath, 'w') as f:
            f.write(str(data))
    
    print(f"Resultados guardados en: {filepath}")