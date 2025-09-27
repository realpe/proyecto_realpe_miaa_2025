"""
Proyecto Final - MIAA 2025 - Universidad ICESI
Inteligencia Artificial Aplicada

Este paquete contiene módulos para análisis de datos y machine learning:
- utils: Funciones utilitarias para manipulación de datos
- data_preprocessing: Preprocesamiento y limpieza de datos
- train_model: Entrenamiento y evaluación de modelos ML
"""

__version__ = "1.0.0"
__author__ = "realpe"
__institution__ = "Universidad ICESI"
__course__ = "Inteligencia Artificial Aplicada - MIAA 2025"

from .utils import (
    load_data,
    basic_info,
    plot_distributions,
    correlation_heatmap,
    save_results
)

from .data_preprocessing import (
    DataPreprocessor,
    generate_sample_data
)

from .train_model import (
    MLTrainer,
    get_default_param_grids
)

__all__ = [
    'load_data',
    'basic_info', 
    'plot_distributions',
    'correlation_heatmap',
    'save_results',
    'DataPreprocessor',
    'generate_sample_data',
    'MLTrainer',
    'get_default_param_grids'
]