#!/usr/bin/env python3
"""
Script Principal - Proyecto Final MIAA 2025
Universidad ICESI

Este script ejecuta todo el pipeline de machine learning:
1. Generación/carga de datos
2. Análisis exploratorio
3. Preprocesamiento
4. Entrenamiento de modelos
5. Evaluación y selección del mejor modelo

Uso:
    python main.py [opciones]

Opciones:
    --data-file: Ruta al archivo de datos (opcional, se generan datos de ejemplo si no se proporciona)
    --target: Nombre de la columna objetivo
    --task-type: 'classification' o 'regression' (se detecta automáticamente si no se especifica)
    --test-size: Proporción del conjunto de prueba (default: 0.2)
    --cv-folds: Número de folds para validación cruzada (default: 5)
    --optimize: Activar optimización de hiperparámetros (default: False)
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# Agregar el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor, generate_sample_data
from train_model import MLTrainer, get_default_param_grids
from utils import basic_info, save_results


def main():
    parser = argparse.ArgumentParser(description='Pipeline de Machine Learning - MIAA 2025')
    parser.add_argument('--data-file', type=str, help='Ruta al archivo de datos')
    parser.add_argument('--target', type=str, default='score', help='Nombre de la columna objetivo')
    parser.add_argument('--task-type', type=str, choices=['classification', 'regression'], 
                       help='Tipo de tarea')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proporción del conjunto de prueba')
    parser.add_argument('--cv-folds', type=int, default=5, help='Número de folds para CV')
    parser.add_argument('--optimize', action='store_true', help='Optimizar hiperparámetros')
    parser.add_argument('--verbose', action='store_true', help='Mostrar información detallada')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🎓 PROYECTO FINAL - INTELIGENCIA ARTIFICIAL APLICADA")
    print("📍 Universidad ICESI - MIAA 2025")
    print("=" * 70)
    print(f"⏰ Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 1. CARGA DE DATOS
    print("📊 PASO 1: CARGA DE DATOS")
    print("-" * 30)
    
    if args.data_file and os.path.exists(args.data_file):
        print(f"📁 Cargando datos desde: {args.data_file}")
        try:
            if args.data_file.endswith('.csv'):
                df_raw = pd.read_csv(args.data_file)
            elif args.data_file.endswith(('.xlsx', '.xls')):
                df_raw = pd.read_excel(args.data_file)
            else:
                print(f"❌ Formato de archivo no soportado: {args.data_file}")
                return
        except Exception as e:
            print(f"❌ Error al cargar datos: {e}")
            return
    else:
        print("🔄 Generando datos de ejemplo...")
        df_raw = generate_sample_data(n_samples=1000)
    
    print(f"✅ Datos cargados: {df_raw.shape[0]} filas, {df_raw.shape[1]} columnas")
    
    if args.verbose:
        print("\n📋 Información básica de los datos:")
        basic_info(df_raw)
    
    # 2. PREPROCESAMIENTO
    print(f"\n🔧 PASO 2: PREPROCESAMIENTO DE DATOS")
    print("-" * 40)
    
    preprocessor = DataPreprocessor()
    
    # Manejo de valores faltantes
    print("🔍 Imputando valores faltantes...")
    df = preprocessor.handle_missing_values(df_raw, strategy='mean')
    
    # Codificación de variables categóricas
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if len(categorical_cols) > 0:
        print(f"🔤 Codificando {len(categorical_cols)} variables categóricas...")
        df = preprocessor.encode_categorical_variables(df, categorical_cols, method='onehot')
    
    # Escalado de características
    print("📏 Escalando características numéricas...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = preprocessor.scale_features(df, numeric_cols, method='standard')
    
    # Separar características y variable objetivo
    if args.target in df.columns:
        X = df.drop(columns=[args.target])
        y = df[args.target]
        print(f"🎯 Variable objetivo: {args.target}")
    else:
        print(f"⚠️  Variable objetivo '{args.target}' no encontrada. Usando datos completos como características.")
        X = df
        y = pd.Series(np.random.randint(0, 2, len(df)), name='synthetic_target')
        print("🔄 Variable objetivo sintética creada para demostración")
    
    # Determinar tipo de tarea
    if args.task_type:
        task_type = args.task_type
    else:
        if y.dtype == 'object' or y.nunique() < 10:
            task_type = 'classification'
        else:
            task_type = 'regression'
    
    # Convertir a clasificación si se especifica pero es continua
    if task_type == 'classification' and (y.dtype != 'object' and y.nunique() > 10):
        print("🔄 Convirtiendo variable continua a clasificación binaria...")
        y = (y > y.median()).astype(int)
        print(f"   Clases: {np.unique(y)}")
        print(f"   Distribución: {dict(pd.Series(y).value_counts())}")
    
    print(f"🎯 Tipo de problema detectado: {task_type.upper()}")
    
    if task_type == 'classification':
        print(f"   • Clases: {np.unique(y)}")
        print(f"   • Distribución: {dict(pd.Series(y).value_counts())}")
    else:
        print(f"   • Rango: [{y.min():.2f}, {y.max():.2f}]")
        print(f"   • Media: {y.mean():.2f}")
    
    
    # División train/test
    X_train, X_test, y_train, y_test = preprocessor.create_train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )
    
    print(f"✅ División completada: {X_train.shape[0]} train / {X_test.shape[0]} test")
    
    # Guardar datos procesados
    os.makedirs('data/processed', exist_ok=True)
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    preprocessor.save_preprocessors('models/preprocessors.joblib')
    
    # 3. ENTRENAMIENTO DE MODELOS
    print(f"\n🤖 PASO 3: ENTRENAMIENTO DE MODELOS")
    print("-" * 40)
    
    trainer = MLTrainer()
    trainer.initialize_models(task_type=task_type)
    
    print(f"📋 Modelos a evaluar: {list(trainer.models.keys())}")
    print(f"🔄 Entrenando con validación cruzada ({args.cv_folds} folds)...")
    
    cv_results = trainer.train_models(X_train, y_train, cv_folds=args.cv_folds)
    
    # Encontrar mejor modelo
    best_model_name = max(cv_results.keys(), key=lambda x: cv_results[x]['cv_mean'])
    best_cv_score = cv_results[best_model_name]['cv_mean']
    
    print(f"🏆 Mejor modelo: {best_model_name} (CV Score: {best_cv_score:.4f})")
    
    # 4. EVALUACIÓN EN CONJUNTO DE PRUEBA
    print(f"\n📊 PASO 4: EVALUACIÓN EN CONJUNTO DE PRUEBA")
    print("-" * 50)
    
    print("🔍 Evaluando todos los modelos...")
    test_results = trainer.compare_models(X_test, y_test)
    
    # Obtener métricas del mejor modelo
    best_model = trainer.best_model
    y_pred = best_model.predict(X_test)
    
    if task_type == 'classification':
        from sklearn.metrics import accuracy_score, f1_score
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"📈 Accuracy: {accuracy:.4f}")
        print(f"📈 F1-Score: {f1:.4f}")
    else:
        from sklearn.metrics import r2_score, mean_squared_error
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"📈 R²: {r2:.4f}")
        print(f"📈 RMSE: {rmse:.4f}")
    
    # 5. OPTIMIZACIÓN DE HIPERPARÁMETROS (OPCIONAL)
    if args.optimize:
        print(f"\n⚙️  PASO 5: OPTIMIZACIÓN DE HIPERPARÁMETROS")
        print("-" * 50)
        
        param_grids = get_default_param_grids()
        if best_model_name in param_grids:
            print(f"🔧 Optimizando {best_model_name}...")
            optimization_results = trainer.optimize_hyperparameters(
                best_model_name, param_grids[best_model_name], 
                X_train, y_train, cv_folds=args.cv_folds
            )
            
            print(f"✅ Optimización completada")
            print(f"📊 Mejores parámetros: {optimization_results['best_params']}")
            print(f"📈 Mejor CV Score: {optimization_results['best_score']:.4f}")
            
            # Re-evaluar modelo optimizado
            optimized_metrics = trainer.evaluate_model(
                trainer.models[best_model_name], X_test, y_test, 
                f"{best_model_name} (Optimizado)"
            )
        else:
            print(f"⚠️  No hay grilla de parámetros para {best_model_name}")
    
    # 6. GUARDADO DE RESULTADOS
    print(f"\n💾 PASO 6: GUARDADO DE RESULTADOS")
    print("-" * 40)
    
    # Crear directorios
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Guardar modelos
    trainer.save_model(model_name=None, filepath=f'models/best_model_{task_type}.joblib')
    print(f"✅ Mejor modelo guardado: models/best_model_{task_type}.joblib")
    
    # Guardar resultados
    save_results(test_results, f'model_comparison_{task_type}.csv')
    
    # Crear resumen final
    summary = {
        'timestamp': datetime.now().isoformat(),
        'task_type': task_type,
        'best_model': best_model_name,
        'best_cv_score': best_cv_score,
        'dataset_shape': {
            'original': df_raw.shape,
            'processed': X.shape,
            'train': X_train.shape,
            'test': X_test.shape
        },
        'target_column': args.target,
        'models_evaluated': list(trainer.models.keys()),
        'preprocessing_steps': [
            'handle_missing_values',
            'encode_categorical_variables', 
            'scale_features',
            'train_test_split'
        ]
    }
    
    save_results(summary, f'pipeline_summary_{task_type}.csv')
    
    # RESUMEN FINAL
    print(f"\n🎉 PIPELINE COMPLETADO EXITOSAMENTE!")
    print("=" * 70)
    print(f"🏆 Mejor modelo: {best_model_name}")
    print(f"📊 CV Score: {best_cv_score:.4f}")
    print(f"📁 Archivos generados:")
    print(f"   • models/best_model_{task_type}.joblib")
    print(f"   • results/model_comparison_{task_type}.csv")
    print(f"   • results/pipeline_summary_{task_type}.csv")
    print(f"   • data/processed/[X_train, X_test, y_train, y_test].csv")
    print(f"⏰ Tiempo total: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    print(f"\n📝 PRÓXIMOS PASOS RECOMENDADOS:")
    print(f"1. Revisar los notebooks en 'notebooks/' para análisis detallado")
    print(f"2. Explorar la importancia de características")
    print(f"3. Probar con datos reales específicos de tu problema")
    print(f"4. Considerar ingeniería de características adicional")
    print(f"5. Evaluar el modelo en producción")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Ejecución interrumpida por el usuario")
    except Exception as e:
        print(f"\n❌ Error durante la ejecución: {e}")
        if "--verbose" in sys.argv:
            import traceback
            traceback.print_exc()