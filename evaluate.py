import torch
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Importa los módulos de tu proyecto desde la carpeta 'src'
from src import config
from src.data_loader import create_dataloaders
from src.model import create_model
from src.utils import plot_normalized_confusion_matrix, plot_training_results

def main():
    """
    Función principal para cargar un modelo entrenado y evaluarlo.
    """
    # CONFIGURACIÓN DE LOS ARGUMENTOS DE LA LÍNEA DE COMANDOS ---
    parser = argparse.ArgumentParser(description="Evaluar un modelo ViT entrenado.")
    parser.add_argument(
        '--run_dir', 
        type=str, 
        required=True, 
        help='Ruta al directorio de la ejecución de entrenamiento (ej: runs/20250709_141218)'
    )
    args = parser.parse_args()

    # Verificar si el directorio de la ejecución existe
    if not os.path.isdir(args.run_dir):
        print(f"Error: El directorio '{args.run_dir}' no existe.")
        return

    # --- CARGA DE DATOS ---
    # Solo necesitamos el cargador de validación y los nombres de las clases
    _, val_loader, class_names = create_dataloaders()
    print(f"Datos de validación cargados. Clases: {class_names}")

    # --- RECREACIÓN DE LA ARQUITECTURA Y CARGAR EL MEJOR MODELO ---
    model = create_model()
    
    # Construir la ruta al mejor modelo guardado
    model_path = os.path.join(args.run_dir, 'best.pth')
    if not os.path.exists(model_path):
        print(f"Error: No se encontró el archivo 'best.pth' en '{args.run_dir}'.")
        return
        
    # Cargar los pesos del modelo
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.to(config.DEVICE)
    model.eval() # Poner en modo de evaluación
    print(f"Modelo cargado desde: {model_path}")

    # --- 4. REALIZAR PREDICCIONES ---
    print("\nRealizando predicciones en el conjunto de validación...")
    y_pred = []
    y_true = []

    with torch.no_grad(): # Desactiva el cálculo de gradientes
        for images, labels in val_loader:
            images = images.to(config.DEVICE)
            
            outputs = model(images)
            predicted = (outputs > 0.5).cpu().numpy()
            
            y_pred.extend(predicted)
            y_true.extend(labels.cpu().numpy())

    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true)

    # --- GENERACIÓN DE REPORTA Y GUARDAR ---
    print("Generando reportes...")
    
    # Guardar reporte de clasificación en un archivo
    report_path = os.path.join(args.run_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write("--- REPORTE DE CLASIFICACIÓN (Mejor Modelo) ---\n\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    print(f"Reporte de clasificación guardado en: {report_path}")

    # Guardar matriz de confusión (normal)
    cm_path = os.path.join(args.run_dir, 'evaluation_confusion_matrix.png')
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión (Evaluación Final)')
    plt.ylabel('Etiqueta Real'); plt.xlabel('Predicción del Modelo')
    plt.savefig(cm_path)
    plt.close()
    print(f"Matriz de confusión guardada en: {cm_path}")
    
    # Guardar matriz de confusión (normalizada)
    norm_cm_path = os.path.join(args.run_dir, 'evaluation_confusion_matrix_normalized.png')
    plot_normalized_confusion_matrix(y_true, y_pred, class_names, norm_cm_path)

    # Re-generar la gráfica de resultados del entrenamiento (para consistencia)
    csv_path = os.path.join(args.run_dir, 'results.csv')
    if os.path.exists(csv_path):
        results_plot_path = os.path.join(args.run_dir, 'training_results.png')
        plot_training_results(csv_path, results_plot_path)

    print("\nEvaluación completada.")

if __name__ == '__main__':
    main()