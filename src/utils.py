# src/utils.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

def plot_normalized_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2%', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión Normalizada')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Predicción del Modelo')
    plt.savefig(save_path)
    plt.close()
    print(f"Matriz de confusión normalizada guardada en: {save_path}")

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Predicción del Modelo')
    plt.savefig(save_path)
    plt.close()
    print(f"Matriz de confusión guardada en: {save_path}")

def plot_training_results(csv_path, save_path):
    df = pd.read_csv(csv_path)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle('Resultados del Entrenamiento', fontsize=16)
    ax1.plot(df['epoch'], df['train_loss'], 'b-', label='Pérdida de Entrenamiento')
    ax1.plot(df['epoch'], df['val_loss'], 'r-', label='Pérdida de Validación')
    ax1.set_ylabel('Pérdida (Loss)'); ax1.legend(); ax1.grid(True)
    ax2.plot(df['epoch'], df['val_acc'], 'g-', label='Exactitud de Validación')
    ax2.set_xlabel('Época'); ax2.set_ylabel('Exactitud (Accuracy)'); ax2.legend(); ax2.grid(True)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(save_path); plt.close()
    print(f"Gráfica de resultados guardada en: {save_path}")

def visualize_and_save_batch(data_loader, class_names, save_path, grid_size=4):
    """
    Toma un lote del DataLoader, lo visualiza en una cuadrícula y lo guarda como una imagen.
    """
    print(f"Generando visualización de un lote de datos en: {save_path}")
    
    # --- Se obtiene un solo lote de datos ---
    # iter() crea un iterador, next() obtiene el siguiente (y primer) elemento
    images, labels = next(iter(data_loader))

    # Asegurarse de que el número de imágenes a mostrar no exceda el tamaño del lote
    num_images_to_show = min(len(images), grid_size * grid_size)

    # --- Se crea la cuadrícula para el ploteo ---
    # fig es la figura completa, axes es una matriz de sub-plots
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    
    # --- Iteracion y mostrar cada imagen en el lote ---
    for i in range(num_images_to_show):
        # Calcular la posición en la cuadrícula
        row = i // grid_size
        col = i % grid_size
        ax = axes[row, col]

        image_tensor = images[i]
        
        # Normalizar la imagen al rango [0, 1] para una visualización correcta
        min_val = image_tensor.min()
        max_val = image_tensor.max()
        # Evitar división por cero si la imagen es de un solo color
        if max_val > min_val:
            image_normalized = (image_tensor - min_val) / (max_val - min_val)
        else:
            image_normalized = image_tensor
        
        # Permutar y convertir a numpy para Matplotlib
        # Se usa la imagen normalizada para el ploteo
        image_to_plot = image_normalized.permute(1, 2, 0).numpy()

        label_idx = labels[i].item()
        label_name = class_names[label_idx]
        
        ax.imshow(image_to_plot, cmap='gray')
        ax.set_title(f"Clase: {label_name}")
        ax.axis('off')

    for i in range(num_images_to_show, grid_size * grid_size):
        axes[i // grid_size, i % grid_size].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()