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