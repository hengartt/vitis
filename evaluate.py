# evaluate.py
import torch
import argparse
from src.data_loader import create_dataloaders
from src.model import create_model
from src.utils import plot_normalized_confusion_matrix
from src.utils import plot_confusion_matrix
# Lógica para cargar un modelo de una ruta específica de 'runs' y evaluarlo.
# Usa argparse para pasar la ruta del checkpoint como argumento.