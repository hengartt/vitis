# src/config.py
import torch

# Parámetros del dispositivo
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Parámetros de los datos
IMG_SIZE = 512
NUM_CHANNELS = 1
TRAIN_DATA_PATH = 'data/train'
VAL_DATA_PATH = 'data/val'

# Parámetros del modelo
MODEL_NAME = 'vit_base_patch16_224'
NUM_CLASSES = 1 # Para clasificación wood_disease y sana

# Hiperparámetros de entrenamiento
BATCH_SIZE = 4
NUM_EPOCHS = 20
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-4

# Parámetros de salida
OUTPUT_DIR = 'runs'