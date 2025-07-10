# src/data_loader.py
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from . import config
import tifffile as tiff


def tiff_loader(path):
    """
    Carga una imagen TIFF y la convierte a un tensor.
    """
    img = tiff.imread(path)
    if img.ndim == 2:  # Si es una imagen en escala de grises
        img = img[:, :, None]  # Añadir una dimensión de canal
    return transforms.functional.to_tensor(img).float()  # Convertir a tensor y asegurarse de que sea float

def create_dataloaders():
    train_transform = transforms.Compose([
        #transforms.Grayscale(num_output_channels=config.NUM_CHANNELS),
        #transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        ## Para data augmentation
        transforms.RandomHorizontalFlip(p=0.5), # Voltea horizontalmente el 50% de las imágenes
        transforms.RandomRotation(15),           # Rota la imagen hasta 15 grados
        #transforms.ColorJitter(brightness=0.2, contrast=0.2), # Ajusta brillo y contraste
        # as array because it's a single channel and a tiff image
        #transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5], std=[0.5]),  # Normaliza a un rango de [-1, 1]
    ])

    val_transform = transforms.Compose([
        #transforms.Grayscale(num_output_channels=config.NUM_CHANNELS),
        #transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5], std=[0.5]) ,  # Normaliza a un rango de [-1, 1]
    ])

    train_dataset = datasets.ImageFolder(config.TRAIN_DATA_PATH, transform=train_transform, loader=tiff_loader)
    val_dataset = datasets.ImageFolder(config.VAL_DATA_PATH, transform=val_transform, loader=tiff_loader)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    
    class_names = list(val_dataset.class_to_idx.keys())
    
    return train_loader, val_loader, class_names