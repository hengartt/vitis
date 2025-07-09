# src/data_loader.py
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image
from . import config

def tiff_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')

def create_dataloaders():
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=config.NUM_CHANNELS),
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(config.TRAIN_DATA_PATH, transform=transform, loader=tiff_loader)
    val_dataset = datasets.ImageFolder(config.VAL_DATA_PATH, transform=transform, loader=tiff_loader)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    class_names = list(val_dataset.class_to_idx.keys())
    
    return train_loader, val_loader, class_names