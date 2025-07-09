# src/model.py
import torch.nn as nn
import timm
from . import config

def create_model():
    model = timm.create_model(
        config.MODEL_NAME, 
        pretrained=True, 
        num_classes=config.NUM_CLASSES,
        in_chans=config.NUM_CHANNELS, 
        img_size=config.IMG_SIZE
    )
    num_features = model.head.in_features
    model.head = nn.Sequential(nn.Linear(num_features, config.NUM_CLASSES), nn.Sigmoid())
    model.to(config.DEVICE)
    return model