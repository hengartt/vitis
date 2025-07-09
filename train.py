# train.py
from src.data_loader import create_dataloaders
from src.model import create_model
from src.trainer import Trainer

if __name__ == '__main__':
    # 1. Cargar datos
    train_loader, val_loader, _ = create_dataloaders()
    
    # 2. Crear modelo
    model = create_model()
    
    # 3. Iniciar el entrenamiento
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader)
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n[INFO] Entrenamiento interrumpido por el usuario.")
        print("[OK] Entrenamiento finalizado.")