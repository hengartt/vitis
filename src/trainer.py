# src/trainer.py
import torch
import os
import csv
import datetime
import numpy as np
from . import config
from torch.utils.data import DataLoader
from .utils import plot_training_results
from .utils import plot_confusion_matrix
from .utils import plot_normalized_confusion_matrix


class Trainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        self.device = config.DEVICE
        
        # Setup del directorio de resultados
        run_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = os.path.join(config.OUTPUT_DIR, run_name)
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"[INFO] Resultados se guardarán en: {self.results_dir}")

    def _train_epoch(self):
        self.model.train()
        total_loss = 0
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.float().unsqueeze(1).to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)
        
    def _eval_epoch(self):
        self.model.eval()
        total_loss, correct, total_samples = 0, 0, 0
        with torch.no_grad():
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.float().unsqueeze(1).to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                preds = (outputs > 0.5).float()
                correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
                total_loss += loss.item()
        return total_loss / len(self.train_loader), correct / total_samples

    def train(self):
        log_file_path = os.path.join(self.results_dir, 'results.csv')
        with open(log_file_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_acc'])

        best_val_acc = 0.0
        last_model_path = os.path.join(self.results_dir, 'last.pth')
        best_model_path = os.path.join(self.results_dir, 'best.pth')

        print("\n[INFO] Iniciando entrenamiento...")
        for epoch in range(config.NUM_EPOCHS):
            train_loss = self._train_epoch()
            val_loss, val_acc = self._eval_epoch()

            print(f"[{epoch+1}/{config.NUM_EPOCHS}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            with open(log_file_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{val_acc:.4f}"])

            torch.save(self.model.state_dict(), last_model_path)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), best_model_path)
                print(f"[INFO] Nuevo mejor modelo guardado.")
        
        print("[OK] Entrenamiento finalizado.")
        # Llamada a las funciones de reporte final de utils.py
        plot_training_results(log_file_path, os.path.join(self.results_dir, 'training_results.png'))

        # Evaluación final en el conjunto de validación
        val_loader = DataLoader(self.val_loader.dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
        self.model.load_state_dict(torch.load(best_model_path))
        y_pred, y_true = [], []
        self.model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = self.model(images.to(self.device))
                y_pred.extend((outputs > 0.5).cpu().numpy())
                y_true.extend(labels.cpu().numpy())
        
        y_pred = np.array(y_pred).flatten(); y_true = np.array(y_true)

        
