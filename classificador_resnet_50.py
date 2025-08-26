import numpy as np
import os
import matplotlib.pyplot as plt
import time
import pandas as pd

import torch.optim as optim
from torchmetrics import Accuracy
from torchvision import datasets, transforms, models

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
import torchmetrics

from torchvision.models import resnet50, ResNet50_Weights 
from collections import Counter
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import TQDMProgressBar, RichProgressBar

# Configurar precisión para Tensor Cores
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
NUM_WORKERS = 4
CLASES = 10

DEBUG_MODE = False  # True Debug, False Full
 
if DEBUG_MODE:
    BATCH_SIZE = 16
    NUM_EPOCHS = 2
    print("Modo depuración activado: usando dataset ligero y parámetros reducidos.")
else:
    BATCH_SIZE = 32
    NUM_EPOCHS = 500
    print("Modo completo activado: usando configuración completa.")

# Crear el directorio de checkpoints si no existe
os.makedirs("checkpoints/", exist_ok=True)

# DataModule class
class FMNIST_DataModule(pl.LightningDataModule):
    def __init__(self, data_path="./"):
        super().__init__()
        self.data_path = data_path

        # Definir transformaciones aquí para que estén disponibles en cualquier momento
        self.train_transform = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(num_output_channels=3),
            torchvision.transforms.RandomHorizontalFlip(p=0.30),
            torchvision.transforms.RandomVerticalFlip(p=0.30),
            torchvision.transforms.RandomRotation(30),
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485]*3, std=[0.229]*3)])

        self.test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(num_output_channels=3),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485]*3, std=[0.229]*3)])

    def prepare_data(self):
        # Este método puede quedar vacío si no se necesita descargar datos adicionales
        # datasets.FashionMNIST(root=self.data_path, train=True, download=True)
        # datasets.FashionMNIST(root=self.data_path, train=False, download=True)
        pass

    def setup(self, stage=None):
        train = datasets.FashionMNIST(
            root=self.data_path,
            train=True,
            transform=self.train_transform,
            download=True,
        )

        self.test = datasets.FashionMNIST(
            root=self.data_path,
            train=False,
            transform=self.test_transform,
            download=True,
        )

        self.train, self.valid = random_split(train, lengths=[int(len(train)*0.9), int(len(train)*0.1)])

        if DEBUG_MODE:
            print("DEBUG: Forzando el tamaño del conjunto de datos en setup.")
            # Comentando las líneas que limitan el tamaño del dataset en modo depuración
            # self.train = torch.utils.data.Subset(self.train, range(10000))  # 10000 muestras
            # self.valid = torch.utils.data.Subset(self.valid, range(2000))  # 2000 muestras
            print(f"DEBUG: Tamaño ajustado - Entrenamiento: {len(self.train)}, Validación: {len(self.valid)}")

        print("Muestras de entrenamiento:", len(self.train))
        print("Muestras de validación:", len(self.valid))
        print("Muestras de evaluación:", len(self.test))

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train,
            batch_size=BATCH_SIZE,
            drop_last=True,
            shuffle=True,
            num_workers=NUM_WORKERS,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid,
            batch_size=BATCH_SIZE,
            drop_last=False,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test,
            batch_size=BATCH_SIZE,
            drop_last=False,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )

# Model class
class ResNet50TransferLearning(pl.LightningModule):
    def __init__(self, num_classes=10, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
 
        # Cargar ResNet50 con pesos preentrenados
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
 
        # Modificar la primera capa y la capa final
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        for param in self.model.parameters():
            param.requires_grad = True
        # for param in self.model.layer4.parameters():
        #     param.requires_grad = True
        # for param in self.model.fc.parameters():
        #     param.requires_grad = True
 
        # Pérdida y métricas
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = Accuracy(task='multiclass', num_classes=num_classes)
 
    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(1)
        return loss, y, preds
    
    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch, stage="test")
        acc = self.accuracy(predicted_labels, true_labels)
        self.log('train_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, sync_dist=True)
        return loss
 
    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch, stage="test")
        acc = self.accuracy(predicted_labels, true_labels)
        self.log('valid_loss', loss, prog_bar=True, sync_dist=True)
        self.log('valid_acc', acc, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch, stage="test")
        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc, on_epoch=True, on_step=False, sync_dist=True)
 
    def configure_optimizers(self):
        return optim.Adam(self.model.fc.parameters(), lr=self.hparams.learning_rate)

# Main execution
if __name__ == "__main__":
    torch.manual_seed(47)

    # Initialize DataModule
    data_module = FMNIST_DataModule(data_path='./data')

    # Initialize Model
    pytorch_model = ResNet50TransferLearning(num_classes=10, learning_rate=1e-3)

    # Callbacks and Logger
    callback_check = ModelCheckpoint(
        monitor="valid_acc",  # Métrica a monitorear
        mode="max",           # Guardar el modelo con el valor máximo de valid_acc
        save_top_k=1,          # Guardar solo el mejor modelo
        save_last=True,        # Guardar siempre el último modelo
        dirpath="checkpoints/",  # Directorio donde guardar los checkpoints
        filename="best_model"    # Nombre del archivo del mejor modelo
    )
    callback_tqdm = RichProgressBar(leave=True)
    early_stop_callback = EarlyStopping(
        monitor='valid_loss',
        min_delta=0.00,
        patience=15,
        verbose=True,
        mode='min'
    )
    logger = CSVLogger(save_dir="logs/", name="complex-cnn-fmnist")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices='auto',
        log_every_n_steps=10,
        logger=logger,
        callbacks=[callback_check, callback_tqdm, early_stop_callback]
    )

    # Train
    start_time = time.time()
    trainer.fit(model=pytorch_model, datamodule=data_module)
    runtime = (time.time() - start_time) / 60
    print(f"Tiempo de entrenamiento en minutos: {runtime:.2f}")

    # Test
    trainer.test(model=pytorch_model, datamodule=data_module, ckpt_path='best')
