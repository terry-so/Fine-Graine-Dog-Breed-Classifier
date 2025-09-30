import torch
import model
from data_loader import train_loader, val_loader, test_loader
from model import ViTClassifier
import torch.nn as nn 
import torch.optim as optim 
import numpy as np
from torchmetrics.classification import BinaryAccuracy
import wandb
import random

project = 'shiba_akita_sweep'

# https://medium.com/biased-algorithms/a-practical-guide-to-implementing-early-stopping-in-pytorch-for-model-training-99a7cbd46e9d
# Basic setup for early stopping criteria
patience = 5  # epochs to wait after no improvement
delta = 0.01  # minimum change in the monitored metric
best_val_loss = float("inf")  # best validation loss to compare against
no_improvement_count = 0  # count of epochs with no improvement

class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=True, path='best_model.pth'):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False
    
    def check_early_stop(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
            print(f"Validation loss decreased to {val_loss:.6f}. Saving model...")
            torch.save(model.state_dict(), self.path)
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed.")

# Initialize early stopping
early_stopping = EarlyStopping()

config = {"learning_rate": 0.0015457,
          "epochs": 100,
          "patience": 5}





device = "cuda" if torch.cuda.is_available() else "cpu"
model = ViTClassifier(num_classes= 1)
metric = BinaryAccuracy().to(device)
model.to(device)


for name, param in model.named_parameters(): 
    if name != "classifier.classifier.weight" and name != "classifier.classifier.bias":
        param.requires_grad = False
    else:
        print(name)
        param.requires_grad = True



with wandb.init(config = config) as run:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=run.config.learning_rate)
    early_stopping = EarlyStopping(patience = run.config.patience)

    for epoch in range(run.config.epochs):
        model.train()
        training_loss = 0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(features)
            loss = criterion(outputs, labels.float().view(-1, 1))
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
        avg_train_loss = training_loss/len(train_loader)
        print(f"Epoch: {epoch}, Loss: {avg_train_loss}")
        val_loss = 0
        metric.reset()
        model.eval()
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                labels = labels.to(device)

                v_outputs = model(features)
                
                loss = criterion(v_outputs, labels.float().view(-1, 1))
                probabilities = torch.sigmoid(v_outputs)
                preds = (probabilities >= 0.5).float() 
                metric(preds, labels.view(-1, 1))
                val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch: {epoch}/{run.config.epochs}, Val Loss: {avg_val_loss}")

            accuracy = metric.compute() 
            run.log({"train_loss": avg_train_loss, "val_acc": accuracy, "val_loss": avg_val_loss })
            print(f"Validation Accuracy:{accuracy}")

            early_stopping.check_early_stop(avg_val_loss)
    
            if early_stopping.stop_training:
                print(f"Early stopping at epoch {epoch}")
                run.finish()
                break
            
        