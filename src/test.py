import torch
import model
from data_loader import train_loader, val_loader, test_loader
from model import ViTClassifier
import torch.nn as nn 
import torch.optim as optim 
import numpy as np
from torchmetrics.classification import BinaryAccuracy

device = "cuda" if torch.cuda.is_available() else "cpu"
metric = BinaryAccuracy().to(device)

model = ViTClassifier(num_classes= 1)
model = model.to(device)
model.load_state_dict(torch.load("best_model.pth"))
criterion = nn.BCEWithLogitsLoss()
metric.reset()
test_loss = 0.0

model.eval()
with torch.no_grad():
    for features, labels in test_loader:
        features = features.to(device)
        labels = labels.to(device)

        outputs = model(features)
            
        
        probabilities = torch.sigmoid(outputs)
        preds = (probabilities >= 0.5).float() 
        metric(preds, labels.view(-1, 1))
        loss = criterion(outputs, labels.float().view(-1, 1))
        test_loss += loss.item()

test_loss = test_loss/len(test_loader)
accuracy = metric.compute() 
print(f"Test Loss:{test_loss:.4f}")
print(f"Test Accuracy:{accuracy*100:.2f}%")