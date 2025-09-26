import torch
import model
from data_loader import train_loader, val_loader, test_loader
from model import ViTClassifier
import torch.nn as nn 
import torch.optim as optim 
import numpy as np
from torchmetrics.classification import BinaryAccuracy

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ViTClassifier(num_classes= 1)
metric = BinaryAccuracy().to(device)
model.to(device)
learning_rate = 1e-4

for name, param in model.named_parameters(): 
    if name != "classifier.classifier.weight" and name != "classifier.classifier.bias":
        param.requires_grad = False
    else:
        print(name)
        param.requires_grad = True

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr = learning_rate)
epochs = 20
min_valid_loss = np.inf

for epoch in range(epochs):
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

    print(f"Epoch: {epoch}, Loss: {training_loss/len(train_loader)}")
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
        print(f"Epoch: {epoch}, Val Loss: {val_loss/len(val_loader)}")
        if min_valid_loss > val_loss:
            print(f"Val Loss Decreased {min_valid_loss:.6f} ---> {val_loss:.6f} Saving The Model")
            min_valid_loss = val_loss
            torch.save(model.state_dict(), "saved_model.pth")
    accuracy = metric.compute() 
    print(f"Accuracy:{accuracy}")