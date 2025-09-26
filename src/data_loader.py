from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.transforms import v2
from pathlib import Path
import torch
import os 
from torch.utils.data import DataLoader

train_transforms = v2.Compose([
   v2.RandomHorizontalFlip(),
   v2.RandomRotation(45),
   v2.ToImage(), 
   v2.ToDtype(torch.float32, scale=True),
   v2.Resize((224,224)),
   v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

val_transforms = v2.Compose([
   v2.ToImage(), 
   v2.ToDtype(torch.float32, scale=True),
   v2.Resize((224,224)),
   v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

test_transforms = v2.Compose([
   v2.ToImage(), 
   v2.ToDtype(torch.float32, scale=True),
   v2.Resize((224,224)),
   v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

cwd = Path.cwd()
parent = cwd.parent
train_dir  = os.path.join(parent, "data", "Train")
val_dir  = os.path.join(parent, "data", "Val")
test_dir  = os.path.join(parent, "data", "Test")

dataset_train = ImageFolder(train_dir, transform = train_transforms)
dataset_val = ImageFolder(val_dir, transform = val_transforms)
dataset_test = ImageFolder(test_dir, transform = test_transforms)

train_loader = DataLoader(dataset_train, shuffle = True, batch_size=32)
val_loader = DataLoader(dataset_val, shuffle = True, batch_size=32)
test_loader = DataLoader(dataset_test, shuffle = True, batch_size=32)

