import torch
import model
from model import ViTClassifier
import torch.nn as nn 
import torch.optim as optim 
import numpy as np
from torchvision import transforms
from torchvision.transforms import v2
from PIL import Image
import argparse



device = "cuda" if torch.cuda.is_available() else "cpu"

model = ViTClassifier(num_classes= 1)
model = model.to(device)
model.load_state_dict(torch.load("best_model.pth",weights_only=True))


inference_transforms = v2.Compose([
   v2.ToImage(), 
   v2.ToDtype(torch.float32, scale=True),
   v2.Resize((224,224)),
   v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

parser = argparse.ArgumentParser()
parser.add_argument("image_path", type= str, help = "path of input image",)
args = parser.parse_args()

def inference(image_path):
    img = Image.open(image_path).convert("RGB")
    model.eval()
    with torch.no_grad():
        output = inference_transforms(img)
        output = torch.unsqueeze(output,0)
        output = output.to(device)
        output = model(output)
        probabilities = torch.sigmoid(output)
        pred = (probabilities >= 0.5).float() 
        return "Shiba" if pred == 1 else "Akita"
    


if __name__ == "__main__":
    
    from pathlib import Path
    import os 
    import shutil

    cwd = Path.cwd()
    parent = cwd.parent
    inference_dir  = os.path.join(parent, "inference_folder")

    inference_files = [f for f in os.listdir(inference_dir) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.webp')]

    for file in inference_files:
        
       
        print(inference(image_path = os.path.join(inference_dir, file)))