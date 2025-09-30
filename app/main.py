from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
from pathlib import Path
import uvicorn
from pydantic import BaseModel
import sys
import os
from src.model import ViTClassifier
import torch
from torchvision.transforms import v2
from PIL import Image
import io


device = "cuda" if torch.cuda.is_available() else "cpu"
model = ViTClassifier(num_classes= 1)
model = model.to(device)
model.load_state_dict(torch.load("src/best_model.pth",weights_only=True))


inference_transforms = v2.Compose([
   v2.ToImage(), 
   v2.ToDtype(torch.float32, scale=True),
   v2.Resize((224,224)),
   v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])


app = FastAPI()


@app.post("/predict")
async def inference(file: UploadFile = File(...)):
    contents = await file.read()

    # if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
    #     raise HTTPException(
    #         status_code=415,
    #         detail=f"Unsupported media type: {file.content_type}. Only JPEG, PNG, or WEBP allowed."
    #     )
    
    img = io.BytesIO(contents)
    img = Image.open(img).convert("RGB")
    model.eval()
    with torch.no_grad():
        output = inference_transforms(img)
        output = torch.unsqueeze(output,0)
        output = output.to(device)
        output = model(output)
        probabilities = torch.sigmoid(output)
        pred = (probabilities >= 0.5).float() 
        dog_breed = "shiba" if pred == 1 else "akita"
        return {"result":dog_breed}

# uvicorn app.main:app --reload
#curl.exe -X POST "http://127.0.0.1:8000/predict" -F "file=@C:\Users\hozen\Desktop\Projects\Fine-Grained-Dog-Breed-Classifier\inference_folder\Shiba_inu_taiki.jpg"