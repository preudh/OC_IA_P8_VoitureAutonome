from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import io

app = FastAPI()

# Charger le modèle sauvegardé
model_path = "models/vgg16_unet_complete_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path, map_location=device).to(device)
model.eval()

# Transformation des images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Charger l'image
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Effectuer la prédiction
    with torch.no_grad():
        output = model(input_tensor)
        predicted_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    # Convertir le masque prédit en liste JSON sérialisable
    predicted_mask = predicted_mask.tolist()

    return JSONResponse(content={"predicted_mask": predicted_mask})
