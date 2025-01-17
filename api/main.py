from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import io
import os
from models import VGG16UNet  # Importer le modèle défini dans models.py

app = FastAPI()

# Charger le modèle sauvegardé
model_path = os.path.join(os.getcwd(), "models/vgg16_unet_model.pth")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recréer l'architecture du modèle et charger les poids
model = VGG16UNet(num_classes = 8).to(device)
model.load_state_dict(torch.load(model_path, map_location = device))
model.eval()

# Transformation des images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Endpoint pour effectuer une prédiction sur une image donnée et retourner le masque sous forme de fichier en mémoire.
    """
    try:
        # Charger l'image
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Effectuer la prédiction
        with torch.no_grad():
            output = model(input_tensor)
            predicted_mask = torch.argmax(output, dim = 1).squeeze().cpu().numpy()

        # Convertir le masque en image
        mask_image = Image.fromarray((predicted_mask * 10).astype(np.uint8))  # Échelle pour visualisation

        # Enregistrer le masque dans un objet en mémoire
        buffer = io.BytesIO()
        mask_image.save(buffer, format = "PNG")
        buffer.seek(0)

        # Retourner le fichier en réponse (en mémoire)
        return StreamingResponse(buffer, media_type = "image/png",
                                 headers = {"Content-Disposition": "attachment; filename=predicted_mask.png"})

    except Exception as e:
        return {"error": str(e)}





