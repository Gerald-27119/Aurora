import os
import json
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "car_classifier_mobilenetv2.pth")
CLASSES_PATH = os.path.join(BASE_DIR, "classes.json")

# Ładowanie klas z pliku JSON
if not os.path.exists(CLASSES_PATH):
    raise FileNotFoundError(f"Plik klas {CLASSES_PATH} nie został znaleziony.")

with open(CLASSES_PATH, "r") as f:
    CLASS_NAMES = json.load(f)

def load_model():
    num_classes = len(CLASS_NAMES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Dostosowanie ostatniej warstwy klasyfikatora do liczby klas
    num_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_features, num_classes)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Plik modelu {MODEL_PATH} nie został znaleziony.")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_image(image_path, model):
    device = next(model.parameters()).device
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Nie można otworzyć obrazu: {e}")

    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        confidence_percentage = confidence.item() * 100

    return {"result": predicted_class, "confidence": f"{confidence_percentage:.2f}%"}
