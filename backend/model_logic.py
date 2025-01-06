import torch
from PIL import Image
from torchvision import transforms
from transformers import EfficientNetForImageClassification
import torch.nn as nn

MODEL_PATH = "efficientnet_tank_classifier.pth"

def load_model():
    model = EfficientNetForImageClassification.from_pretrained("google/efficientnet-b0")
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.classifier.in_features, 2)
    )
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def predict_with_threshold(model, image_tensor, threshold=0.7):
    with torch.no_grad():
        logits = model(image_tensor).logits
        probs = torch.softmax(logits, dim=-1)
        confidence, predicted_class = torch.max(probs, dim=-1)
    return "Tank" if confidence.item() >= threshold and predicted_class.item() == 1 else "Non-Tank"
