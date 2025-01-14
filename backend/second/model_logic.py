import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import torch.nn.functional as F


def get_class_names():
    training_dir = os.path.join(os.path.dirname(__file__), "../datasets/car_data/train")
    return sorted([d for d in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, d))])


def load_model(num_classes, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Initialize the model without pretrained weights
    base_model = models.efficientnet_b0(weights=None)
    base_model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(base_model.classifier[1].in_features, num_classes)
    )

    # Load custom weights if provided
    if model_path and os.path.exists(model_path):
        base_model.load_state_dict(torch.load(model_path, map_location=device))

    base_model.eval()
    base_model = base_model.to(device)
    return base_model


def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet mean and std
    ])
    image = Image.open(img_path).convert("RGB")
    img_tensor = transform(image)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor


def predict_car_model(img_path):
    class_names = get_class_names()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_model(num_classes=len(class_names), device=device)
    img_tensor = preprocess_image(img_path).to(device)

    with torch.no_grad():
        predictions = model(img_tensor)  # Logits
        probabilities = F.softmax(predictions, dim=1)  # Convert logits to probabilities
        confidence, predicted_idx = torch.max(probabilities, 1)  # Confidence and class index
        predicted_class = class_names[predicted_idx.item()]
        confidence_percentage = confidence.item() * 100

    return {
        "result": predicted_class,
        "confidence": f"{confidence_percentage:.2f}%"
    }
