import torch
from PIL import Image
from torchvision import transforms
from transformers import EfficientNetForImageClassification

MODEL_PATH = "efficientnet_tank_classifier.pth"

def load_model():
    model = EfficientNetForImageClassification.from_pretrained("google/efficientnet-b0")
    model.classifier = torch.nn.Linear(model.classifier.in_features, 2)  # 2 classes: tank, non-tank
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
    return transform(image).unsqueeze(0)  # Add batch dimension

def predict(model, image_tensor):
    with torch.no_grad():
        logits = model(image_tensor).logits
        predicted_class = logits.argmax(-1).item()
    return "Tank" if predicted_class == 1 else "Non-Tank"
