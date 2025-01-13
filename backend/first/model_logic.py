import torch
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = "./first/car_classifier.pth"

from torchvision.models import ResNet50_Weights

def load_model():
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    return model


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_path, model):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor).item()
        confidence = torch.sigmoid(torch.tensor(output)).item()
        label = "Tank" if confidence > 0.5 else "Non-Tank"
    # formatting confidence as precentage
    formatted_confidence = f"{confidence * 100:.2f}%"
    return {"result": label, "confidence": formatted_confidence}
