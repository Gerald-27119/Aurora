import json
import logging
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet50_Weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR = "../datasets/car_data/train"
# zbiór testowy 4 pojazdów
# DATA_DIR = "../datasets/car_data/small_train"
MODEL_PATH = "./car_classifier.pth"
CLASSES_PATH = "./classes.json"

def train_model(train_loader, model, criterion, optimizer, device, epochs=20):
    logger.info("Starting training...")
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        logger.info(f"Epoch {epoch + 1}/{epochs} started.")

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 10 == 0:
                logger.info(
                    f"Batch {i}/{len(train_loader)} processed. "
                    f"Current batch loss: {loss.item():.4f}"
                )

        logger.info(
            f"Epoch {epoch + 1}/{epochs} completed. "
            f"Average Loss: {running_loss / len(train_loader):.4f}"
        )

    torch.save(model.state_dict(), MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")

def validate_model(val_loader, model, criterion, device):
    logger.info("Starting validation...")
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    logger.info(
        f"Validation completed. Average Loss: {total_loss / len(val_loader):.4f}, "
        f"Accuracy: {accuracy:.2%}"
    )

if __name__ == "__main__":
    logger.info("Script started.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Zmiana rozmiaru zdjęć oraz dostosowanie ich do formatu pytorcha
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    logger.info("Loading dataset...")
    dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    logger.info(f"Dataset loaded with {len(dataset)} images.")
    logger.info(f"Classes: {dataset.classes}")

    # Zapisanie listy klas do pliku JSON
    with open(CLASSES_PATH, "w") as f:
        json.dump(dataset.classes, f)
    logger.info(f"Lista klas zapisana do {CLASSES_PATH}")

    num_classes = len(dataset.classes)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    logger.info(f"Training set size: {len(train_dataset)}")
    logger.info(f"Validation set size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    logger.info("Loading model...")
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)
    logger.info("Model loaded successfully.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(train_loader, model, criterion, optimizer, device, epochs=30)
    validate_model(val_loader, model, criterion, device)

    logger.info("Script finished.")
