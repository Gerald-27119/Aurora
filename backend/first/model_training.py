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
MODEL_PATH = "./car_classifier.pth"

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset
logger.info("Loading dataset...")
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
logger.info(f"Dataset loaded with {len(dataset)} images.")
logger.info(f"Classes: {dataset.classes}")

# Number of classes
num_classes = len(dataset.classes)

# Split into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
)

logger.info(f"Training set size: {len(train_dataset)}")
logger.info(f"Validation set size: {len(val_dataset)}")

# Load datasets into DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Load pre-trained ResNet50 model
logger.info("Loading model...")
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Multi-class output
model.to(device)
logger.info("Model loaded successfully.")
logger.info(f"Model architecture:\n{model}")

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(epochs=10):
    logger.info("Starting training...")
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        logger.info(f"Epoch {epoch + 1}/{epochs} started.")

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

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

def validate_model():
    logger.info("Starting validation...")
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
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
    train_model(epochs=10)
    validate_model()
    logger.info("Script finished.")
