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

DATA_DIR = "../datasets"
MODEL_PATH = "./tank_classifier.pth"

# Ustawienie urządzenia - jeśli dostępna jest GPU, będzie to "cuda"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Używane urządzenie: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

logger.info("Loading dataset...")
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
logger.info(f"Dataset loaded with {len(dataset)} images.")
logger.info(f"Classes: {dataset.classes}")

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
model.fc = nn.Linear(model.fc.in_features, 1)  # Adjust for binary classification
logger.info("Model loaded successfully.")
logger.info(f"Model architecture:\n{model}")

# Przeniesienie modelu na GPU/CPU
model.to(device)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # Binary classification loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(epochs=10):
    logger.info("Starting training...")
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        logger.info(f"Epoch {epoch + 1}/{epochs} started.")

        for i, (inputs, labels) in enumerate(train_loader):
            # Przeniesienie wejść i etykiet na GPU/CPU
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(inputs).unsqueeze(1).squeeze(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 10 == 0:  # Log every 10 batches
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

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            outputs = model(inputs).unsqueeze(1).squeeze(-1)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    logger.info(
        f"Validation completed. Average Loss: {total_loss / len(val_loader):.4f}"
    )

if __name__ == "__main__":
    logger.info("Script started.")
    train_model(epochs=1)
    validate_model()
    logger.info("Script finished.")
