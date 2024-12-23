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

DATA_DIR = "../datasets/tanks"
MODEL_PATH = "./tank_classifier.pth"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

logger.info("Loading dataset...")
dataset = datasets.ImageFolder(root="../datasets", transform=transform)
logger.info(f"Dataset loaded with {len(dataset)} images.")

# Split into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

logger.info(f"Training set size: {len(train_dataset)}")
logger.info(f"Validation set size: {len(val_dataset)}")

# Load datasets into DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Load pre-trained ResNet50 model
logger.info("Loading model...")
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 1)
logger.info("Model loaded successfully.")
logger.info(f"Model architecture:\n{model}")

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(epochs=10):
    logger.info("Starting training...")
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        logger.info(f"Epoch {epoch + 1}/{epochs} started.")
        for i, (inputs, _) in enumerate(train_loader):
            labels = torch.ones(inputs.size(0), 1)  # All labels are "1" for tanks
            optimizer.zero_grad()
            outputs = model(inputs).unsqueeze(1).squeeze(-1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 10 == 0:  # Log every 10 batches
                logger.info(f"Batch {i}/{len(train_loader)} processed. Current batch loss: {loss.item():.4f}")

        logger.info(f"Epoch {epoch + 1}/{epochs} completed. Average Loss: {running_loss / len(train_loader):.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")

def validate_model():
    logger.info("Starting validation...")
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, _ in val_loader:
            labels = torch.ones(inputs.size(0), 1)
            outputs = model(inputs).unsqueeze(1).squeeze(-1)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    logger.info(f"Validation completed. Average Loss: {total_loss / len(val_loader):.4f}")

if __name__ == "__main__":
    logger.info("Script started.")
    train_model(epochs=1)
    validate_model()
    logger.info("Script finished.")
