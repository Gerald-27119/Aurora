import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import os


class CarDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.class_names = sorted(os.listdir(directory))
        self.class_indices = {name: i for i, name in enumerate(self.class_names)}

        self.image_paths = []
        self.labels = []
        for class_name in self.class_names:
            class_dir = os.path.join(directory, class_name)
            for img_name in os.listdir(class_dir):
                self.image_paths.append(os.path.join(class_dir, img_name))
                self.labels.append(self.class_indices[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet mean and std
])

def create_dataloader(directory, batch_size=32):
    dataset = CarDataset(directory, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, dataset.class_names


def train_model():
    train_dir = "dataset/car_data/train"
    test_dir = "dataset/car_data/test"

    train_loader, train_class_names = create_dataloader(train_dir)
    val_loader, _ = create_dataloader(test_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_model = models.efficientnet_b0(pretrained=True)
    base_model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(base_model.classifier[1].in_features, len(train_class_names))
    )
    base_model = base_model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(base_model.parameters(), lr=0.001)

    epochs = 10
    early_stopping_patience = 5
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        base_model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = base_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        print(f"Training loss: {train_loss / len(train_loader):.4f}, Training accuracy: {train_acc:.2f}%")

        base_model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = base_model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_acc = 100. * correct / total
        val_loss /= len(val_loader)
        print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(base_model.state_dict(), "models/efficientnetb0_best_weights.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    torch.save(base_model.state_dict(), "models/efficientnetb0_final_weights.pth")
    print("Training completed.")


if __name__ == "__main__":
    train_model()
