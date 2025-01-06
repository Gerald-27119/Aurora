from torchvision import datasets, transforms
import torch
from transformers import EfficientNetForImageClassification
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        ce_loss = self.ce_loss(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss

def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder("datasets", transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    model = EfficientNetForImageClassification.from_pretrained("google/efficientnet-b0")
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.classifier.in_features, 2)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn = FocalLoss(alpha=1, gamma=2)

    print("Started training...")

    for epoch in range(5):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images).logits
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images).logits
                val_loss += F.cross_entropy(outputs, labels).item()
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss / len(val_loader)}")

    print("Training finished")

    torch.save(model.state_dict(), "efficientnet_tank_classifier.pth")
    print("Model saved")

if __name__ == "__main__":
    main()
