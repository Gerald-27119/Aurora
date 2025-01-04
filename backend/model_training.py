from torchvision import datasets, transforms
import torch
from transformers import EfficientNetForImageClassification
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder("dataset", transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])


model = EfficientNetForImageClassification.from_pretrained("google/efficientnet-b0")
model.classifier = torch.nn.Linear(model.classifier.in_features, 2)  # 2 classes: tank, non-tank



train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

optimizer = Adam(model.parameters(), lr=1e-4)

for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images).logits
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images).logits
            val_loss += F.cross_entropy(outputs, labels).item()
    print(f"Epoch {epoch + 1}, Validation Loss: {val_loss / len(val_loader)}")

torch.save(model.state_dict(), "efficientnet_tank_classifier.pth")
