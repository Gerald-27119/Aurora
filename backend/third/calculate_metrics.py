import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image
import numpy as np
import torch.nn.functional as F
import json


class CalculateMetrics:
    def __init__(self, test_images_path, training_dir, model_path, classes_path, output_file="model_metrics.txt"):
        self.test_images_path = test_images_path
        self.training_dir = training_dir
        self.model_path = model_path
        self.output_file = output_file

        # Wczytanie klas z pliku JSON
        if not os.path.exists(classes_path):
            raise FileNotFoundError(f"Plik klas {classes_path} nie został znaleziony.")
        with open(classes_path, "r") as f:
            self.class_names = json.load(f)

        self.model = self.load_model()  # Wczytaj model przy inicjalizacji

    def load_model(self):
        """
        Własna funkcja wczytania modelu MobileNetV2.
        """
        num_classes = len(self.class_names)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Inicjalizacja modelu MobileNetV2 z pretrenowanymi wagami
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

        # Dostosowanie ostatniej warstwy klasyfikatora do liczby klas
        num_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_features, num_classes)

        # Wczytanie wag modelu
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Plik modelu {self.model_path} nie został znaleziony.")
        model.load_state_dict(torch.load(self.model_path, map_location=device))

        model.to(device)
        model.eval()
        return model

    def preprocess_image(self, img_path):
        """
        Wstępne przetwarzanie obrazu przed podaniem go do modelu.
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = Image.open(img_path).convert("RGB")
        img_tensor = transform(image)
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor

    def predict_image(self, img_path):
        """
        Wykonuje predykcję na pojedynczym obrazie.
        """
        device = next(self.model.parameters()).device
        img_tensor = self.preprocess_image(img_path).to(device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            predicted_class = self.class_names[predicted_idx.item()]
            confidence_percentage = confidence.item() * 100

        return {"result": predicted_class, "confidence": f"{confidence_percentage:.2f}%"}

    def evaluate_model(self):
        print("Start...")

        true_labels = []
        predicted_labels = []

        for class_name in self.class_names:
            print(f"Processing class: {class_name}")
            class_dir = os.path.join(self.test_images_path, class_name)
            if not os.path.isdir(class_dir):
                continue

            for img_name in os.listdir(class_dir):
                if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                img_path = os.path.join(class_dir, img_name)
                prediction = self.predict_image(img_path)
                true_labels.append(class_name)
                predicted_labels.append(prediction["result"])

        # Oblicz metryki
        self.calculate_and_save_metrics(true_labels, predicted_labels)

    def calculate_and_save_metrics(self, true_labels, predicted_labels):
        true_labels = np.array(true_labels)
        predicted_labels = np.array(predicted_labels)

        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        f1 = f1_score(true_labels, predicted_labels, average='weighted')

        # Zapisz metryki do pliku
        with open(self.output_file, 'w') as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")

        print("Evaluation complete. Results saved to", self.output_file)


# Przykład użycia
if __name__ == "__main__":
    test_images_path = "../datasets/car_data/test"
    training_dir = "../datasets/car_data/train"
    model_path = "car_classifier_mobilenetv2.pth"
    classes_path = "classes.json"

    metrics_calculator = CalculateMetrics(test_images_path, training_dir, model_path, classes_path)
    metrics_calculator.evaluate_model()
