import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model_logic import load_model, predict_image  # Załóżmy, że te funkcje są zaimportowane

class CalculateMetrics:
    def __init__(self, test_images_path, training_dir, model_path, output_file="model_metrics.txt"):
        self.test_images_path = test_images_path
        self.training_dir = training_dir
        self.model_path = model_path
        self.output_file = output_file
        self.model = load_model(model_path=model_path)  # Wczytaj model przy inicjalizacji
        self.class_names = sorted(os.listdir(self.training_dir))

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
                prediction = predict_image(img_path, self.model)  # Użycie funkcji predict_image
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
    model_path = "car_classifier.pth"

    metrics_calculator = CalculateMetrics(test_images_path, training_dir, model_path)
    metrics_calculator.evaluate_model()
