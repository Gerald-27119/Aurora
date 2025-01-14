import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model_logic import load_model, predict_image

class CalculateMetrics:
    def __init__(self, test_images_path, classes_path, model_path, output_file="model_metrics.txt"):
        self.test_images_path = test_images_path
        self.model_path = model_path
        self.output_file = output_file

        # Wczytanie klas z pliku JSON
        if not os.path.exists(classes_path):
            raise FileNotFoundError(f"Plik klas {classes_path} nie został znaleziony.")

        with open(classes_path, "r") as f:
            self.class_names = json.load(f)

        # Wczytaj model przy inicjalizacji
        self.model = load_model()

    def evaluate_model(self):
        print("Start evaluating...")

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
                prediction = predict_image(img_path, self.model)
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
    classes_path = "classes.json"
    model_path = "car_classifier_mobilenetv2.pth"

    metrics_calculator = CalculateMetrics(test_images_path, classes_path, model_path)
    metrics_calculator.evaluate_model()
