import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model_logic import predict_car_model

test_images_path = '../dataset/car_data/test'
training_dir = '../dataset/car_data/train'
output_file = 'model_metrics.txt'

def evaluate_model():
    print("Start...")

    class_names = sorted(os.listdir(training_dir))

    true_labels = []
    predicted_labels = []

    for class_name in class_names:
        print(class_name)
        class_dir = os.path.join(test_images_path, class_name)
        if not os.path.isdir(class_dir):
            continue

        for img_name in os.listdir(class_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(class_dir, img_name)
            predicted_class = predict_car_model(img_path)

            true_labels.append(class_name)
            predicted_labels.append(predicted_class)

    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    with open(output_file, 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")

    print("Evaluation complete. Results saved to", output_file)

if __name__ == '__main__':
    evaluate_model()
