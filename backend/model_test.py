import os
from model_logic import predict_car_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

test_images_path = 'dataset/car_data/test'

def test_model_on_images(test_path):
    for img_name in os.listdir(test_path):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(test_path, img_name)
        predicted_class = predict_car_model(img_path)

        print(f"Image: {img_name} -> Predicted Class: {predicted_class}")

        img = image.load_img(img_path, target_size=(224, 224))
        plt.imshow(img)
        plt.title(f"Predicted: {predicted_class}")
        plt.axis('off')
        plt.show()

if __name__ == '__main__':
    test_model_on_images(test_images_path)
