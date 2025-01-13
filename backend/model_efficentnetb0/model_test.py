import os
from model_logic import predict_car_model

test_images_path = '../test_imgs'

def test_model_on_images(test_path):
    for img_name in os.listdir(test_path):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(test_path, img_name)
        predicted_class = predict_car_model(img_path)

        print(f"Image: {img_name} -> Predicted Class: {predicted_class}")

if __name__ == '__main__':
    test_model_on_images(test_images_path)
