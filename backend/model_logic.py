import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('models/efficientnetb0_best.h5')

# Define class names (same as used during training)
class_names = ['Tesla_Model_3', 'BMW_X5', 'Audi_A4']

def preprocess_image(img_path):
    """
    Preprocess the image for EfficientNet model input.
    :param img_path: Path to the image file
    :return: Preprocessed image as a numpy array
    """
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_car_model(img_path):
    """
    Predict the car model from the given image.
    :param img_path: Path to the image file
    :return: Predicted car model name
    """
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    return predicted_class
