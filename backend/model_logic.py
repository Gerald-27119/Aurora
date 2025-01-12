import tensorflow as tf
import numpy as np
from tensorflow.io import read_file
from tensorflow.image import decode_jpeg, resize
import os


def get_class_names():
    training_dir = "dataset/car_data/train"
    return sorted([d for d in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, d))])


def load_model(num_classes, input_shape=(224, 224, 3)):
    model_path = "models/efficientnetb0_best_weights"
    base_model = tf.keras.applications.EfficientNetB0(include_top=False, input_shape=input_shape)
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.load_weights(model_path)
    return model


def preprocess_image(img_path):
    img = read_file(img_path)
    img = decode_jpeg(img, channels=3)
    img = resize(img, [224, 224]) / 255.0
    img_array = tf.expand_dims(img, axis=0)
    return img_array


def predict_car_model(img_path):
    class_names = get_class_names()

    model = load_model(num_classes=len(class_names))

    img_array = preprocess_image(img_path)

    predictions = model(img_array, training=False)
    predicted_class = class_names[np.argmax(predictions.numpy())]

    return predicted_class
