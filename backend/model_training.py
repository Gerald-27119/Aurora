import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.data import Dataset
import os


def preprocess_image(img_path, label, image_size=(224, 224)):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, image_size)
    img = img / 255.0
    return img, label


def create_dataset(directory, image_size=(224, 224), batch_size=32):
    class_names = sorted(os.listdir(directory))
    class_indices = {name: i for i, name in enumerate(class_names)}

    image_paths = []
    labels = []
    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        for img_name in os.listdir(class_dir):
            image_paths.append(os.path.join(class_dir, img_name))
            labels.append(class_indices[class_name])

    dataset = Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(lambda x, y: preprocess_image(x, y, image_size))
    dataset = dataset.batch(batch_size).shuffle(buffer_size=1000)

    return dataset, class_names


def train_model():
    train_dir = "dataset/car_data/train"
    test_dir = "dataset/car_data/test"

    train_ds, train_class_names = create_dataset(train_dir)
    val_ds, _ = create_dataset(test_dir)

    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(len(train_class_names), activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    epochs = 20
    early_stopping_patience = 5
    best_val_loss = float('inf')
    patience_counter = 0
    num_epochs_trained = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        num_epochs_trained += 1

        for step, (x_batch_train, y_batch_train) in enumerate(train_ds):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train, training=True)
                loss_value = loss_fn(y_batch_train, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            train_acc_metric.update_state(y_batch_train, logits)

        train_acc = train_acc_metric.result()
        print(f"Training accuracy: {train_acc:.4f}")
        train_acc_metric.reset_states()

        val_loss = 0.0
        for x_batch_val, y_batch_val in val_ds:
            val_logits = model(x_batch_val, training=False)
            val_loss += loss_fn(y_batch_val, val_logits).numpy()
            val_acc_metric.update_state(y_batch_val, val_logits)

        val_loss /= len(val_ds)
        val_acc = val_acc_metric.result()
        print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_acc:.4f}")
        val_acc_metric.reset_states()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_weights("models/efficientnetb0_best_weights")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    model.save_weights("models/efficientnetb0_final_weights")
    print(f"Training completed. Total epochs trained: {num_epochs_trained}")


if __name__ == "__main__":
    train_model()
