from model_logic import load_model, preprocess_image, predict
import os

if __name__ == "__main__":
    model = load_model()
    test_dir = "test_imgs"
    print("-----START-----\n\n")
    for filename in os.listdir(test_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(test_dir, filename)
                image_tensor = preprocess_image(image_path)
                label = predict(model, image_tensor)
                print(f"{filename}: {label}")
    print("\n\n-----FINISHED-----")