# Aurora Project

## Project Contributors

The project was developed by **Adam Langmesser (s27119)** and **Stanisław Oziemczuk (s26982)**:

- **Adam Langmesser** worked on the implementation of the **ResNet50** and **MobileNetV2** models, including the calculation of their performance metrics (accuracy, recall, precision, F1 score, and others). He also took care of version control and set up the backend endpoints for communication with the frontend.
- **Stanisław Oziemczuk** focused on the **EfficientNetB0** model and its statistics. He also developed the **frontend panel**, allowing users to upload images via a web interface and view the performance statistics for each model. Additionally, Stanisław identified and prepared the appropriate **dataset** for the project.

## Overview

The **Aurora Project** is designed to recognize the model of a car from a dataset containing hundreds of vehicle images. The backend processes images using three different pre-trained deep learning models to classify the car models, and the results are compared based on metrics such as accuracy, precision, recall, and F1 score. Each model also returns a **confidence score**, which is the probability (in percentage) that the model assigns to its predicted class.

## Models Used

The project leverages the following models for car model classification:

1. **ResNet50**  
   ResNet50 (Residual Neural Network) is a powerful deep convolutional neural network known for its ability to handle the vanishing gradient problem in deep architectures. Its residual blocks allow for efficient and accurate training of very deep networks.  
   **Reason for use**: ResNet50 is a well-established model for image classification tasks and provides a strong baseline for comparing performance.  
   **Performance**:  
   - Accuracy: **0.9211**  
   - Precision: **0.8976**  
   - Recall: **0.9012**  
   - F1 Score: **0.9212**  

2. **EfficientNetB0**  
   EfficientNetB0 is part of the EfficientNet family, which uses a compound scaling method to balance the depth, width, and resolution of the model for optimal performance. It achieves high accuracy with fewer parameters and less computational cost compared to other models.  
   **Reason for use**: EfficientNetB0 offers a balance between computational efficiency and accuracy, making it ideal for resource-constrained environments.  
   **Performance**:  
   - Accuracy: **0.4234**  
   - Precision: **0.5623**  
   - Recall: **0.5544**  
   - F1 Score: **0.4311**  

3. **MobileNetV2**  
   MobileNetV2 is a lightweight model designed for mobile and embedded applications. It uses depthwise separable convolutions to reduce computational cost while maintaining accuracy.  
   **Reason for use**: MobileNetV2 is suitable for tasks requiring fast inference with limited hardware resources, such as real-time car model recognition.  
   **Performance**:  
   - Accuracy: **0.7831**  
   - Precision: **0.8032**  
   - Recall: **0.7832**  
   - F1 Score: **0.7754**  

---

## Summary and Insights

The models showed varying levels of performance on the car model recognition task:

- **ResNet50** significantly outperformed the other models, achieving the highest accuracy, precision, recall, and F1 score. This is likely due to its deep architecture and residual connections, which help in learning complex features and achieving robust predictions.
- **EfficientNetB0**, despite its computational efficiency, struggled with this task, achieving the lowest performance across all metrics. This could be due to insufficient epochs for full convergence or the model's inherent bias towards lightweight computations at the expense of fine-grained accuracy.
- **MobileNetV2** performed moderately well, with metrics indicating a balance between computational efficiency and predictive performance. It is a good choice for scenarios requiring real-time inference on resource-constrained devices.

### Conclusion

ResNet50 is the best-performing model for this task, making it suitable for applications where high accuracy and reliability are essential. MobileNetV2 offers a practical trade-off between accuracy and speed, while EfficientNetB0 may require further fine-tuning or additional training to improve its results. These findings highlight the importance of selecting the right model architecture based on the specific requirements of the task and computational resources.

---

## Dataset

Our dataset is **Stanford Car Dataset by classes folder** [[link](https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder)]. It contains images of **196 car models**, with a total of **16,185 images**. The data is split into **8,144 training images** and **8,041 testing images**.

---

## Requirements

- Python 3.11
- Node.js (for frontend)

---

## Backend Setup

The backend is built using FastAPI. Follow these steps to set up and run the backend:

1. **Clone the repository:**

    ```sh
    git clone <repository-url>
    cd Aurora/backend
    ```

2. **Create a virtual environment:**

    ```sh
    python -m venv .venv
    ```

3. **Activate the virtual environment:**

    - On Windows:

        ```sh
        .\.venv\Scripts\activate
        ```

    - On macOS/Linux:

        ```sh
        source .venv/bin/activate
        ```

4. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

5. **Run the backend server:**

    ```sh
    uvicorn main:app --reload
    ```

    The backend will be running at `http://localhost:8000`.

---

## Frontend Setup

The frontend is built using Vite. Follow these steps to set up and run the frontend:

1. **Navigate to the frontend directory:**

    ```sh
    cd ../frontend
    ```

2. **Install the required packages:**

    ```sh
    npm install
    ```

3. **Run the frontend server:**

    ```sh
    npm run dev
    ```

    The frontend will be running at `http://localhost:5173`.

---

## Project should be up and running
