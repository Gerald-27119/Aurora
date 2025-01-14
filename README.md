# Aurora Project

## Overview

The **Aurora Project** is designed to recognize the model of a car from a dataset containing hundreds of vehicle images. The backend processes images using three different pre-trained deep learning models to classify the car models, and the results are compared based on metrics such as accuracy, precision, recall, and F1 score.

## Models Used

The project leverages the following models for car model classification:

1. **ResNet50**  
   ResNet50 (Residual Neural Network) is a powerful deep convolutional neural network known for its ability to handle the vanishing gradient problem in deep architectures. Its residual blocks allow for efficient and accurate training of very deep networks.  
   **Reason for use**: ResNet50 is a well-established model for image classification tasks and provides a strong baseline for comparing performance.

2. **EfficientNetB0**  
   EfficientNetB0 is part of the EfficientNet family, which uses a compound scaling method to balance the depth, width, and resolution of the model for optimal performance. It achieves high accuracy with fewer parameters and less computational cost compared to other models.  
   **Reason for use**: EfficientNetB0 offers a balance between computational efficiency and accuracy, making it ideal for resource-constrained environments.

3. **MobileNetV2**  
   MobileNetV2 is a lightweight model designed for mobile and embedded applications. It uses depthwise separable convolutions to reduce computational cost while maintaining accuracy.  
   **Reason for use**: MobileNetV2 is suitable for tasks requiring fast inference with limited hardware resources, such as real-time car model recognition.

## Why These Models?

The primary goal of the project is to compare the performance of different neural network architectures on the car model recognition task. By using ResNet50, EfficientNetB0, and MobileNetV2, we can analyze their strengths and weaknesses in terms of accuracy, precision, recall, and F1 score. This comparison provides valuable insights into which architecture is best suited for the specific requirements of car model classification.

## Requirements

- Python 3.11
- Node.js (for frontend)

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

    The backend will be running at `http://127.0.0.1:8000`.

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

    The frontend will be running at `http://127.0.0.1:5173`.

## Project should be up and running
