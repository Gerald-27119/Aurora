# Aurora Project

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
