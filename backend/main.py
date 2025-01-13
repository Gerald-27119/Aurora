import os
from fastapi import FastAPI, UploadFile, File
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from first.model_logic import load_model, predict_image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Dostosuj do swoich potrzeb
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ładowanie modelu podczas startu aplikacji
model = load_model()

@app.get("/")
async def root():
    return {"message": "Hello World From FastAPI"}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        # Sprawdzenie czy przesłany plik jest obrazem
        if not image.content_type.startswith("image/"):
            return JSONResponse({"error": "Plik musi być obrazem."}, status_code=400)

        # Ścieżka do tymczasowego pliku
        temp_file_path = f"./temp_{image.filename}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(await image.read())

        # Predykcja
        prediction = predict_image(temp_file_path, model)

        # Usunięcie tymczasowego pliku
        os.remove(temp_file_path)

        return JSONResponse(prediction)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
