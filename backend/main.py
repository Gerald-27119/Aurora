import os
from fastapi import FastAPI, UploadFile, File
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from first.model_logic import load_model as load_resnet_model, predict_image as predict_resnet
from third.model_logic import load_model as load_mobile_model, predict_image as predict_image_mobile
from second.model_logic import predict_car_model as predict_image_efficient

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

resnet_model = load_resnet_model()
mobilenet_model = load_mobile_model()
# efficientNetB0 sam sobie wczyta model podczas predykcji

@app.get("/")
async def root():
    return {"message": "Hello World From FastAPI"}
metrics_paths = {
    "resnet50": "first/model_metrics.txt",
    "efficientnetb0": "second/model_metrics.txt",
    "mobilenetv2": "third/model_metrics.txt"
}

def load_metrics(file_path):
    if not os.path.exists(file_path):
        return {"error": f"Metrics file {file_path} not found."}

    metrics = {}
    with open(file_path, "r") as file:
        for line in file:
            key, value = line.strip().split(":")
            metrics[key.strip()] = float(value.strip())
    return metrics

@app.get("/metrics")
async def metrics():
    all_metrics = {}
    for model_name, path in metrics_paths.items():
        all_metrics[model_name] = load_metrics(path)
    return all_metrics
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        if not image.content_type.startswith("image/"):
            return JSONResponse({"error": "Plik musi być obrazem."}, status_code=400)

        temp_file_path = f"./temp_{image.filename}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(await image.read())

        prediction_resnet = predict_resnet(temp_file_path, resnet_model)

        prediction_mobilenet = predict_image_mobile(temp_file_path, mobilenet_model)

        prediction_efficient = predict_image_efficient(temp_file_path)

        os.remove(temp_file_path)

        combined_prediction = {
            "resnet50": prediction_resnet,
            "mobilenetv2": prediction_mobilenet,
            "efficientnetb0": prediction_efficient
        }

        return JSONResponse(combined_prediction)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
