import os
from fastapi import FastAPI

app = FastAPI(title="MLOps Basics App")

predictor_type = os.getenv("PREDICTOR_TYPE", "tensorrt")

if predictor_type == "onnx":
    from inference_onnx import ColaONNXPredictor
    predictor = ColaONNXPredictor("./models/model.onnx")
else:
    from inference_tensorrt import ColatensorRTPredictor
    predictor = ColatensorRTPredictor("./models/model.trt")

@app.get("/")
async def home_page():
    return "<h2>Sample prediction API</h2>"


@app.get("/predict")
async def get_prediction(text: str):
    result =  predictor.predict(text)
    return result