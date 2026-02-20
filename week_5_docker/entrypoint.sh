#!/bin/bash
set -e

# Build TensorRT engine from ONNX if it doesn't exist yet.
# This must happen inside the container to match the container's TRT version.
if [ ! -f "/app/models/model.trt" ]; then
    echo "==> model.trt not found. Building TensorRT engine from model.onnx..."
    python /app/build_trt_engine.py
    echo "==> TensorRT engine built successfully."
else
    echo "==> model.trt already exists, skipping build."
fi

echo "==> Starting uvicorn server..."
exec uvicorn app:app --host 0.0.0.0 --port 8000
