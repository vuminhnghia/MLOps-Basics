#!/bin/bash
set -e

# Build TRT engine từ ONNX nếu chưa có.
# Phải chạy bên trong container để đảm bảo TRT version khớp.
if [ ! -f "/models/cola_trt/1/model.plan" ]; then
    echo "==> model.plan not found. Building TensorRT engine from model.onnx..."
    /usr/src/tensorrt/bin/trtexec \
        --onnx=/models/cola_trt/1/model.onnx \
        --saveEngine=/models/cola_trt/1/model.plan \
        --minShapes=input_ids:1x128,attention_mask:1x128 \
        --optShapes=input_ids:64x128,attention_mask:64x128 \
        --maxShapes=input_ids:128x128,attention_mask:128x128 \
        --fp16
    echo "==> TensorRT engine built successfully."
else
    echo "==> model.plan already exists, skipping build."
fi

echo "==> Starting Triton Inference Server..."
exec tritonserver --model-repository=/models --log-verbose=1
