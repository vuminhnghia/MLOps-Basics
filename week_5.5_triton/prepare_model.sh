#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ONNX_SRC="$SCRIPT_DIR/../week_5_docker/models/model.onnx"
ONNX_DATA_SRC="$SCRIPT_DIR/../week_5_docker/models/model.onnx.data"
TRT_DST="$SCRIPT_DIR/model_repository/cola_trt/1/model.plan"

echo "==> Checking source model..."

if [ ! -f "$ONNX_SRC" ]; then
    echo "ERROR: model.onnx not found at $ONNX_SRC"
    echo "Run convert_model_to_onnx.py in week_5_docker first."
    exit 1
fi

echo "==> Copying ONNX model to cola_trt model repository..."
cp "$ONNX_SRC" "$SCRIPT_DIR/model_repository/cola_trt/1/model.onnx"

if [ -f "$ONNX_DATA_SRC" ]; then
    cp "$ONNX_DATA_SRC" "$SCRIPT_DIR/model_repository/cola_trt/1/model.onnx.data"
    echo "==> Copied model.onnx.data as well."
fi

echo ""
echo "==> Done! Model repository is ready."
echo "    Next: docker compose up"
