# Use the official NVIDIA TensorRT base image
FROM nvcr.io/nvidia/tensorrt:25.01-py3

# Install 'uv' for high-performance dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# --- Environment Configurations ---
# UV_SYSTEM_PYTHON=1: Forces uv to use the system Python
# UV_PYTHON_DOWNLOADS=never: Prevents uv from trying to download another Python version
ENV UV_SYSTEM_PYTHON=1 \
    UV_PYTHON_DOWNLOADS=never \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    PREDICTOR_TYPE=tensorrt

# --- Dependency Installation ---
# Only install what TensorRT inference actually needs.
# TensorRT uses CUDA directly via pycuda.
RUN uv pip install --break-system-packages \
    "transformers>=4.40.0" \
    "numpy>=2.2.0,<2.3.0" \
    "scipy>=1.11.0" \
    "fastapi" \
    "uvicorn" \
    "pycuda"

# Pre-download tokenizer weights into image layer for faster container startup
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/bert_uncased_L-2_H-128_A-2')"

# Copy the rest of the application source code.
# Note: models/*.trt is excluded via .dockerignore to avoid TRT version mismatch.
#       The engine will be built inside the container on first startup via entrypoint.sh.
COPY ./ /app

# Make the entrypoint script executable
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000

# The entrypoint builds the TRT engine (if needed) then starts the server.
# Building inside the container guarantees the engine matches this TRT version.
ENTRYPOINT ["/app/entrypoint.sh"]
