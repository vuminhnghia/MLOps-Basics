# Use the official NVIDIA TensorRT base image
FROM nvcr.io/nvidia/tensorrt:25.01-py3

# Install 'uv' for high-performance dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Copy dependency manifest first to leverage Docker cache
COPY pyproject.toml /app/

# --- Environment Configurations ---
# UV_SYSTEM_PYTHON=1: Forces uv to use the system Python
# UV_PYTHON_DOWNLOADS=never: Prevents uv from trying to download another Python version
ENV UV_SYSTEM_PYTHON=1 \
    UV_PYTHON_DOWNLOADS=never \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    PREDICTOR_TYPE=tensorrt

# --- Dependency Installation ---
# We use --break-system-packages to bypass PEP 668 (Externally Managed Environment).
# This is safe and standard practice inside a dedicated Docker container.
RUN uv pip install --break-system-packages --upgrade "typing-extensions>=4.10.0" && \
    uv pip install --break-system-packages -e . && \
    uv pip install --break-system-packages "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu128 && \
    uv pip install --break-system-packages pycuda

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
