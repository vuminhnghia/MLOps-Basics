"""
Standalone TensorRT engine builder.
Converts model.onnx -> model.trt using the container's own TensorRT version.
Run this inside the container to avoid version mismatch errors.
"""
import sys
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

ONNX_PATH = "/app/models/model.onnx"
TRT_PATH = "/app/models/model.trt"

# Must match configs/processing/default.yaml
OPT_BATCH = 64
MAX_BATCH = 128
MAX_LENGTH = 128


def build_engine():
    print(f"[TRT] TensorRT version: {trt.__version__}")
    print(f"[TRT] Loading ONNX model from: {ONNX_PATH}")

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    if not parser.parse_from_file(ONNX_PATH):
        print("[TRT] ERROR: Failed to parse ONNX model", file=sys.stderr)
        for i in range(parser.num_errors):
            print(f"  {parser.get_error(i)}", file=sys.stderr)
        sys.exit(1)

    print("[TRT] ONNX model parsed successfully")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)

    if builder.platform_has_fast_fp16:
        print("[TRT] FP16 enabled")
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    profile.set_shape("input_ids",     (1, MAX_LENGTH), (OPT_BATCH, MAX_LENGTH), (MAX_BATCH, MAX_LENGTH))
    profile.set_shape("attention_mask", (1, MAX_LENGTH), (OPT_BATCH, MAX_LENGTH), (MAX_BATCH, MAX_LENGTH))
    config.add_optimization_profile(profile)

    print(f"[TRT] Building engine (opt_batch={OPT_BATCH}, max_batch={MAX_BATCH}, seq_len={MAX_LENGTH})...")
    print("[TRT] This may take several minutes on first run...")

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("[TRT] ERROR: Failed to build TensorRT engine", file=sys.stderr)
        sys.exit(1)

    with open(TRT_PATH, "wb") as f:
        f.write(serialized_engine)

    print(f"[TRT] Engine saved to: {TRT_PATH}")


if __name__ == "__main__":
    build_engine()
