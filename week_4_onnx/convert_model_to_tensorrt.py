import os
import hydra
import logging
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # This initializes CUDA context automatically

logger = logging.getLogger(__name__)

# TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def convert_model(cfg):
    root_dir = hydra.utils.get_original_cwd()
    onnx_model_path = f"{root_dir}/models/model.onnx"
    trt_model_path = f"{root_dir}/models/model.trt"

    # Check if ONNX model exists
    if not os.path.exists(onnx_model_path):
        logger.error(f"ONNX model not found at: {onnx_model_path}")
        logger.info("Please run convert_model_to_onnx.py first")
        return

    logger.info(f"Loading ONNX model from: {onnx_model_path}")

    # Create builder and network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX model
    logger.info("Parsing ONNX model...")
    # Use parse_from_file to handle external data correctly
    if not parser.parse_from_file(onnx_model_path):
        logger.error("Failed to parse ONNX model")
        for error in range(parser.num_errors):
            logger.error(parser.get_error(error))
        return

    logger.info("ONNX model parsed successfully")

    # Configure builder
    config = builder.create_builder_config()

    # Set memory pool limit (8GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)

    # Enable FP16 precision if supported
    if builder.platform_has_fast_fp16:
        logger.info("FP16 mode is supported and enabled")
        config.set_flag(trt.BuilderFlag.FP16)

    # Set optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()

    # Define dynamic batch size range
    # min_batch, opt_batch, max_batch
    min_batch = 1
    opt_batch = cfg.processing.batch_size
    max_batch = cfg.processing.batch_size * 2

    max_length = cfg.processing.max_length

    # Set shapes for input_ids
    profile.set_shape(
        "input_ids",
        (min_batch, max_length),  # min
        (opt_batch, max_length),  # opt
        (max_batch, max_length)   # max
    )

    # Set shapes for attention_mask
    profile.set_shape(
        "attention_mask",
        (min_batch, max_length),  # min
        (opt_batch, max_length),  # opt
        (max_batch, max_length)   # max
    )

    config.add_optimization_profile(profile)

    # Build TensorRT engine
    logger.info("Building TensorRT engine... This may take a few minutes")
    logger.info(f"Optimization config:")
    logger.info(f"  - Batch size: min={min_batch}, opt={opt_batch}, max={max_batch}")
    logger.info(f"  - Sequence length: {max_length}")
    logger.info(f"  - FP16: {builder.platform_has_fast_fp16}")

    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        logger.error("Failed to build TensorRT engine")
        return

    # Save engine to file
    logger.info(f"Saving TensorRT engine to: {trt_model_path}")
    with open(trt_model_path, 'wb') as f:
        f.write(serialized_engine)

    logger.info("Model converted to TensorRT successfully!")
    logger.info(f"TensorRT engine saved at: {trt_model_path}")

    # Print engine info
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    logger.info(f"Engine info:")
    logger.info(f"  - Number of I/O tensors: {engine.num_io_tensors}")
    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        tensor_mode = engine.get_tensor_mode(tensor_name)
        logger.info(f"  - Tensor {i}: {tensor_name} (mode: {tensor_mode})")


if __name__ == "__main__":
    convert_model()
