import os
import hydra
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@hydra.main(config_path="./configs", config_name="config", version_base=None)
def convert_model(cfg):
    """
    Convert ONNX model to OpenVINO IR format
    OpenVINO uses Intermediate Representation (IR) format with .xml and .bin files
    """
    try:
        import openvino as ov
    except ImportError:
        logger.error("OpenVINO is not installed. Install it with: pip install openvino")
        return

    root_dir = hydra.utils.get_original_cwd()
    onnx_model_path = f"{root_dir}/models/model.onnx"
    openvino_model_dir = f"{root_dir}/models/openvino"

    # Check if ONNX model exists
    if not os.path.exists(onnx_model_path):
        logger.error(f"ONNX model not found at: {onnx_model_path}")
        logger.info("Please run convert_model_to_onnx.py first")
        return

    # Create output directory
    os.makedirs(openvino_model_dir, exist_ok=True)

    logger.info(f"Loading ONNX model from: {onnx_model_path}")
    logger.info("Converting to OpenVINO IR format...")

    # Create OpenVINO Core
    core = ov.Core()

    # Read and convert ONNX model
    model = core.read_model(model=onnx_model_path)

    # Get model information
    logger.info("Model input information:")
    for input_layer in model.inputs:
        logger.info(f"  - Input name: {input_layer.get_any_name()}")
        logger.info(f"    Shape: {input_layer.partial_shape}")
        logger.info(f"    Type: {input_layer.element_type}")

    logger.info("Model output information:")
    for output_layer in model.outputs:
        logger.info(f"  - Output name: {output_layer.get_any_name()}")
        logger.info(f"    Shape: {output_layer.partial_shape}")
        logger.info(f"    Type: {output_layer.element_type}")

    # Serialize the model to IR format
    output_model_path = os.path.join(openvino_model_dir, "model.xml")
    logger.info(f"Saving OpenVINO IR model to: {output_model_path}")

    ov.save_model(model, output_model_path)

    logger.info("Model converted to OpenVINO successfully!")
    logger.info(f"OpenVINO IR files saved at:")
    logger.info(f"  - {output_model_path} (model structure)")
    logger.info(f"  - {output_model_path.replace('.xml', '.bin')} (weights)")

    # Available devices info
    logger.info("\nAvailable devices for inference:")
    available_devices = core.available_devices
    for device in available_devices:
        logger.info(f"  - {device}")


if __name__ == "__main__":
    convert_model()
