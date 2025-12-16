import numpy as np
import openvino as ov
from scipy.special import softmax

from data import DataModule
from utils import timing


class ColaOpenVINOPredictor:
    def __init__(self, model_path, device="CPU"):
        """
        Initialize OpenVINO predictor

        Args:
            model_path: Path to OpenVINO IR model (.xml file)
            device: Device to run inference on. Options:
                    - "CPU": CPU execution
                    - "GPU": GPU execution (requires GPU plugin)
                    - "AUTO": Automatically select best device
        """
        self.model_path = model_path
        self.device = device
        self.processor = DataModule()
        self.labels = ["unacceptable", "acceptable"]

        # Create OpenVINO Core and load model
        self.core = ov.Core()
        self.model = self.core.read_model(model=model_path)

        # Compile model for specific device
        self.compiled_model = self.core.compile_model(self.model, device)

        # Get input and output layers
        self.input_layer_ids = self.compiled_model.input(0)
        self.input_layer_mask = self.compiled_model.input(1)
        self.output_layer = self.compiled_model.output(0)

        print(f"Model loaded on device: {device}")
        print(f"Input shapes: {self.input_layer_ids.partial_shape}, {self.input_layer_mask.partial_shape}")
        print(f"Output shape: {self.output_layer.partial_shape}")

    @timing
    def predict(self, text):
        inference_sample = {"sentence": text}
        processed = self.processor.tokenize_data(inference_sample)

        # Prepare input data
        input_ids = np.array([processed["input_ids"]], dtype=np.int64)
        attention_mask = np.array([processed["attention_mask"]], dtype=np.int64)

        # Run inference
        results = self.compiled_model([input_ids, attention_mask])

        # Get output
        output = results[self.output_layer]

        # Apply softmax
        scores = softmax(output)[0]
        predictions = []
        for score, label in zip(scores, self.labels):
            predictions.append({"label": label, "score": float(score)})

        return predictions


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenVINO Inference")
    parser.add_argument(
        "--device",
        type=str,
        default="CPU",
        choices=["CPU", "GPU", "AUTO"],
        help="Device to run inference on"
    )
    args = parser.parse_args()

    sentence = "The boy is sitting on a bench"
    model_path = "./models/openvino/model.xml"

    print(f"\n=== OpenVINO Inference on {args.device} ===\n")
    predictor = ColaOpenVINOPredictor(model_path, device=args.device)
    print(predictor.predict(sentence))

    print("\nRunning 10 inference iterations for timing comparison...")
    sentences = ["The boy is sitting on a bench"] * 10
    for sentence in sentences:
        predictor.predict(sentence)
