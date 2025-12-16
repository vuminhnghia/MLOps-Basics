import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from scipy.special import softmax

from data import DataModule
from utils import timing


class ColatensorRTPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.processor = DataModule()
        self.labels = ["unacceptable", "acceptable"]

        # Load TensorRT engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(model_path, "rb") as f:
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.inputs, self.outputs, self.tensor_addresses, self.stream = self._allocate_buffers()

    def _allocate_buffers(self):
        """Allocate device memory for inputs and outputs"""
        inputs = {}
        outputs = {}
        tensor_addresses = {}
        stream = cuda.Stream()

        # Iterate through all I/O tensors
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            tensor_mode = self.engine.get_tensor_mode(tensor_name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))

            # We'll allocate max size buffers for dynamic shapes
            # For this model: max batch=128, seq_len=128
            if tensor_mode == trt.TensorIOMode.INPUT:
                shape = (128, 128)  # max batch size and sequence length
            else:  # OUTPUT
                shape = (128, 2)  # max batch size and 2 classes

            size = int(np.prod(shape))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Store device memory address
            tensor_addresses[tensor_name] = int(device_mem)

            # Append to the appropriate dict
            if tensor_mode == trt.TensorIOMode.INPUT:
                inputs[tensor_name] = {"host": host_mem, "device": device_mem, "dtype": dtype}
            else:
                outputs[tensor_name] = {"host": host_mem, "device": device_mem, "dtype": dtype}

        return inputs, outputs, tensor_addresses, stream

    @timing
    def predict(self, text):
        inference_sample = {"sentence": text}
        processed = self.processor.tokenize_data(inference_sample)

        # Prepare input data
        input_ids = np.array([processed["input_ids"]], dtype=np.int64)
        attention_mask = np.array([processed["attention_mask"]], dtype=np.int64)

        # Set input shapes for dynamic batch
        batch_size = 1
        seq_length = len(processed["input_ids"])

        # Set input shapes (TensorRT 10.x API)
        self.context.set_input_shape("input_ids", (batch_size, seq_length))
        self.context.set_input_shape("attention_mask", (batch_size, seq_length))

        # Copy input data to host buffers
        np.copyto(self.inputs["input_ids"]["host"][:input_ids.size], input_ids.ravel())
        np.copyto(self.inputs["attention_mask"]["host"][:attention_mask.size], attention_mask.ravel())

        # Transfer input data to GPU
        cuda.memcpy_htod_async(
            self.inputs["input_ids"]["device"],
            self.inputs["input_ids"]["host"],
            self.stream
        )
        cuda.memcpy_htod_async(
            self.inputs["attention_mask"]["device"],
            self.inputs["attention_mask"]["host"],
            self.stream
        )

        # Set tensor addresses for execution
        for name, address in self.tensor_addresses.items():
            self.context.set_tensor_address(name, address)

        # Run inference (TensorRT 10.x API)
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Transfer predictions back to host
        cuda.memcpy_dtoh_async(
            self.outputs["output"]["host"],
            self.outputs["output"]["device"],
            self.stream
        )

        # Synchronize the stream
        self.stream.synchronize()

        # Get output
        output_size = batch_size * 2  # 2 classes
        output = self.outputs["output"]["host"][:output_size].reshape(batch_size, 2)

        # Apply softmax
        scores = softmax(output)[0]
        predictions = []
        for score, label in zip(scores, self.labels):
            predictions.append({"label": label, "score": float(score)})

        return predictions

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'stream'):
            del self.stream
        if hasattr(self, 'context'):
            del self.context
        if hasattr(self, 'engine'):
            del self.engine


if __name__ == "__main__":
    sentence = "The boy is sitting on a bench"
    predictor = ColatensorRTPredictor("./models/model.trt")
    print(predictor.predict(sentence))
    print("\nRunning 10 inference iterations for timing comparison...")
    sentences = ["The boy is sitting on a bench"] * 10
    for sentence in sentences:
        predictor.predict(sentence)
