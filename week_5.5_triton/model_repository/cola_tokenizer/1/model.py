"""
Triton Python backend — Tokenizer
Input : text (TYPE_STRING, [batch, 1])
Output: input_ids (TYPE_INT64, [batch, 128])
        attention_mask (TYPE_INT64, [batch, 128])
"""
import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer

MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"
MAX_LENGTH = 128


class TritonPythonModel:
    def initialize(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def execute(self, requests):
        responses = []

        for request in requests:
            # Input shape: [batch, 1], dtype bytes
            raw = pb_utils.get_input_tensor_by_name(request, "text").as_numpy()
            # Decode each item: raw[i][0] is bytes
            texts = [item[0].decode("utf-8") for item in raw]

            encoded = self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=MAX_LENGTH,
                return_tensors="np",
            )

            input_ids = encoded["input_ids"].astype(np.int64)         # [batch, 128]
            attention_mask = encoded["attention_mask"].astype(np.int64)  # [batch, 128]

            out_input_ids = pb_utils.Tensor("input_ids", input_ids)
            out_attention_mask = pb_utils.Tensor("attention_mask", attention_mask)

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[out_input_ids, out_attention_mask]
                )
            )

        return responses

    def finalize(self):
        pass
