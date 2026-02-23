"""
Triton Python backend — Postprocessor
Input : logits (TYPE_FP32, [batch, 2])
Output: label  (TYPE_STRING, [batch, 1])  — predicted class name
        score  (TYPE_FP32,   [batch, 1])  — confidence score
"""
import numpy as np
import triton_python_backend_utils as pb_utils
from scipy.special import softmax

LABELS = ["unacceptable", "acceptable"]


class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []

        for request in requests:
            logits = pb_utils.get_input_tensor_by_name(request, "logits").as_numpy()
            # logits shape: [batch, 2]

            probs = softmax(logits, axis=1)                       # [batch, 2]
            predicted_ids = np.argmax(probs, axis=1)              # [batch]
            scores = probs[np.arange(len(predicted_ids)), predicted_ids]  # [batch]

            labels = np.array(
                [[LABELS[i]] for i in predicted_ids], dtype=object
            )                                                      # [batch, 1]
            scores = scores.reshape(-1, 1).astype(np.float32)    # [batch, 1]

            out_label = pb_utils.Tensor("label", labels)
            out_score = pb_utils.Tensor("score", scores)

            responses.append(
                pb_utils.InferenceResponse(output_tensors=[out_label, out_score])
            )

        return responses

    def finalize(self):
        pass
