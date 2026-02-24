"""
Triton HTTP client for cola_pipeline ensemble.

Install: pip install tritonclient[http]

Usage:
  python client/http_client.py
  python client/http_client.py "The dog is eating"
"""
import sys
import numpy as np
import tritonclient.http as httpclient


TRITON_URL = "localhost:8100"
MODEL_NAME = "cola_pipeline"

SAMPLE_SENTENCES = [
    "The dog is eating",
    "The dog am eat",
    "She doesn't know nothing about it",
    "I have been to Paris",
    "He go to school every day",
]


def predict(client: httpclient.InferenceServerClient, texts: list[str]) -> list[dict]:
    # Input: [batch, 1] array of bytes
    input_data = np.array([[t.encode("utf-8")] for t in texts], dtype=object)

    inputs = [httpclient.InferInput("text", input_data.shape, "BYTES")]
    inputs[0].set_data_from_numpy(input_data)

    outputs = [
        httpclient.InferRequestedOutput("label"),
        httpclient.InferRequestedOutput("score"),
    ]

    response = client.infer(MODEL_NAME, inputs=inputs, outputs=outputs)

    labels = response.as_numpy("label")   # [batch, 1] bytes
    scores = response.as_numpy("score")   # [batch, 1] float32

    results = []
    for text, label_row, score_row in zip(texts, labels, scores):
        label = label_row[0].decode("utf-8") if isinstance(label_row[0], bytes) else label_row[0]
        results.append({"text": text, "label": label, "score": float(score_row[0])})

    return results


def main():
    client = httpclient.InferenceServerClient(url=TRITON_URL)

    # Check server health
    if not client.is_server_live():
        print("ERROR: Triton server is not live at", TRITON_URL)
        sys.exit(1)
    if not client.is_model_ready(MODEL_NAME):
        print(f"ERROR: Model '{MODEL_NAME}' is not ready")
        sys.exit(1)

    # Use command-line arg or default samples
    texts = sys.argv[1:] if len(sys.argv) > 1 else SAMPLE_SENTENCES

    print(f"==> Sending {len(texts)} request(s) to {MODEL_NAME}\n")
    results = predict(client, texts)

    for r in results:
        print(f"  [{r['label']:>12}]  {r['score']:.4f}  |  {r['text']}")

    # --- Batch throughput demo ---
    if len(texts) == len(SAMPLE_SENTENCES):
        print("\n==> Batch of 64 (repeated samples) for throughput demo...")
        big_batch = (SAMPLE_SENTENCES * 13)[:64]
        batch_results = predict(client, big_batch)
        print(f"    Received {len(batch_results)} predictions")


if __name__ == "__main__":
    main()
