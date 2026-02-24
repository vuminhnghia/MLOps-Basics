FROM nvcr.io/nvidia/tritonserver:25.01-py3

# Install Python backend dependencies
# transformers: for tokenizer in cola_tokenizer
# scipy: for softmax in cola_postprocess
RUN pip install --no-cache-dir \
    transformers>=4.40.0 \
    scipy>=1.11.0

# Pre-download tokenizer weights vào image layer
RUN python3 -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/bert_uncased_L-2_H-128_A-2')"

# Copy entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
