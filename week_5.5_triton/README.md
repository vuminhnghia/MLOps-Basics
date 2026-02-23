# Week 5.5 — Triton Inference Server

Deploy BERT text classification (CoLA) với NVIDIA Triton, sử dụng Model Ensemble pipeline:

```
text (string)
  └─► cola_tokenizer  (Python backend)   → input_ids, attention_mask
        └─► cola_trt  (ORT + TensorRT EP) → logits
              └─► cola_postprocess (Python backend) → label, score
```

---

## Concepts học được

| Concept | File |
|---------|------|
| Python backend (pre/post process) | `model_repository/cola_tokenizer/1/model.py` |
| ONNX Runtime + TensorRT EP | `model_repository/cola_trt/config.pbtxt` |
| Dynamic batching | `preferred_batch_size` trong config.pbtxt |
| Model Ensemble (pipeline) | `model_repository/cola_pipeline/config.pbtxt` |
| HTTP client | `client/http_client.py` |
| Prometheus metrics | `localhost:8002/metrics` |
| Perf Analyzer | Xem bên dưới |

---

## Chuẩn bị

### 1. Copy model vào model repository

```bash
bash prepare_model.sh
```

Script này copy `model.onnx` từ `../week_5_docker/models/` vào `model_repository/cola_trt/1/`.

> **Lưu ý:** Triton sẽ dùng ONNX Runtime backend với TensorRT Execution Provider.
> Engine `.trt` sẽ được tự động build và cache lần đầu khi model load.

### 2. Kiểm tra cấu trúc

```
model_repository/
├── cola_tokenizer/
│   ├── config.pbtxt
│   └── 1/model.py
├── cola_trt/
│   ├── config.pbtxt
│   └── 1/model.onnx          ← sau khi chạy prepare_model.sh
├── cola_postprocess/
│   ├── config.pbtxt
│   └── 1/model.py
└── cola_pipeline/
    ├── config.pbtxt
    └── 1/                    ← empty (ensemble không cần model file)
```

---

## Chạy Triton Server

```bash
docker compose up
```

Chờ đến khi log hiện:
```
I tritonserver.cc ... Started GRPCInferenceService at 0.0.0.0:8001
I tritonserver.cc ... Started HTTPService at 0.0.0.0:8000
I tritonserver.cc ... Started Metrics Service at 0.0.0.0:8002
```

### Kiểm tra server

```bash
# Health check
curl localhost:8000/v2/health/ready

# Xem tất cả models đã load
curl localhost:8000/v2/models | python3 -m json.tool

# Xem metadata của pipeline
curl localhost:8000/v2/models/cola_pipeline | python3 -m json.tool
```

---

## Chạy Client

```bash
pip install tritonclient[http]

# Dùng sample sentences mặc định
python client/http_client.py

# Dùng câu tùy chọn
python client/http_client.py "The dog is eating" "He go to school"
```

Output mẫu:
```
==> Sending 5 request(s) to cola_pipeline

  [  acceptable]  0.9123  |  The dog is eating
  [unacceptable]  0.8741  |  The dog am eat
  [unacceptable]  0.7653  |  She doesn't know nothing about it
  [  acceptable]  0.9456  |  I have been to Paris
  [unacceptable]  0.8234  |  He go to school every day
```

---

## Xem Metrics (Prometheus)

```bash
curl localhost:8002/metrics | grep nv_inference
```

Các metrics quan trọng:
- `nv_inference_request_success` — số requests thành công
- `nv_inference_queue_duration_us` — thời gian chờ trong queue (dynamic batching)
- `nv_inference_compute_infer_duration_us` — thời gian inference thực tế

---

## Perf Analyzer

`perf_analyzer` là tool benchmark của NVIDIA, có sẵn trong Triton container.

```bash
# Exec vào container
docker exec -it triton_server bash

# Benchmark cola_pipeline với concurrency 1→8
perf_analyzer \
  -m cola_pipeline \
  --input-data zero \
  --shape text:1,1 \
  --string-length 20 \
  --concurrency-range 1:8:1 \
  --measurement-interval 5000 \
  -u localhost:8000

# Output sẽ hiển thị:
# - Throughput (infer/sec)
# - Latency p50, p90, p99
# - So sánh giữa các mức concurrency
```

---

## Dynamic Batching — Cách hoạt động

Với config `preferred_batch_size: [8, 16, 32, 64]` và `max_queue_delay_microseconds: 100`:

1. Khi có request, Triton giữ trong queue tối đa 100µs
2. Nếu gom đủ preferred batch size → xử lý ngay
3. Nếu hết 100µs chưa đủ → xử lý batch hiện tại

Để thấy rõ dynamic batching, chạy nhiều request đồng thời:
```bash
# Mở nhiều terminal, chạy song song
for i in {1..20}; do
  python client/http_client.py "The cat is sleeping" &
done
wait
```

Sau đó check metrics để thấy batch size thực tế được dùng.
