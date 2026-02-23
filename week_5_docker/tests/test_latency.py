"""
Latency test script for TensorRT inference service.
Tests different scenarios to identify the root cause of slowness.
"""
import time
import urllib.request
import urllib.parse
import json

BASE_URL = "http://localhost:8000/predict"

INPUTS = [
    "The boy is sitting on a bench",
    "This sentence it not make sense",
    "The cat sat on the mat",
    "Him go store yesterday for milk",
    "She reads books every evening",
    "The children are playing in the park",
    "Me want eat pizza now please",
    "Scientists discovered a new species of fish",
    "They goes to school every day",
    "The government announced a new policy on climate change",
]


def predict(text: str) -> tuple[dict, float]:
    url = f"{BASE_URL}?{urllib.parse.urlencode({'text': text})}"
    t0 = time.perf_counter()
    with urllib.request.urlopen(url) as resp:
        result = json.loads(resp.read())
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return result, elapsed_ms


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def run(label: str, text: str, delay_before: float = 0.0):
    if delay_before > 0:
        time.sleep(delay_before)
    result, ms = predict(text)
    top = max(result, key=lambda x: x["score"])
    print(f"[{ms:7.2f} ms]  [{top['label']:12s} {top['score']:.3f}]  \"{text[:45]}\"")
    return ms


# ── Scenario 1: Same input repeated immediately ─────────────────────
section("Scenario 1: Same input, no delay (2x)")
for i in range(2):
    run(f"call {i+1}", INPUTS[0])

# ── Scenario 2: Different inputs, no delay ───────────────────────────
section("Scenario 2: Different inputs, NO delay between them")
for text in INPUTS:
    run("", text)

# ── Scenario 3: Different inputs with 3s delay between each ──────────
section("Scenario 3: Different inputs, 3s delay between each")
for text in INPUTS:
    run("", text, delay_before=3.0)

# ── Scenario 4: Same input with 3s delay between each ────────────────
section("Scenario 4: Same input, 3s delay between each (3x)")
for i in range(3):
    run(f"call {i+1}", INPUTS[0], delay_before=3.0 if i > 0 else 0.0)

# ── Summary ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  DIAGNOSIS GUIDE")
print("="*60)
print("Scenario 2 all fast  → GPU idle (time gap) is the cause")
print("Scenario 2 has slow  → input change itself is the cause")
print("Scenario 3 all slow  → GPU goes idle after ~3s")
print("Scenario 4 all slow  → GPU goes idle even with same input")
