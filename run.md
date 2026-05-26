# Quick run & Mac benchmark

## One-time local download (faster loads)

Download weights **into the repo** once (not just HF cache):

```bash
conda activate mobileo-arm
pip install -r requirements-mac.txt

python scripts/download_models_mac.py
```

This creates:

```text
models/Mobile-O-0.5B/
  model.safetensors   (~4.8 GB — must NOT be 135 bytes)
  config.json
  tokenizer files …
```

Also pre-caches **SANA** (DiT/VAE) so the first benchmark run skips that download.

Then always use:

```text
--model-path models/Mobile-O-0.5B
```

`hf download` only fills `~/.cache/huggingface/` — that works too, but a **local folder** avoids Hub lookups and is easier to verify (`ls -lh models/Mobile-O-0.5B/model.safetensors`).

---

## Mac benchmark (same 3 tasks as iPhone app)

Model load timed **once**; inference = 1 warmup + 3 measured runs with per-stage metrics.

```bash
python scripts/run_mobileo_benchmark_mac.py \
  --model-path models/Mobile-O-0.5B \
  --caption-image assets/cute_cat.png \
  --num-steps 20 \
  --guidance-scale 1.5
```

```bash
python3 scripts/analyze_mobileo_benchmark.py mobileo_benchmark_mac_*.csv
```

---

## Single-task inference

```bash
python infer_image_understanding.py \
  --model_path models/Mobile-O-0.5B \
  --image_path assets/cute_cat.png \
  --prompt "What is in the image?"

python infer_image_generation.py \
  --model_path models/Mobile-O-0.5B \
  --prompt "a photo of a cute cat" \
  --output predictions/my_image.png
```

Second run loads much faster (weights already on disk; SANA already cached).
