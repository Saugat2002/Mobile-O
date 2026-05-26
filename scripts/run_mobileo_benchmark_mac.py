#!/usr/bin/env python3
"""
Mac benchmark matching the iOS app BenchmarkRunner (3 tasks, same CSV columns).

- Model load is timed once (separate from inference).
- Each inference run records per-stage timings (gen: tokenize / LLM / connector / DiT / VAE).
- 1 warmup + 3 measured runs per task (configurable).

Setup:
  pip install -r requirements-mac.txt

Run:
  python scripts/run_mobileo_benchmark_mac.py \\
    --model-path Amshaker/Mobile-O-0.5B \\
    --caption-image assets/cute_cat.png \\
    --num-steps 20 --guidance-scale 1.5

Summarize (non-warmup rows):
  python3 scripts/analyze_mobileo_benchmark.py mobileo_benchmark_mac_*.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from transformers import TextIteratorStreamer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mobileo.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from mobileo.conversation import conv_templates
from mobileo.mm_utils import process_images, tokenizer_image_token
from mobileo.model.builder import load_pretrained_model
from mobileo.utils import disable_torch_init

GEN_PROMPT = "a red rose with morning dew drops"
CAPTION_PROMPT = "What is in this image? Describe in one sentence."
CHAT_PROMPT = "Explain what a neural network is in two sentences."


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        torch.mps.synchronize()


def mac_device_name() -> str:
    try:
        model = subprocess.check_output(["sysctl", "-n", "hw.model"], text=True).strip()
        chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        return f"Mac ({model}, {chip})"
    except Exception:
        return f"Mac ({platform.machine()})"


def memory_mb() -> tuple[float, float]:
    try:
        import psutil

        rss = psutil.Process().memory_info().rss / (1024 * 1024)
        return rss, rss
    except ImportError:
        return 0.0, 0.0


class MemoryPeakMonitor:
    def __init__(self) -> None:
        self._peak = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._peak = memory_mb()[1]
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._peak = max(self._peak, memory_mb()[1])
            time.sleep(0.05)

    def stop(self) -> float:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._peak = max(self._peak, memory_mb()[1])
        return self._peak

    def peak_mb(self) -> float:
        return max(self._peak, memory_mb()[1])


@dataclass
class GenTiming:
    tokenization_s: float
    llm_encode_s: float
    connector_s: float
    diffusion_s: float
    vae_s: float
    pipeline_total_s: float
    wall_clock_s: float


class CsvWriter:
    def __init__(self, path: Path, run_id: str, device: str, os_version: str):
        self.path = path
        self.run_id = run_id
        self.device = device
        self.os_version = os_version
        self.rows: list[dict] = []
        self.config: dict[str, str] = {}

    def add(
        self,
        task: str,
        phase: str,
        run_index: int,
        is_warmup: bool,
        metric: str,
        value: float,
        unit: str,
        notes: str = "",
    ) -> None:
        self.rows.append({
            "run_id": self.run_id,
            "device_model": self.device,
            "os_version": self.os_version,
            "task": task,
            "phase": phase,
            "run_index": str(run_index),
            "is_warmup": str(is_warmup).lower(),
            "metric": metric,
            "value": f"{value:.6f}",
            "unit": unit,
            "notes": notes,
        })

    def add_memory(
        self,
        task: str,
        run_index: int,
        is_warmup: bool,
        before: tuple[float, float],
        peak_mb: float,
        after: tuple[float, float],
    ) -> None:
        self.add(task, "total", run_index, is_warmup, "memory_resident_before_mb", before[0], "MB")
        self.add(task, "total", run_index, is_warmup, "memory_footprint_before_mb", before[1], "MB")
        self.add(task, "total", run_index, is_warmup, "memory_footprint_peak_mb", peak_mb, "MB")
        self.add(task, "total", run_index, is_warmup, "memory_footprint_after_mb", after[1], "MB")
        self.add(task, "total", run_index, is_warmup, "memory_resident_after_mb", after[0], "MB")

    def flush(self) -> None:
        fieldnames = [
            "run_id", "device_model", "os_version", "task", "phase", "run_index",
            "is_warmup", "metric", "value", "unit", "notes",
        ]
        with self.path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(self.rows)

    def write_json(self, path: Path) -> None:
        metrics = []
        for row in self.rows:
            metrics.append({
                "task": row["task"],
                "phase": row["phase"],
                "run_index": int(row["run_index"]),
                "is_warmup": row["is_warmup"] == "true",
                "metric": row["metric"],
                "value": float(row["value"]),
                "unit": row["unit"],
                "notes": row["notes"],
            })
        payload = {
            "run_id": self.run_id,
            "created_at": datetime.now().isoformat(),
            "device_model": self.device,
            "os_version": self.os_version,
            "config": self.config,
            "metrics": metrics,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_gen_prompt_text(prompt: str) -> str:
    qs = "Please generate image based on the following caption: " + prompt
    conv = conv_templates["qwen_2"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def build_understanding_prompt_text(prompt: str) -> str:
    qs = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    conv = conv_templates["qwen_2"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def build_chat_prompt_text(prompt: str) -> str:
    conv = conv_templates["qwen_2"].copy()
    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


@torch.no_grad()
def timed_generate_image(
    model,
    tokenizer,
    device: str,
    prompt: str,
    num_steps: int,
    guidance_scale: float,
    enable_cfg: bool,
) -> GenTiming:
    """Per-stage timings aligned with iOS BenchmarkRunner image_generation phases."""
    wall_start = time.perf_counter()

    t_tok = time.perf_counter()
    text = build_gen_prompt_text(prompt)
    input_ids = tokenizer_image_token(
        text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    tokenization_s = time.perf_counter() - t_tok

    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.to(torch.bfloat16)

    t_llm = time.perf_counter()
    inputs_embeds = model.get_model().embed_tokens(input_ids)
    model.model = model.model.to(torch.bfloat16)
    outputs = model.model(
        inputs_embeds=inputs_embeds,
        attention_mask=None,
        output_hidden_states=True,
        return_dict=True,
    )
    img_hidden_states = outputs.hidden_states
    llm_encode_s = time.perf_counter() - t_llm

    # Match sample_images(): batch_size before CFG concat; latents stay batch=1 when CFG doubles DiT input.
    batch_size = img_hidden_states[0].shape[0]
    if enable_cfg:
        pred_latents = tuple(
            torch.cat([torch.zeros_like(layer), layer], dim=0) for layer in img_hidden_states
        )
    else:
        pred_latents = img_hidden_states

    t_conn = time.perf_counter()
    encoder_hidden_states = model.model.diffusion_connector(pred_latents).float()
    connector_s = time.perf_counter() - t_conn

    latent_size = model.get_model().dit.config.sample_size
    latent_channels = model.get_model().dit.config.in_channels
    latents = randn_tensor(
        shape=(batch_size, latent_channels, latent_size, latent_size),
        generator=None,
        device=device,
        dtype=torch.float32,
    )

    t_diff = time.perf_counter()
    model.model.noise_scheduler.set_timesteps(num_steps)
    for t in model.model.noise_scheduler.timesteps:
        if enable_cfg:
            latent_model_input = torch.cat([latents] * 2)
        else:
            latent_model_input = latents

        if hasattr(model.model.noise_scheduler, "scale_model_input"):
            latent_model_input = model.model.noise_scheduler.scale_model_input(
                latent_model_input, t
            )

        noise_pred = model.model.dit(
            hidden_states=latent_model_input.to(torch.bfloat16),
            encoder_hidden_states=encoder_hidden_states.to(torch.bfloat16),
            timestep=t.unsqueeze(0).expand(latent_model_input.shape[0]).to(device),
            encoder_attention_mask=None,
        ).sample.float()

        if enable_cfg:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        latents = model.model.noise_scheduler.step(noise_pred, t, latents).prev_sample

    diffusion_s = time.perf_counter() - t_diff

    t_vae = time.perf_counter()
    _ = model.decode_latents(latents.to(model.model.vae.dtype), return_tensor=False)
    vae_s = time.perf_counter() - t_vae

    sync_device(device)
    pipeline_total_s = time.perf_counter() - wall_start
    wall_clock_s = pipeline_total_s

    return GenTiming(
        tokenization_s=tokenization_s,
        llm_encode_s=llm_encode_s,
        connector_s=connector_s,
        diffusion_s=diffusion_s,
        vae_s=vae_s,
        pipeline_total_s=pipeline_total_s,
        wall_clock_s=wall_clock_s,
    )


def generate_with_ttft(
    model,
    tokenizer,
    input_ids,
    device: str,
    dtype: torch.dtype,
    images: torch.Tensor | None,
    max_new_tokens: int,
) -> tuple[float, float, int]:
    """Returns (time_to_first_token_s, total_decode_s, tokens_generated)."""
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        input_ids=input_ids,
        do_sample=False,
        temperature=0.0,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        streamer=streamer,
    )
    if images is not None:
        gen_kwargs["images"] = images

    decode_start = time.perf_counter()
    thread = threading.Thread(
        target=lambda: model.generate(**gen_kwargs),
        daemon=True,
    )
    thread.start()

    ttft: float | None = None
    token_count = 0
    for _ in streamer:
        if ttft is None:
            ttft = time.perf_counter() - decode_start
        token_count += 1

    thread.join()
    total_s = time.perf_counter() - decode_start
    if ttft is None:
        ttft = total_s
    return ttft, total_s, token_count


def resolve_model_path(path_str: str) -> str:
    p = Path(path_str)
    if p.exists():
        return str(p.resolve())
    if "/" in path_str or path_str.count("-") >= 1:
        return path_str
    raise SystemExit(f"Model path not found: {path_str}")


def log_model_load(writer: CsvWriter, load_pretrained_s: float, move_device_s: float, peak_mb: float) -> None:
    """One-time model load metrics (not counted in task warmup/measured runs)."""
    total = load_pretrained_s + move_device_s
    writer.add("model_load", "setup", 0, False, "load_pretrained_s", load_pretrained_s, "s",
               notes="tokenizer + weights + Sana/VAE init")
    writer.add("model_load", "setup", 0, False, "move_to_device_s", move_device_s, "s")
    writer.add("model_load", "setup", 0, False, "load_total_s", total, "s")
    writer.add("model_load", "setup", 0, False, "memory_footprint_peak_mb", peak_mb, "MB",
               notes="peak during load")


def log_generation_run(writer: CsvWriter, run_index: int, is_warmup: bool,
                       timing: GenTiming, num_steps: int,
                       before: tuple[float, float], peak_mb: float, after: tuple[float, float]) -> None:
    writer.add_memory("image_generation", run_index, is_warmup, before, peak_mb, after)
    writer.add("image_generation", "total", run_index, is_warmup, "wall_clock_s", timing.wall_clock_s, "s")
    writer.add("image_generation", "text_encoding", run_index, is_warmup,
               "tokenization_s", timing.tokenization_s, "s")
    writer.add("image_generation", "text_encoding", run_index, is_warmup,
               "llm_encode_s", timing.llm_encode_s, "s")
    writer.add("image_generation", "connector", run_index, is_warmup,
               "connector_s", timing.connector_s, "s")
    writer.add("image_generation", "diffusion", run_index, is_warmup,
               "diffusion_s", timing.diffusion_s, "s", notes="DiT steps")
    writer.add("image_generation", "vae_decode", run_index, is_warmup,
               "vae_s", timing.vae_s, "s")
    writer.add("image_generation", "total", run_index, is_warmup,
               "pipeline_total_s", timing.pipeline_total_s, "s")
    writer.add("image_generation", "total", run_index, is_warmup,
               "inference_steps", float(num_steps), "count")


def main() -> None:
    p = argparse.ArgumentParser(description="Mobile-O Mac benchmark (matches iOS BenchmarkRunner)")
    p.add_argument(
        "--model-path",
        type=str,
        default="models/Mobile-O-0.5B",
        help="Local folder (models/Mobile-O-0.5B) or HF id (Amshaker/Mobile-O-0.5B)",
    )
    p.add_argument("--caption-image", type=str, default="assets/cute_cat.png")
    p.add_argument("--output-dir", type=str, default=".")
    p.add_argument("--warmup-runs", type=int, default=1)
    p.add_argument("--measured-runs", type=int, default=3)
    p.add_argument("--num-steps", type=int, default=20)
    p.add_argument("--guidance-scale", type=float, default=1.5)
    p.add_argument("--no-cfg", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    args = p.parse_args()

    device = pick_device() if args.device == "auto" else (
        "cuda:0" if args.device == "cuda" else args.device
    )
    caption_image = Path(args.caption_image)
    if not caption_image.is_file():
        sys.exit(f"Caption image not found: {caption_image}")

    model_path = resolve_model_path(args.model_path)
    dtype = torch.bfloat16 if device != "cpu" else torch.float32

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    os_label = f"macOS {platform.mac_ver()[0]}"
    out_csv = Path(args.output_dir) / f"mobileo_benchmark_mac_{run_id}.csv"
    out_json = Path(args.output_dir) / f"mobileo_benchmark_mac_{run_id}.json"
    writer = CsvWriter(out_csv, run_id, mac_device_name(), os_label)
    writer.config = {
        "warmup_runs": str(args.warmup_runs),
        "measured_runs": str(args.measured_runs),
        "num_steps": str(args.num_steps),
        "enable_cfg": str(not args.no_cfg),
        "guidance_scale": str(args.guidance_scale),
        "scheduler": "pytorch_dpm_solver",
        "stack": "pytorch_mps_or_cuda",
        "model_path": model_path,
        "generation_prompt": GEN_PROMPT,
        "caption_prompt": CAPTION_PROMPT,
        "chat_prompt": CHAT_PROMPT,
        "caption_image": str(caption_image.resolve()),
    }

    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print("=== Model load (timed separately, not in task runs) ===")

    disable_torch_init()
    load_mon = MemoryPeakMonitor()
    load_mon.start()
    t_load = time.perf_counter()
    tokenizer, model, _ = load_pretrained_model(model_path)
    load_pretrained_s = time.perf_counter() - t_load

    t_move = time.perf_counter()
    model.to(dtype).to(device)
    move_device_s = time.perf_counter() - t_move
    load_peak = load_mon.stop()

    log_model_load(writer, load_pretrained_s, move_device_s, load_peak)
    print(f"  load_pretrained: {load_pretrained_s:.1f}s")
    print(f"  move_to_device:  {move_device_s:.1f}s")
    print(f"  load_total:      {load_pretrained_s + move_device_s:.1f}s")

    image_processor = model.get_vision_tower().image_processor
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Pre-build caption tensors (not timed per run except vision/decode)
    cap_text = build_understanding_prompt_text(CAPTION_PROMPT)
    cap_input_ids = tokenizer_image_token(
        cap_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    cap_images = process_images(
        [Image.open(caption_image).convert("RGB")], image_processor, model.config
    )[0].unsqueeze(0).to(dtype).to(device)

    chat_text = build_chat_prompt_text(CHAT_PROMPT)
    chat_input_ids = tokenizer_image_token(
        chat_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)

    total_runs = args.warmup_runs + args.measured_runs
    enable_cfg = not args.no_cfg

    # --- Image generation ---
    print(f"\n=== Image generation ({total_runs} runs: {args.warmup_runs} warmup + {args.measured_runs} measured) ===")
    for run in range(total_runs):
        is_warmup = run < args.warmup_runs
        label = "warmup" if is_warmup else f"measured {run - args.warmup_runs + 1}"
        print(f"  run {run} ({label})…")

        before = memory_mb()
        mon = MemoryPeakMonitor()
        mon.start()
        timing = timed_generate_image(
            model, tokenizer, device, GEN_PROMPT,
            args.num_steps, args.guidance_scale, enable_cfg,
        )
        peak = mon.stop()
        after = memory_mb()

        log_generation_run(writer, run, is_warmup, timing, args.num_steps, before, peak, after)
        print(f"    pipeline {timing.pipeline_total_s:.2f}s "
              f"(DiT {timing.diffusion_s:.2f}s, VAE {timing.vae_s:.2f}s)")

    # --- Image captioning ---
    print(f"\n=== Image captioning ({total_runs} runs) ===")
    for run in range(total_runs):
        is_warmup = run < args.warmup_runs
        print(f"  run {run}…")

        before = memory_mb()
        mon = MemoryPeakMonitor()
        mon.start()

        t_vis = time.perf_counter()
        with torch.inference_mode():
            _ = model.get_model().get_vision_tower()(cap_images)
        vision_s = time.perf_counter() - t_vis
        sync_device(device)

        ttft, total_s, n_tokens = generate_with_ttft(
            model, tokenizer, cap_input_ids, device, dtype, cap_images, args.max_new_tokens
        )
        peak = mon.stop()
        after = memory_mb()
        tps = n_tokens / total_s if total_s > 0 else 0.0

        writer.add_memory("image_captioning", run, is_warmup, before, peak, after)
        writer.add("image_captioning", "vision_encoder", run, is_warmup, "vision_encoder_s", vision_s, "s")
        writer.add("image_captioning", "llm_decode", run, is_warmup, "time_to_first_token_s", ttft, "s")
        writer.add("image_captioning", "llm_decode", run, is_warmup, "total_s", total_s, "s")
        writer.add("image_captioning", "llm_decode", run, is_warmup, "tokens_generated", float(n_tokens), "count")
        writer.add("image_captioning", "llm_decode", run, is_warmup, "tokens_per_second", tps, "tok/s")
        print(f"    vision {vision_s:.3f}s, TTFT {ttft:.3f}s, total {total_s:.3f}s")

    # --- Text chat ---
    print(f"\n=== Text chat ({total_runs} runs) ===")
    for run in range(total_runs):
        is_warmup = run < args.warmup_runs
        print(f"  run {run}…")

        before = memory_mb()
        mon = MemoryPeakMonitor()
        mon.start()

        ttft, total_s, n_tokens = generate_with_ttft(
            model, tokenizer, chat_input_ids, device, dtype, None, args.max_new_tokens
        )
        peak = mon.stop()
        after = memory_mb()
        tps = n_tokens / total_s if total_s > 0 else 0.0

        writer.add_memory("text_chat", run, is_warmup, before, peak, after)
        writer.add("text_chat", "llm_decode", run, is_warmup, "time_to_first_token_s", ttft, "s")
        writer.add("text_chat", "llm_decode", run, is_warmup, "total_s", total_s, "s")
        writer.add("text_chat", "llm_decode", run, is_warmup, "tokens_generated", float(n_tokens), "count")
        writer.add("text_chat", "llm_decode", run, is_warmup, "tokens_per_second", tps, "tok/s")
        print(f"    TTFT {ttft:.3f}s, total {total_s:.3f}s")

    writer.add("meta", "config", 0, False, "warmup_runs", float(args.warmup_runs), "count")
    writer.add("meta", "config", 0, False, "measured_runs", float(args.measured_runs), "count")
    writer.flush()
    writer.write_json(out_json)

    print(f"\nSaved CSV:  {out_csv.resolve()}")
    print(f"Saved JSON: {out_json.resolve()}")
    print(f"Summarize:  python3 scripts/analyze_mobileo_benchmark.py {out_csv.name}")


if __name__ == "__main__":
    main()
