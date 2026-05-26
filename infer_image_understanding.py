import torch
from PIL import Image
from argparse import ArgumentParser

from mobileo.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from mobileo.model.builder import load_pretrained_model
from mobileo.utils import disable_torch_init
from mobileo.mm_utils import tokenizer_image_token, process_images
from mobileo.conversation import conv_templates


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


parser = ArgumentParser()
parser.add_argument("--model_path", type=str, default="models/Mobile-O-0.5B")
parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
parser.add_argument("--prompt", type=str, default="What is in the image?")
parser.add_argument("--max-new-tokens", type=int, default=256)
args = parser.parse_args()

device = pick_device()
print(f"Using device: {device}")

disable_torch_init()
print("Loading model (first run may take 10–20 min: weights + Sana/VAE from Hugging Face)...")
tokenizer, model, _ = load_pretrained_model(args.model_path)
print("Model loaded. Running inference...")
dtype = torch.bfloat16 if device != "cpu" else torch.float32
model.to(dtype).to(device)

image_processor = model.get_vision_tower().image_processor

qs = DEFAULT_IMAGE_TOKEN + "\n" + args.prompt
conv = conv_templates["qwen_2"].copy()
conv.append_message(conv.roles[0], qs)
conv.append_message(conv.roles[1], None)
text = conv.get_prompt()

model.generation_config.pad_token_id = tokenizer.pad_token_id
input_ids = tokenizer_image_token(
    text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
).unsqueeze(0).to(device)

image_tensor = process_images(
    [Image.open(args.image_path).convert("RGB")], image_processor, model.config
)[0]

with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=image_tensor.unsqueeze(0).to(dtype).to(device),
        do_sample=False,
        temperature=0.0,
        num_beams=1,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

print(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip())
