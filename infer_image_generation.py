import os
import torch
from argparse import ArgumentParser
from mobileo.constants import IMAGE_TOKEN_INDEX
from mobileo.model.builder import load_pretrained_model
from mobileo.mm_utils import tokenizer_image_token
from mobileo.conversation import conv_templates


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


parser = ArgumentParser()
parser.add_argument("--model_path", type=str, default="models/Mobile-O-0.5B")
parser.add_argument("--prompt", type=str, default="a photo of a cute cat")
parser.add_argument("--output", type=str, default="predictions/mobileo_gen.png")
parser.add_argument("--num-steps", type=int, default=20)
parser.add_argument("--guidance-scale", type=float, default=1.5)
parser.add_argument("--no-cfg", action="store_true")
parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
args = parser.parse_args()

device = pick_device() if args.device == "auto" else (
    "cuda:0" if args.device == "cuda" else args.device
)
print(f"Using device: {device}")

tokenizer, model, _ = load_pretrained_model(args.model_path)
dtype = torch.bfloat16 if device != "cpu" else torch.float32
model.to(dtype).to(device)


def infer(prompt: str):
    qs = "Please generate image based on the following caption: " + prompt
    conv = conv_templates["qwen_2"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    text = conv.get_prompt()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    input_ids = tokenizer_image_token(
        text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device)
    with torch.inference_mode():
        output_image = model.generate_image(
            input_ids,
            pixel_values=None,
            with_cfg=not args.no_cfg,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
        )
    return output_image[0]


def main():
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    image = infer(args.prompt)
    image.save(args.output)
    print(f"Saved: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
