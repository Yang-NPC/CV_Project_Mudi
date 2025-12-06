"""One-shot layout generation with Qwen2.5-VL.

Usage:
    python auto_layout_once.py \
        --prompt "a dog wearing the hat and sitting next to one cat in the garden" \
        --num-images 3 \
        --model /home/zchengay/Model/qwen2.5 \
        --output scheduler_plots/qwen_layout_once.png

Dependencies:
    pip install "transformers>=4.45" qwen-vl-utils matplotlib pillow
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info  # type: ignore


def load_vl_model(model_path: str):
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device_map = "auto"
    else:
        dtype = torch.float32
        device_map = {"": "cpu"}

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor


def draw_layout(phrases: List[str], boxes: List[List[float]], width: int, height: int, save_path: Path) -> Path:
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    fig, ax = plt.subplots(figsize=(width / 200, height / 200), dpi=200)
    ax.imshow(img)
    ax.set_title("Qwen-generated layout")
    colors = plt.cm.get_cmap("tab10")

    for idx, (label, b) in enumerate(zip(phrases, boxes)):
        x1, y1, x2, y2 = b
        rect = plt.Rectangle(
            (x1 * width, y1 * height),
            (x2 - x1) * width,
            (y2 - y1) * height,
            linewidth=2,
            edgecolor=colors(idx % 10),
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1 * width,
            y1 * height - 5,
            label,
            color=colors(idx % 10),
            fontsize=10,
            weight="bold",
        )

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def extract_json_block(response: str) -> Dict[str, Any]:
    start = response.find("{")
    end = response.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    return json.loads(response[start : end + 1])


def main():
    parser = argparse.ArgumentParser(description="One-shot layout via Qwen2.5-VL")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--num-images", type=int, required=True)
    parser.add_argument("--model", type=str, default="/home/zchengay/Model/qwen2.5")
    parser.add_argument("--output", type=Path, default=Path("scheduler_plots/qwen_layout_once.png"))
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    model, processor = load_vl_model(args.model)

    # Ask VL for phrases+boxes in one shot
    messages = [
        {
            "role": "system",
            "content": (
                "Return ONLY JSON: {\"phrases\": [[...]], \"boxes\": [[[x1,y1,x2,y2],...]]}. "
                "phrases are noun-only lowercase; counts must equal num_images; coords in [0,1]."
            ),
        },
        {"role": "user", "content": f"num_images={args.num_images}. description: {args.prompt}"},
    ]

    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[chat_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.temperature > 0,
        )

    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)]
    response = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    data = extract_json_block(response)

    phrases = data.get("phrases") or []
    boxes = data.get("boxes") or []
    if phrases and isinstance(phrases[0], list):
        phrases = phrases[0]
    if boxes and isinstance(boxes[0], list):
        boxes = boxes[0]

    phrases = [str(p).strip().lower() for p in phrases][: args.num_images]
    boxes_clean: List[List[float]] = []
    for b in boxes[: args.num_images]:
        x1, y1, x2, y2 = [float(v) for v in b]
        boxes_clean.append([max(0.0, min(1.0, x1)), max(0.0, min(1.0, y1)), max(0.0, min(1.0, x2)), max(0.0, min(1.0, y2))])

    draw_layout(phrases, boxes_clean, args.width, args.height, args.output)
    print(json.dumps({"phrases": [phrases], "boxes": [boxes_clean]}, indent=2))
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
