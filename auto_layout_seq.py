"""Sequential layout generation with Qwen2.5-VL.

Flow:
- Ask VL for subject list (noun-only, count = num_images).
- For each subject i:
    * Render current boxes to an image.
    * Send image + existing coords + prompt to VL and request ONE box.
- Draw final layout and print JSON.

Deps: pip install "transformers>=4.45" qwen-vl-utils matplotlib pillow

Usage:
    python auto_layout_seq.py \
      --prompt "a dog wearing the hat and sitting next to one cat in the garden" \
      --num-images 3 \
      --model /home/zchengay/Model/qwen2.5 \
      --output scheduler_plots/qwen_layout_seq.png
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


def chat_vl(messages: List[Dict[str, Any]], model, processor, max_new_tokens: int, temperature: float) -> str:
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
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)]
    response = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response.strip()


def get_subjects(prompt: str, num_images: int, model, processor, max_new_tokens: int, temperature: float) -> List[str]:
    messages = [
        {
            "role": "system",
            "content": "Return JSON {\"subjects\": [...]} with exactly num_images noun-only lowercase words, no commentary.",
        },
        {"role": "user", "content": f"num_images={num_images}. description: {prompt}"},
    ]
    raw = chat_vl(messages, model, processor, max_new_tokens, temperature)
    data = extract_json_block(raw)
    subjects = data.get("subjects") or data.get("phrases") or []
    if isinstance(subjects, list) and subjects and isinstance(subjects[0], list):
        subjects = subjects[0]
    if not isinstance(subjects, list):
        raise ValueError("Could not parse subjects list")
    subjects = [str(s).strip().lower() for s in subjects][: num_images]
    while len(subjects) < num_images:
        subjects.append("")
    return subjects


def place_one(prompt: str, subject: str, idx: int, existing_boxes: List[List[float]], phrases: List[str], image_path: Path, model, processor, max_new_tokens: int, temperature: float) -> List[float]:
    desc_existing = [f"{phrases[i]}: {existing_boxes[i]}" for i in range(len(existing_boxes))] or ["none"]
    messages = [
        {
            "role": "system",
            "content": "Given the image and existing boxes, add ONE box for the subject. Return ONLY JSON {\"box\": [x1,y1,x2,y2]} in [0,1]. Avoid heavy overlap.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {
                    "type": "text",
                    "text": (
                        f"Instruction: {prompt}\nExisting boxes: {desc_existing}\nPlace subject #{idx + 1}: '{subject}'."
                    ),
                },
            ],
        },
    ]

    raw = chat_vl(messages, model, processor, max_new_tokens, temperature)
    data = extract_json_block(raw)
    box = data.get("box") or data.get("bbox")
    if not box:
        raise ValueError("No box returned")
    x1, y1, x2, y2 = [float(v) for v in box]
    # clamp to [0,1]
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    return [x1, y1, x2, y2]


def main():
    parser = argparse.ArgumentParser(description="Sequential layout via Qwen2.5-VL")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--num-images", type=int, required=True)
    parser.add_argument("--model", type=str, default="/home/zchengay/Model/qwen2.5")
    parser.add_argument("--output", type=Path, default=Path("scheduler_plots/qwen_layout_seq.png"))
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()

    model, processor = load_vl_model(args.model)

    subjects = get_subjects(
        prompt=args.prompt,
        num_images=args.num_images,
        model=model,
        processor=processor,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    boxes: List[List[float]] = []
    phrases: List[str] = subjects

    for idx, subject in enumerate(subjects):
        box = None
        feedback = ""
        for _ in range(args.max_attempts):
            step_img = args.output.parent / f"step_{idx}.png"
            draw_layout(phrases[: len(boxes)], boxes, args.width, args.height, step_img)
            try:
                box_candidate = place_one(
                    prompt=args.prompt if not feedback else f"{args.prompt}\nPrevious issue: {feedback}",
                    subject=subject or f"subject_{idx}",
                    idx=idx,
                    existing_boxes=boxes,
                    phrases=phrases,
                    image_path=step_img,
                    model=model,
                    processor=processor,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
                box = box_candidate
                break
            except Exception as exc:
                feedback = str(exc)
                continue
        if box is None:
            raise SystemExit(f"Failed to place subject {idx}: {subject}")
        boxes.append(box)

    draw_layout(phrases, boxes, args.width, args.height, args.output)
    print(json.dumps({"phrases": [phrases], "boxes": [boxes]}, indent=2))
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
