"""Generate bounding boxes from a natural-language instruction using Qwen 2.5 (text) and visualize them.

workflow:
1. Provide a textual layout description (e.g. "a hat on the cat, a dog on the left").
2. The script queries a Qwen 2.5 chat model to produce `phrases` and `boxes` in the
    same schema expected by `inference.py`.
3. The parsed layout is rendered to an image for quick inspection.

requirements:
    pip install transformers accelerate matplotlib pillow
    # (add `flash-attn` or other optimizations if desired)

usage:
    python auto_layout_from_qwen.py \
        --prompt "a hat on the cat" \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --output ./scheduler_plots/qwen_layout_preview.png

Notes:
- Adjust the model name/path to any locally available Qwen 2.5 checkpoint.
- Generation defaults are conservative; tweak `--max-new-tokens`, `--temperature`, etc.
- The script assumes at most 10 subjects as requested.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = """You are a helpful assistant that extracts structured bounding boxes.
Input: a natural description AND the number of subject images (num_images).
Output: ONLY a JSON object:
{
    "phrases": [["subject1", "subject2", ...]],
    "boxes": [[[x1, y1, x2, y2], ...]]
}
Rules (STRICT):
- Return exactly ONE inner list for phrases and ONE inner list for boxes.
- The number of entries in phrases[0] MUST equal num_images; same for boxes[0].
- Each phrase MUST be the main noun only, lowercase, no adjectives (e.g., "dog", "cat", "hat").
- Coordinates are floats in [0,1] for (x1,y1,x2,y2) relative to a 1024x1024 canvas; keep one decimal place.
- Spatial hints: if instruction says "A wearing B" or "B on A", place B's box inside the top of A's box. If it says "next to", place subjects horizontally adjacent with minimal overlap. Keep relative positions faithful.
- Up to 10 subjects. No commentary outside the JSON."""


def clamp_box(box: List[float]) -> List[float]:
    """Clamp a box to [0,1] and enforce minimum size."""

    x1, y1, x2, y2 = box
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    # Ensure ordering and a small minimum area
    eps = 0.02
    if x2 - x1 < eps:
        x2 = min(1.0, x1 + eps)
    if y2 - y1 < eps:
        y2 = min(1.0, y1 + eps)
    if x1 >= x2:
        x1 = max(0.0, x2 - eps)
    if y1 >= y2:
        y1 = max(0.0, y2 - eps)
    return [x1, y1, x2, y2]


def load_text_model(model_name: str):
    """Load a Qwen2.5 text model and tokenizer once."""

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device_map = "auto"
    else:
        dtype = torch.float32
        device_map = {"": "cpu"}

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        return model, tokenizer
    except Exception:
        # Fallback for Qwen2.5-VL checkpoints which expose a different class name
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration  # type: ignore

            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            return model, tokenizer
        except Exception as exc:
            raise RuntimeError(
                "Failed to load model. Please upgrade transformers to >=4.45 and ensure Qwen2.5-VL checkpoint is intact."
            ) from exc


def chat_generate(
    messages: List[Dict[str, str]],
    model,
    tokenizer,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Generic chat generation with preloaded model/tokenizer."""

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer([chat_text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

    trimmed = [out_ids[len(inp_ids):] for inp_ids, out_ids in zip(inputs.input_ids, outputs)]
    response = tokenizer.batch_decode(trimmed, skip_special_tokens=True)[0]
    return response.strip()


def chat_with_qwen(prompt: str, num_images: int, model_name: str, max_new_tokens: int, temperature: float) -> str:
    """Call local Qwen 2.5 instruct model and return the raw text response."""

    # Prefer bfloat16 on GPU when available; fall back otherwise
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device_map = "auto"
    else:
        dtype = torch.float32
        device_map = {"": "cpu"}

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"num_images={num_images}. description: {prompt}"},
    ]

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer([chat_text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

    # Trim the input portion
    trimmed = [out_ids[len(inp_ids):] for inp_ids, out_ids in zip(inputs.input_ids, outputs)]
    response = tokenizer.batch_decode(trimmed, skip_special_tokens=True)[0]
    return response.strip()


def extract_layout(response: str) -> Dict[str, Any]:
    """Extract the JSON block containing phrases and boxes."""

    match = re.search(r"\{[\s\S]*\}", response)
    if not match:
        raise ValueError("Unable to find JSON object in model response")

    data = json.loads(match.group())
    if "phrases" not in data or "boxes" not in data:
        raise ValueError("Model JSON missing required keys")
    return data


def extract_json_block(response: str) -> Dict[str, Any]:
    """Generic JSON extractor for verifier responses."""
    match = re.search(r"\{[\s\S]*\}", response)
    if not match:
        raise ValueError("Unable to find JSON object in verifier response")
    return json.loads(match.group())


def get_subject_list(
    prompt: str,
    num_images: int,
    model,
    tokenizer,
    max_new_tokens: int,
    temperature: float,
) -> List[str]:
    """Ask the LLM for noun-only subjects in order."""

    messages = [
        {
            "role": "system",
            "content": (
                "Extract the key subjects as noun-only words, lowercase. Return JSON {\"subjects\": [...]} "
                "with exactly num_images items, no commentary."
            ),
        },
        {
            "role": "user",
            "content": f"num_images={num_images}. description: {prompt}",
        },
    ]

    raw = chat_generate(messages, model, tokenizer, max_new_tokens, temperature)
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


def validate_layout(boxes: List[List[float]], num_images: int) -> Tuple[bool, List[str]]:
    """Validate normalized boxes.

    Checks: count matches, coords within [0,1], x1<x2,y1<y2, min area, mild overlap penalty.
    """

    reasons = []
    if len(boxes) != num_images:
        reasons.append(f"expected {num_images} boxes, got {len(boxes)}")

    valid_boxes = []
    for i, b in enumerate(boxes):
        if len(b) != 4:
            reasons.append(f"box {i} malformed: {b}")
            continue
        x1, y1, x2, y2 = b
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            reasons.append(f"box {i} out of range or inverted: {b}")
            continue
        area = (x2 - x1) * (y2 - y1)
        if area < 0.01:  # require at least 1% of canvas
            reasons.append(f"box {i} too small: area={area:.4f}")
            continue
        valid_boxes.append(b)

    # Simple overlap check (IoU > 0.7 considered too high)
    def iou(b1, b2):
        xa1, ya1, xa2, ya2 = b1
        xb1, yb1, xb2, yb2 = b2
        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)
        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            return 0.0
        inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = (xa2 - xa1) * (ya2 - ya1)
        area_b = (xb2 - xb1) * (yb2 - yb1)
        return inter / (area_a + area_b - inter + 1e-8)

    for i in range(len(valid_boxes)):
        for j in range(i + 1, len(valid_boxes)):
            if iou(valid_boxes[i], valid_boxes[j]) > 0.7:
                reasons.append(f"boxes {i} and {j} overlap too much")
                break

    return len(reasons) == 0, reasons


def validate_box_with_existing(box: List[float], existing: List[List[float]]) -> Tuple[bool, List[str]]:
    """Validate a single box against canvas bounds and light overlap with existing boxes."""

    reasons = []
    if len(box) != 4:
        return False, ["box must have four values"]

    x1, y1, x2, y2 = box
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        reasons.append("box out of bounds or inverted")
    area = (x2 - x1) * (y2 - y1)
    if area < 0.01:
        reasons.append("box too small (<1% area)")

    def iou(b1, b2):
        xa1, ya1, xa2, ya2 = b1
        xb1, yb1, xb2, yb2 = b2
        inter_x1 = max(xa1, xb1)
        inter_y1 = max(ya1, yb1)
        inter_x2 = min(xa2, xb2)
        inter_y2 = min(ya2, yb2)
        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            return 0.0
        inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = (xa2 - xa1) * (ya2 - ya1)
        area_b = (xb2 - xb1) * (yb2 - yb1)
        return inter / (area_a + area_b - inter + 1e-8)

    for idx, b in enumerate(existing):
        if iou(box, b) > 0.8:
            reasons.append(f"overlaps too much with box {idx}")
            break

    return len(reasons) == 0, reasons


def place_subject(
    prompt: str,
    subject: str,
    idx: int,
    existing: List[List[float]],
    model,
    tokenizer,
    max_new_tokens: int,
    temperature: float,
) -> List[float]:
    """Ask LLM to place one subject given prior boxes."""

    placed_desc = [f"{i}: {box}" for i, box in enumerate(existing)] or ["none"]
    user_text = (
        f"Full instruction: {prompt}\n"
        f"We have placed {len(existing)} subjects so far: {', '.join(placed_desc)}.\n"
        f"Place subject #{idx + 1}: '{subject}'. Return ONLY JSON: {{\"box\": [x1,y1,x2,y2]}}."
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Place exactly one bounding box for the requested subject. "
                "Keep coordinates in [0,1], one decimal, min area 1%. Do not modify existing boxes."
            ),
        },
        {"role": "user", "content": user_text},
    ]

    raw = chat_generate(messages, model, tokenizer, max_new_tokens, temperature)
    data = extract_json_block(raw)
    box = data.get("box") or data.get("bbox")
    if not box:
        raise ValueError("No box returned")
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except Exception as exc:
        raise ValueError(f"Invalid box parse: {box}") from exc
    return clamp_box([x1, y1, x2, y2])


def verify_with_image(
    image_path: Path,
    prompt: str,
    num_images: int,
    model_name: str,
    max_new_tokens: int,
    temperature: float,
) -> Optional[Dict[str, Any]]:
    """Ask a Qwen2.5-VL model to judge the layout using the official HF flow (process_vision_info)."""

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor  # type: ignore
        from qwen_vl_utils import process_vision_info  # type: ignore
    except Exception as exc:
        raise RuntimeError("Failed to import Qwen2.5-VL classes; ensure transformers>=4.45 and qwen-vl-utils installed") from exc

    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device_map = "auto"
    else:
        dtype = torch.float32
        device_map = {"": "cpu"}

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a layout verifier. Check if the boxes match the instruction. "
                "If incorrect, return corrected JSON with phrases/boxes."
            ),
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {
                    "type": "text",
                    "text": (
                        f"Instruction: {prompt}\nnum_images={num_images}. "
                        "Respond ONLY JSON: {\"ok\": bool, \"reason\": str, \"phrases\": [[...]], \"boxes\": [[[...]]]}."
                    ),
                },
            ],
        },
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
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    response = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    return extract_json_block(response)


def draw_layout(phrases: List[List[str]], boxes: List[List[List[float]]], width: int, height: int, save_path: Path) -> Path:
    """Render bounding boxes onto a blank canvas using matplotlib."""

    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    fig, ax = plt.subplots(figsize=(width / 200, height / 200), dpi=200)
    ax.imshow(img)
    ax.set_title("Qwen-generated layout")

    colors = plt.cm.get_cmap("tab10")

    flat_phrases = phrases[0] if phrases else []
    raw_boxes = boxes[0] if boxes else []

    # Coerce to float and validate
    flat_boxes = []
    for b in raw_boxes:
        try:
            x1, y1, x2, y2 = [float(v) for v in b]
            flat_boxes.append([x1, y1, x2, y2])
        except Exception:
            continue

    for idx, (label, bbox) in enumerate(zip(flat_phrases, flat_boxes)):
        x1, y1, x2, y2 = bbox
        rect_width = (x2 - x1) * width
        rect_height = (y2 - y1) * height
        rect = plt.Rectangle(
            (x1 * width, y1 * height),
            rect_width,
            rect_height,
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


def main():
    parser = argparse.ArgumentParser(description="Generate bounding boxes via Qwen and visualize them")
    parser.add_argument("--prompt", type=str, required=True, help="Natural language layout description (<=10 subjects)")
    parser.add_argument("--model", type=str, default="/home/zchengay/Model/qwen2.5", help="Qwen2.5 model local path")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=Path("scheduler_plots/qwen_layout_preview.png"))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num-images", type=int, required=True, help="Number of subjects (boxes) expected")
    parser.add_argument("--max-attempts", type=int, default=3, help="Max regeneration attempts if layout invalid")
    parser.add_argument("--verifier-model", type=str, default=None, help="Optional Qwen2.5-VL model path for image-based verification (text-only models will be skipped)")
    args = parser.parse_args()
    # Load model once
    model, tokenizer = load_text_model(args.model)

    # 1) Extract ordered subjects
    subjects = get_subject_list(
        prompt=args.prompt,
        num_images=args.num_images,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    boxes: List[List[float]] = []
    phrases: List[str] = subjects

    # 2) Place each subject sequentially
    for idx, subject in enumerate(subjects):
        box: Optional[List[float]] = None
        feedback = ""
        for attempt in range(1, args.max_attempts + 1):
            try:
                box_candidate = place_subject(
                    prompt=args.prompt if not feedback else f"{args.prompt}\nPrevious attempt issue: {feedback}",
                    subject=subject or f"subject_{idx}",
                    idx=idx,
                    existing=boxes,
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
            except Exception as exc:
                feedback = str(exc)
                continue

            is_ok, reasons = validate_box_with_existing(box_candidate, boxes)
            if is_ok:
                box = box_candidate
                break
            feedback = "; ".join(reasons)
            print(f"Subject {idx} ({subject}) invalid (attempt {attempt}): {feedback}")

        if box is None:
            raise SystemExit(f"Failed to place subject {idx}: {subject}")

        boxes.append(box)

        # Optional step: verify incrementally if VL model is provided
        if args.verifier_model:
            img_path = draw_layout([phrases], [boxes], args.width, args.height, args.output)
            verdict = verify_with_image(
                image_path=img_path,
                prompt=args.prompt,
                num_images=len(boxes),
                model_name=args.verifier_model,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            if verdict and isinstance(verdict, dict) and verdict.get("ok") is False:
                print(f"Verifier feedback after subject {idx}: {verdict.get('reason', 'unknown reason')}")

    # Final validation for full layout
    is_valid, reasons = validate_layout(boxes, args.num_images)
    if not is_valid:
        raise SystemExit(f"Final layout invalid: {'; '.join(reasons)}")

    draw_layout([phrases], [boxes], args.width, args.height, args.output)
    print("Parsed layout:")
    print(json.dumps({"phrases": [phrases], "boxes": [boxes]}, indent=2))
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
