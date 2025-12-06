"""Prompt-to-image pipeline using Qwen2.5-VL layout + MUSE schedules.

Flow:
1) Qwen2.5-VL predicts phrases and boxes (2 decimal coordinates) for given prompt.
2) Run MUSE with three scale schedules: static, increasing, decreasing.

Example:
    python auto_layout_pipeline.py \
        --prompt "a dog wearing the hat and sitting next to one cat in the garden" \
        --imgs imgs/dog1.jpg imgs/dog2.jpg imgs/cat1.jpg \
        --output-dir result/auto_pipeline

Requirements: transformers>=4.45, qwen-vl-utils, diffusers, torch, pillow, matplotlib (only for potential debugging).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from qwen_vl_utils import process_vision_info  # type: ignore
from transformers import (
    AutoProcessor,
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    Qwen2_5_VLForConditionalGeneration,
)

from adapter_modules.adapter import PositionNetPLUS as PositionNet
from adapter_modules.model import MuseAdapter
from adapter_modules.utils import save_generated_images, seed_everything

# Default model locations tailored to current workspace
DEFAULT_QWEN_PATH = "/home/zchengay/Model/qwen2.5"
DEFAULT_SDXL_PATH = "/home/zchengay/Model/sdxl-base"
DEFAULT_CLIP_PATH = "/home/zchengay/Model/CLIP-G"
DEFAULT_MUSE_CKPT = "/home/zchengay/Model/MUSE/muse_weight.pth"
NUM_TOKENS = 4


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


def extract_json_block(response: str) -> Dict[str, Any]:
    start = response.find("{")
    end = response.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response")
    return json.loads(response[start : end + 1])


def clamp_round(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 2)


def round_boxes(boxes: Sequence[Sequence[float]]) -> List[List[float]]:
    cleaned: List[List[float]] = []
    for b in boxes:
        if len(b) != 4:
            raise ValueError(f"Box must have four values, got {b}")
        x1, y1, x2, y2 = [clamp_round(v) for v in b]
        cleaned.append([x1, y1, x2, y2])
    return cleaned


class MusePipeline(torch.nn.Module):
    def __init__(
        self,
        muse_ckpt: str,
        base_model_path: str = DEFAULT_SDXL_PATH,
        image_encoder_path: str = DEFAULT_CLIP_PATH,
        dynamic_scale: bool = False,
        scale_schedule: str = "linear",
        num_tokens: int = NUM_TOKENS,
    ):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.image_processor = CLIPImageProcessor()
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to(
            self.device, dtype=torch.float16
        )

        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            add_watermarker=False,
        )
        self.pipe.to(self.device)

        self.image_proj_model = PositionNet(
            in_dim_text=768,
            in_dim_image=self.image_encoder.config.hidden_size,
            out_dim_text=2048,
            out_dim_image=self.pipe.unet.config.cross_attention_dim,
            fourier_freqs=16,
            num_tokens_text=1,
            num_tokens_image=4,
        ).to(self.device, dtype=torch.float16)

        self.muse_model = MuseAdapter(
            self.pipe.unet,
            self.image_proj_model,
            ckpt_path=muse_ckpt,
            device=self.device,
            num_tokens=num_tokens,
            dynamic_scale=dynamic_scale,
            scale_schedule=scale_schedule,
        )

    def generate(
        self,
        imgs: Sequence[str],
        phrases: List[List[str]],
        boxes: List[List[List[float]]],
        prompt: str,
        save_img_name: str,
        result_path: str,
        num_samples: int = 1,
        log_id: str = "auto",
        height: int = 1024,
        width: int = 1024,
        max_box_num: int | None = None,
        scale: float = 1.0,
        seed: int = 0,
        num_inference_steps: int = 30,
        debug_predict_step: int | None = None,
        debug_save_dir: str | None = None,
    ):
        pil_images = [Image.open(p).convert("RGB").resize((512, 512)) for p in imgs]

        if max_box_num is None:
            max_box_num = len(imgs)

        for _ in range(max_box_num - len(phrases[0])):
            phrases[0].append("")

        masks = torch.zeros([1, max_box_num]).to(self.device, dtype=torch.float16)
        masks[0, : len(imgs)] = 1

        images = self.muse_model.generate(
            pipe=self.pipe,
            num_samples=num_samples,
            num_inference_steps=num_inference_steps,
            seed=seed,
            prompt=[prompt],
            scale=scale,
            boxes=boxes,
            phrases=phrases,
            height=height,
            width=width,
            max_box_num=max_box_num,
            masks=masks,
            text_masks=masks,
            pil_images=pil_images,
            image_masks=masks,
            image_encoder=self.image_encoder,
            image_processor=self.image_processor,
            debug_predict_step=debug_predict_step,
            debug_save_dir=debug_save_dir,
        )

        save_path = None
        if result_path and log_id:
            save_path = Path(result_path) / log_id
            save_generated_images(images, boxes, phrases, str(save_path), save_img_name, prompt, width, height)
        return images, save_path


def run_qwen_layout(model, processor, prompt: str, num_images: int, max_new_tokens: int, temperature: float):
    messages = [
        {
            "role": "system",
            "content": (
                "Return ONLY JSON: {\"phrases\": [[...]], \"boxes\": [[[x1,y1,x2,y2],...]]}. "
                "phrases are noun-only lowercase; counts must equal num_images; coords in [0,1] with two decimals."
            ),
        },
        {"role": "user", "content": f"num_images={num_images}. description: {prompt}"},
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
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

    trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, outputs)]
    response = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    data = extract_json_block(response)

    phrases = data.get("phrases") or []
    boxes = data.get("boxes") or []
    if phrases and isinstance(phrases[0], list):
        phrases = phrases[0]
    if boxes and isinstance(boxes[0], list):
        boxes = boxes[0]

    phrases = [str(p).strip().lower() for p in phrases][:num_images]
    boxes_clean = round_boxes(boxes[:num_images])
    return phrases, boxes_clean


def parse_args():
    parser = argparse.ArgumentParser(description="Prompt -> Qwen layout -> MUSE generation")
    parser.add_argument("--prompt", type=str, required=True, help="User prompt describing the scene")
    parser.add_argument("--imgs", nargs="+", required=True, help="List of subject reference images")
    parser.add_argument("--qwen-model", type=str, default=DEFAULT_QWEN_PATH)
    parser.add_argument("--sdxl-model", type=str, default=DEFAULT_SDXL_PATH)
    parser.add_argument("--clip-model", type=str, default=DEFAULT_CLIP_PATH)
    parser.add_argument("--muse-ckpt", type=str, default=DEFAULT_MUSE_CKPT)
    parser.add_argument("--output-dir", type=Path, default=Path("result/auto_pipeline"))
    parser.add_argument("--log-id", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--scale", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--schedules",
        nargs="+",
        default=["static", "increasing", "decreasing"],
        help="Scale schedules to run",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    num_images = len(args.imgs)
    model, processor = load_vl_model(args.qwen_model)
    phrases, boxes = run_qwen_layout(
        model,
        processor,
        prompt=args.prompt,
        num_images=num_images,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print(json.dumps({"phrases": [phrases], "boxes": [boxes]}, indent=2))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for schedule in args.schedules:
        print(f"\n=== Running schedule: {schedule} (scale={args.scale}) ===")
        pipeline = MusePipeline(
            muse_ckpt=args.muse_ckpt,
            base_model_path=args.sdxl_model,
            image_encoder_path=args.clip_model,
            dynamic_scale=True,
            scale_schedule=schedule,
            num_tokens=NUM_TOKENS,
        )

        save_img_name = f"auto_{schedule}.png"
        log_id = f"{args.log_id}_{schedule}"

        images, save_path = pipeline.generate(
            imgs=args.imgs,
            phrases=[phrases.copy()],
            boxes=[boxes],
            prompt=args.prompt,
            num_samples=1,
            log_id=log_id,
            result_path=str(args.output_dir),
            height=args.height,
            width=args.width,
            max_box_num=num_images,
            save_img_name=save_img_name,
            scale=args.scale,
            seed=args.seed,
            num_inference_steps=args.num_steps,
            debug_predict_step=None,
            debug_save_dir=None,
        )

        out_msg = f"Saved schedule {schedule} to {save_path}" if save_path else f"Completed {schedule}"
        print(out_msg)

        # Free GPU between schedules
        del pipeline
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
