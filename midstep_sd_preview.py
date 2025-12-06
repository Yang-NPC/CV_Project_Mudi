"""Minimal example showing how to export a mid-step velocity prediction with SDXL.

The script runs a single prompt ("a detailed wooden table") through the SDXL base
pipeline and saves both the final output and a velocity-based x0 estimate captured
halfway through the diffusion process.
"""

import os
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline

from adapter_modules.utils import debug_save_latents


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    base_model_path = "/home/zchengay/Model/sdxl-base"

    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        add_watermarker=False,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    prompt = "a detailed wooden table in a sunlit studio"
    height = 1024
    width = 1024
    num_inference_steps = 30
    mid_step = 25  # Late step where latents are mostly denoised

    out_dir = Path("/home/zchengay/CV/MUSE/mid_debug_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )
    prompt_embeds = prompt_embeds.to(device)
    pooled_prompt_embeds = pooled_prompt_embeds.to(device)

    text_encoder_projection_dim = (
        pipe.text_encoder_2.config.projection_dim
        if pipe.text_encoder_2 is not None
        else int(pooled_prompt_embeds.shape[-1])
    )
    add_time_ids = pipe._get_add_time_ids(
        original_size=(height, width),
        crops_coords_top_left=(0, 0),
        target_size=(height, width),
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    ).to(device)
    time_ids = add_time_ids.repeat(prompt_embeds.shape[0], 1)

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    sigmas = pipe.scheduler.sigmas.to(device)
    timesteps = pipe.scheduler.timesteps.to(device)
    prediction_type = getattr(pipe.scheduler.config, "prediction_type", "v_prediction")

    preview_state = {"captured": False}

    def on_step_end(pipe_, step, timestep, callback_kwargs):
        # Save at multiple steps to see progression
        latents = callback_kwargs["latents"].detach()
        
        # Save raw latents stats for debugging
        print(f"Step {step}: latents min={latents.min().item():.4f}, max={latents.max().item():.4f}, mean={latents.mean().item():.4f}")
        
        # Save at steps 10, 20, 29
        if step in [10, 20, 29]:
            debug_save_latents(
                vae=pipe_.vae,
                latents=latents,
                step=step,
                save_dir=str(out_dir),
            )
        return callback_kwargs

    generator = torch.Generator(device=device).manual_seed(0)
    images = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        guidance_scale=1.0,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        generator=generator,
        callback_on_step_end=on_step_end,
        callback_on_step_end_tensor_inputs=["latents"],
    ).images

    final_path = out_dir / "final_table.png"
    images[0].save(final_path)
    print(f"Saved final image to {final_path}")
    if preview_state["captured"]:
        print(
            f"Mid-step prediction saved next to the final image in {out_dir}."
        )


if __name__ == "__main__":
    main()
