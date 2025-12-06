"""Simple script to generate an image and save partial denoised previews."""

import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
import numpy as np
from pathlib import Path


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    print("Loading pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "/home/zchengay/Model/sdxl-base",
        torch_dtype=dtype,
        add_watermarker=False,
    )
    pipe.to(device)

    out_dir = Path("/home/zchengay/CV/MUSE/denoise_previews")
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt = "a beautiful sunset over mountains, detailed, 4k"
    num_steps = 30

    # Callback to save intermediate images
    def save_intermediate(pipe_obj, step, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        
        # Print latent stats
        print(f"Step {step}: latents range [{latents.min().item():.3f}, {latents.max().item():.3f}], mean={latents.mean().item():.3f}")
        
        # Decode latents to image
        with torch.no_grad():
            # SDXL VAE scaling factor
            image = pipe_obj.vae.decode(latents / pipe_obj.vae.config.scaling_factor, return_dict=False)[0]
        
        # Print decoded image stats before normalization
        print(f"  Decoded range [{image.min().item():.3f}, {image.max().item():.3f}]")
        
        # Convert to PIL
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype(np.uint8)
        pil_image = Image.fromarray(image[0])
        
        # Save at every 3 steps
        if step % 3 == 0 or step == 29:
            save_path = out_dir / f"step_{step:02d}.png"
            pil_image.save(save_path)
            print(f"  -> Saved to {save_path}")
        
        return callback_kwargs

    print(f"Generating image with prompt: '{prompt}'")
    generator = torch.Generator(device=device).manual_seed(42)
    
    result = pipe(
        prompt=prompt,
        num_inference_steps=num_steps,
        guidance_scale=7.5,
        generator=generator,
        callback_on_step_end=save_intermediate,
        callback_on_step_end_tensor_inputs=["latents"],
    )

    # Save final image
    final_path = out_dir / "final.png"
    result.images[0].save(final_path)
    print(f"Saved final image to {final_path}")
    print(f"\nAll images saved to {out_dir}")


if __name__ == "__main__":
    main()
