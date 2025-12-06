import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def save_generated_images(images, boxes, phrases, save_path, save_img_name, prompt, width, height):
    os.makedirs(save_path, exist_ok=True)
    font_path = 'adapter_modules/DejaVuSerif.ttf'
    font = ImageFont.truetype(font_path , 30)
    for i, image in enumerate(images):
        image.save(os.path.join(save_path, f"{save_img_name}_{i}.jpg"))
        draw = ImageDraw.Draw(image)
        for obj_i, bbox in enumerate(boxes[0]):
            color = np.concatenate([255.0 * np.random.random(3),], axis=0).astype(np.uint8)
            color = tuple(color)
            x1,y1,x2,y2 = bbox
            x1,x2 = int(x1*width),int(x2*width)
            y1,y2 = int(y1*height),int(y2*height)
            draw.rectangle([x1,y1,x2,y2], outline= color )
            draw.text( (x1,y1), phrases[0][obj_i] , fill = color, font= font)
        name = save_img_name if save_img_name is not None else prompt
        image.save(os.path.join(save_path, f"anno_{name}_{i}.jpg"))


@torch.no_grad()
def decode_latents_to_image(vae, latents):
    """Directly decode latents to PIL image without any x0 prediction."""
    scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
    scaled_latents = latents / scaling_factor
    decoded = vae.decode(scaled_latents).sample
    images = (decoded / 2 + 0.5).clamp(0, 1)
    
    images_np = images.detach().cpu().float().permute(0, 2, 3, 1).numpy()
    images_np = (images_np * 255.0).round().clip(0, 255).astype(np.uint8)
    pil_images = [Image.fromarray(img) for img in images_np]
    
    return pil_images[0] if len(pil_images) == 1 else pil_images


@torch.no_grad()
def debug_save_latents(vae, latents, step, save_dir="."):
    """Save decoded latents at any step for debugging."""
    os.makedirs(save_dir, exist_ok=True)
    pil_image = decode_latents_to_image(vae, latents)
    
    image_to_save = pil_image if isinstance(pil_image, Image.Image) else pil_image[0]
    save_path = os.path.join(save_dir, f"latents_step{step}.png")
    image_to_save.save(save_path)
    print(f"Saved latents decode at step {step} to {save_path}")
    return save_path


@torch.no_grad()
def predict_clean_image_from_velocity(
    model,
    vae,
    x_t,
    t_idx,
    sigmas,
    cond,
    prediction_type="v_prediction",
    timesteps=None,
):
    """Estimate the clean latent/image using the scheduler's prediction type."""
    if not 0 <= t_idx < len(sigmas):
        raise ValueError(f"t_idx {t_idx} out of range for sigmas of length {len(sigmas)}")

    sigma_t = sigmas[t_idx]
    if not isinstance(sigma_t, torch.Tensor):
        sigma_t = torch.tensor(sigma_t, device=x_t.device, dtype=x_t.dtype)
    else:
        sigma_t = sigma_t.to(x_t.device, x_t.dtype)

    if timesteps is None:
        raise ValueError("timesteps tensor is required to run the UNet call for debugging")

    timestep_index = min(t_idx, len(timesteps) - 1)
    timestep_value = timesteps[timestep_index]

    model_kwargs = cond if cond is not None else {}
    sigma_sq = sigma_t * sigma_t
    model_input = x_t / torch.sqrt(sigma_sq + 1.0)

    model_out = model(model_input, timestep_value, **model_kwargs)
    model_pred = model_out.sample if hasattr(model_out, "sample") else model_out

    # Compute predicted x0 based on the scheduler's prediction_type.
    # For epsilon prediction: x0 = (sample - sigma * noise_pred) but we need to account for
    # the fact that model_input was scaled, so the formula becomes:
    # x0 = sample - sigma_hat * model_pred where sigma_hat accounts for scaling
    if prediction_type in {"original_sample", "sample"}:
        x0_hat = model_pred
    elif prediction_type == "epsilon":
        # The model predicts noise (epsilon). The x0 estimate is:
        # x0 = sample - sigma * epsilon
        # But sample here is the unscaled x_t, so:
        x0_hat = x_t - sigma_t * model_pred
    elif prediction_type == "v_prediction":
        # v = alpha_t * epsilon - sigma_t * x0
        # Solving for x0: x0 = (sample / (sigma^2 + 1)) - sigma * v / sqrt(sigma^2 + 1)
        alpha_t = 1.0 / torch.sqrt(sigma_sq + 1.0)
        x0_hat = alpha_t * x_t - sigma_t * alpha_t * model_pred
    else:
        raise ValueError(
            f"Unsupported prediction_type '{prediction_type}'. Expected 'epsilon', 'v_prediction', or 'sample'."
        )

    scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
    latents = x0_hat / scaling_factor
    decoded = vae.decode(latents).sample
    images = (decoded / 2 + 0.5).clamp(0, 1)

    images_np = images.detach().cpu().permute(0, 2, 3, 1).numpy()
    images_np = (images_np * 255.0).round().clip(0, 255).astype(np.uint8)
    pil_images = [Image.fromarray(img) for img in images_np]

    pil_result = pil_images[0] if len(pil_images) == 1 else pil_images
    return x0_hat, pil_result


def debug_predict_x0(
    model,
    vae,
    x_t,
    t_idx,
    sigmas,
    cond,
    save_dir=".",
    prediction_type="v_prediction",
    timesteps=None,
):
    """Utility helper that runs prediction and saves the decoded image for inspection."""
    os.makedirs(save_dir, exist_ok=True)
    x0_hat, pil_image = predict_clean_image_from_velocity(
        model,
        vae,
        x_t,
        t_idx,
        sigmas,
        cond,
        prediction_type=prediction_type,
        timesteps=timesteps,
    )

    image_to_save = pil_image if isinstance(pil_image, Image.Image) else pil_image[0]
    save_path = os.path.join(save_dir, f"predicted_x0_step{t_idx}.png")
    image_to_save.save(save_path)
    return x0_hat, save_path

