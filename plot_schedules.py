"""Visualize different diffusion scheduler sigma schedules used with SDXL.

This script will:
- Instantiate several common schedulers from `diffusers`.
- Compute their noise scales (sigmas) over a fixed number of inference steps.
- Plot sigma vs. step for each scheduler in separate and combined graphs.

Run:
    python plot_schedules.py
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from diffusers import (
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
)


def get_schedulers(model_dir: str, num_train_timesteps: int = 1000):
    """Create a set of schedulers with consistent base config.

    We load the SDXL base scheduler config (if present) and then
    build several different scheduler classes from that config
    so they share the same beta / alpha schedule.
    """

    # Base config from SDXL model directory
    euler = EulerDiscreteScheduler.from_pretrained(model_dir, subfolder="scheduler")

    common_kwargs = dict(
        num_train_timesteps=euler.config.num_train_timesteps,
        beta_start=euler.config.beta_start,
        beta_end=euler.config.beta_end,
        beta_schedule=euler.config.beta_schedule,
        prediction_type=euler.config.prediction_type,
    )

    schedulers = {
        "EulerDiscrete": euler,
        "EulerAncestral": EulerAncestralDiscreteScheduler(**common_kwargs),
        "DDIM": DDIMScheduler(**common_kwargs),
        "DDPM": DDPMScheduler(**common_kwargs),
        "DPM-Solver++": DPMSolverMultistepScheduler(**common_kwargs),
    }

    return schedulers


def compute_sigmas(scheduler, num_inference_steps: int, device: torch.device):
    """Return the sigma schedule for a given scheduler.

    For discrete schedulers, we call `set_timesteps` and then read
    either `scheduler.sigmas` (if it exists) or derive sigmas from
    alphas/betas.
    """

    scheduler.set_timesteps(num_inference_steps, device=device)

    if hasattr(scheduler, "sigmas"):
        sigmas = scheduler.sigmas.detach().cpu().float()
        # Some schedulers append a final 0 sigma -> drop for plotting clarity
        if sigmas.numel() == num_inference_steps + 1:
            sigmas = sigmas[:-1]
        return sigmas

    # Fallback: derive sigmas from alphas_cumprod if available
    if hasattr(scheduler, "alphas_cumprod"):
        alphas_cumprod = scheduler.alphas_cumprod.detach().cpu().float()
        sigmas_full = ((1 - alphas_cumprod) / alphas_cumprod).sqrt()
        # Sample according to the scheduler's timesteps
        timesteps = scheduler.timesteps.detach().cpu().long()
        timesteps = torch.clamp(timesteps, 0, len(sigmas_full) - 1)
        sigmas = sigmas_full[timesteps]
        if sigmas.numel() > num_inference_steps:
            sigmas = sigmas[:num_inference_steps]
        return sigmas

    raise RuntimeError(f"Cannot get sigmas for scheduler type {type(scheduler)}")


def plot_schedules(model_dir: str, num_inference_steps: int = 30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    schedulers = get_schedulers(model_dir)

    out_dir = Path("scheduler_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_steps = torch.arange(num_inference_steps)
    plt.figure(figsize=(10, 6))

    for name, sched in schedulers.items():
        sigmas = compute_sigmas(sched, num_inference_steps, device)

        # Individual plot
        plt_ind = plt.figure(figsize=(8, 5))
        plt_ind.suptitle(f"Sigma schedule: {name}")
        ax_ind = plt_ind.add_subplot(111)
        ax_ind.plot(all_steps.numpy(), sigmas.numpy(), marker="o")
        ax_ind.set_xlabel("Step")
        ax_ind.set_ylabel("Sigma")
        ax_ind.grid(True, alpha=0.3)
        plt_ind.tight_layout()
        ind_path = out_dir / f"{name.replace(' ', '_')}_sigmas.png"
        plt_ind.savefig(ind_path)
        plt_ind.close()

        # Add to combined plot
        plt.plot(all_steps.numpy(), sigmas.numpy(), marker="o", label=name)

    plt.title("Comparison of sigma schedules")
    plt.xlabel("Step")
    plt.ylabel("Sigma")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    combined_path = out_dir / "all_schedulers_sigmas.png"
    plt.savefig(combined_path)
    plt.close()

    print(f"Saved individual and combined scheduler plots in {out_dir.resolve()}")


if __name__ == "__main__":
    # Adjust this path if your SDXL base checkpoint is elsewhere
    base_model_path = "/home/zchengay/Model/sdxl-base"
    if not os.path.isdir(base_model_path):
        raise SystemExit(f"Model directory not found: {base_model_path}")

    plot_schedules(base_model_path, num_inference_steps=30)
