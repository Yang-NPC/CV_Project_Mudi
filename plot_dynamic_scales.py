"""Visualize the dynamic scale schedules defined in `MUSE_AttnProcessor`.

We replicate `_compute_dynamic_scale` from `adapter_modules/attention_processor.py`
with the assumptions mentioned:
- Total denoising steps: 30
- Base scale (provided from inference): 0.8

Running this script produces a line plot showing how the effective scale evolves
for each schedule option (static, linear, cosine, exponential, reverse_linear,
increasing, decreasing).

Usage:
    python plot_dynamic_scales.py
"""

from pathlib import Path
import math
import matplotlib.pyplot as plt

BASE_SCALE = 0.8
NUM_STEPS = 30  # assumed denoising steps
SCHEDULES = [
    "static",
    "linear",
    "cosine",
    "exponential",
    "reverse_linear",
    "increasing",
    "decreasing",
]


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def schedule_value(schedule: str, t_norm: float) -> float:
    """Match `_compute_dynamic_scale` schedule outputs."""
    if schedule == "static":
        return 0.5
    if schedule == "linear":
        return 1.0 - t_norm
    if schedule == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * t_norm))
    if schedule == "exponential":
        return math.exp(-2.0 * t_norm)
    if schedule == "reverse_linear":
        return t_norm
    if schedule == "increasing":
        return 1.0 - math.exp(-3.0 * t_norm)
    if schedule == "decreasing":
        return math.exp(-3.0 * t_norm)
    # Default fallback mirrors attention processor implementation
    return 1.0 - t_norm


def compute_dynamic_scale(step: int, schedule: str) -> float:
    """Replicate `_compute_dynamic_scale` with fixed arguments."""
    t_norm = clamp01(step / NUM_STEPS)
    sched_val = clamp01(schedule_value(schedule, t_norm))
    scale_factor = 0.75 + 0.5 * sched_val
    return BASE_SCALE * scale_factor


def plot_dynamic_scales():
    steps = list(range(NUM_STEPS))
    out_dir = Path("scheduler_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_fig = plt.figure(figsize=(10, 6))
    combined_ax = combined_fig.add_subplot(111)

    for sched in SCHEDULES:
        values = [compute_dynamic_scale(step, sched) for step in steps]

        # Individual figure per schedule
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        ax.plot(steps, values, marker="o", label=sched)
        ax.axhline(BASE_SCALE, color="black", linestyle="--", linewidth=1, label="base scale (0.8)")
        ax.set_title(f"Dynamic scale: {sched}")
        ax.set_xlabel("Denoising step index")
        ax.set_ylabel("Effective scale")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        indiv_path = out_dir / f"dynamic_scale_{sched}.png"
        fig.savefig(indiv_path)
        fig.close()

        # Add to combined plot
        combined_ax.plot(steps, values, marker="o", label=sched)

    combined_ax.axhline(BASE_SCALE, color="black", linestyle="--", linewidth=1, label="base scale (0.8)")
    combined_ax.set_title("Dynamic scale schedules (base scale = 0.8, steps = 30)")
    combined_ax.set_xlabel("Denoising step index")
    combined_ax.set_ylabel("Effective scale")
    combined_ax.grid(True, alpha=0.3)
    combined_ax.legend()
    combined_fig.tight_layout()

    output_path = out_dir / "dynamic_scale_schedules_combined.png"
    combined_fig.savefig(output_path)
    combined_fig.close()
    print(f"Saved individual and combined schedule plots to {out_dir.resolve()}")


if __name__ == "__main__":
    plot_dynamic_scales()
