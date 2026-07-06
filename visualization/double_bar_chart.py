import matplotlib.pyplot as plt
import numpy as np
import os
import re
from datetime import datetime
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Point:
    name: str
    year: int = 0
    MRR: float = 0.0
    hit_1: float = 0.0
    hit_3: float = 0.0
    hit_10: float = 0.0
    dim: int = 0
    batch_size: int = 0
    negative_sample_size: int = 0
    time_per_epoch: int=0
    peak_gpu_memory: int=0

def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT, path)

def build_output_path(output_dir, title):
    safe_title = re.sub(r'[^\w\-]+', '_', title.strip()).strip('_')
    filename = '{}_{}.png'.format(safe_title, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    return os.path.join(resolve_path(output_dir), filename)

def draw_chart(points: list[Point],
    title: str,
    left_y_axis: str, left_y_color: str, left_y_title: str,
    right_y_axis: str, right_y_color: str, right_y_title: str,
    output_dir: str="visualization/outputs/charts", show_values: bool=True,
    legend_loc: str="lower left"):
    try:
        labels = [p.name for p in points]
        left_vals = [getattr(p, left_y_axis) for p in points]
        right_vals = [getattr(p, right_y_axis) for p in points]
    except AttributeError as e:
        print(f"Error: One of the provided axis attributes does not exist. {e}")
        return

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 7))

    rects1 = ax1.bar(
        x - width / 2, left_vals, width,
        label=left_y_title, color=left_y_color, alpha=0.7,
    )

    ax1.set_xlabel('Loss Functions', fontsize=12, labelpad=10)
    ax1.set_ylabel(left_y_title, color=left_y_color, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=left_y_color)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha='right')

    ax2 = ax1.twinx()

    rects2 = ax2.bar(
        x + width / 2, right_vals, width,
        label=right_y_title, color=right_y_color, alpha=0.7,
    )

    ax2.set_ylabel(right_y_title, color=right_y_color, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=right_y_color)

    if show_values:
        ax1.bar_label(rects1, padding=3, color=left_y_color, fontweight='bold', fontsize=10)
        ax2.bar_label(rects2, padding=3, color=right_y_color, fontweight='bold', fontsize=10)

    plt.title(title, fontsize=14, pad=20, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)

    lines1, legend_labels1 = ax1.get_legend_handles_labels()
    lines2, legend_labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2, legend_labels1 + legend_labels2,
        loc=legend_loc,
    )

    plt.tight_layout()

    output_path = build_output_path(output_dir, title)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print('Chart saved to {}'.format(output_path))

    plt.show()
    plt.close()

if __name__ == "__main__":
    loss_points = [
        Point(name="SE (d=500)",      dim=500,    peak_gpu_memory=5.51, time_per_epoch=42.58),
        Point(name="SE (d=64)",       dim=64,     peak_gpu_memory=0.74, time_per_epoch=24.80),
        Point(name="MR (d=500)",      dim=500,    peak_gpu_memory=5.51, time_per_epoch=42.01),
        Point(name="MR (d=64)",       dim=64,     peak_gpu_memory=0.74, time_per_epoch=24.25),
        Point(name="Hinge (d=500)",   dim=500,    peak_gpu_memory=5.51, time_per_epoch=41.95),
        Point(name="Hinge (d=64)",    dim=64,     peak_gpu_memory=0.74, time_per_epoch=15.35),
        Point(name="BCE (d=500)",     dim=500,    peak_gpu_memory=5.51, time_per_epoch=42.02),
        Point(name="BCE (d=64)",      dim=64,     peak_gpu_memory=0.74, time_per_epoch=23.94),
        Point(name="SANS (d=500)",    dim=500,    peak_gpu_memory=5.51, time_per_epoch=42.04),
        Point(name="SANS (d=64)",     dim=64,     peak_gpu_memory=0.74, time_per_epoch=15.12),
        Point(name="BPR (d=500)",     dim=500,    peak_gpu_memory=5.51, time_per_epoch=42.02),
        Point(name="BPR (d=64)",      dim=64,     peak_gpu_memory=0.74, time_per_epoch=24.00),
        Point(name="CE (d=500)",      dim=500,    peak_gpu_memory=5.51, time_per_epoch=41.97),
        Point(name="CE (d=64)",       dim=64,     peak_gpu_memory=0.76, time_per_epoch=13.35),
        Point(name="KGAU (d=64)",     dim=64,     peak_gpu_memory=0.62, time_per_epoch=27.70)
    ]
    
    draw_chart(
        points=loss_points,
        title="Peak GPU Memory and Time per Epoch of ComplEx training on different Loss Functions",
        left_y_axis="peak_gpu_memory", left_y_color="blue", left_y_title="Peak GPU Memory (GB)",
        right_y_axis="time_per_epoch", right_y_color="green", right_y_title="Time per Epoch (s)",
        output_dir="visualization/outputs/charts", show_values=True,
        legend_loc="lower left",
    )
