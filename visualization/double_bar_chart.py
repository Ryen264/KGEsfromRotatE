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
    mrr: float = 0.0
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
    title: str, x_title: str,
    left_y_axis: str, left_y_color: str, left_y_title: str,
    right_y_axis: str, right_y_color: str, right_y_title: str,
    output_dir: str="visualization/outputs/charts",
    show_bar_values: bool=True, show_line_values: bool=True,
    legend_loc: str="lower left",
    show_line: bool=False,
    line_y_axis: str="", line_y_color: str="red", line_y_title: str="",
    line_on: str="left"):
    try:
        labels = [p.name for p in points]
        left_vals = [getattr(p, left_y_axis) for p in points]
        right_vals = [getattr(p, right_y_axis) for p in points]
        line_vals = [getattr(p, line_y_axis) for p in points] if show_line else None
    except AttributeError as e:
        print(f"Error: One of the provided axis attributes does not exist. {e}")
        return

    if show_line and not line_y_axis:
        print("Error: show_line=True requires line_y_axis to be set.")
        return
    if show_line and line_on not in ("left", "right"):
        print("Error: line_on must be 'left' or 'right'.")
        return

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 7))

    rects1 = ax1.bar(
        x - width / 2, left_vals, width,
        label=left_y_title, color=left_y_color, alpha=0.7,
        zorder=1,
    )

    ax1.set_xlabel(x_title, fontsize=12, labelpad=10)
    ax1.set_ylabel(left_y_title, color=left_y_color, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=left_y_color)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha='right')

    ax2 = ax1.twinx()

    rects2 = ax2.bar(
        x + width / 2, right_vals, width,
        label=right_y_title, color=right_y_color, alpha=0.7,
        zorder=1,
    )

    ax2.set_ylabel(right_y_title, color=right_y_color, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=right_y_color)

    if show_line:
        line_ax = ax1 if line_on == "left" else ax2
        line_label = line_y_title or line_y_axis
        line_ax.plot(
            x, line_vals,
            color=line_y_color, marker='o', linewidth=2, markersize=6,
            label=line_label, zorder=5,
        )
        if show_line_values:
            for xi, yi in zip(x, line_vals):
                line_ax.annotate(
                    f'{yi:g}', (xi, yi),
                    textcoords='offset points', xytext=(0, 8),
                    ha='center', color=line_y_color, fontweight='bold', fontsize=9,
                    zorder=6,
                )

    # Twin axes stack by axes zorder: put the line's axis in front of the other
    # so the line is not buried under the twin's bars.
    if show_line and line_on == "right":
        ax2.set_zorder(ax1.get_zorder() + 1)
        ax2.patch.set_visible(False)
    else:
        ax1.set_zorder(ax2.get_zorder() + 1)
        ax1.patch.set_visible(False)

    if show_bar_values:
        ax1.bar_label(rects1, padding=3, color=left_y_color, fontweight='bold', fontsize=10)
        ax2.bar_label(rects2, padding=3, color=right_y_color, fontweight='bold', fontsize=10)

    plt.title(title, fontsize=14, pad=20, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3, zorder=0)

    lines1, legend_labels1 = ax1.get_legend_handles_labels()
    lines2, legend_labels2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(
        lines1 + lines2, legend_labels1 + legend_labels2,
        loc=legend_loc,
        framealpha=0.95,
    )
    legend.set_zorder(20)

    plt.tight_layout()

    output_path = build_output_path(output_dir, title)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print('Chart saved to {}'.format(output_path))

    plt.show()
    plt.close()

if __name__ == "__main__":
    # loss_points = [
    #     Point(name="SE (d=500)",      dim=500,    peak_gpu_memory=5.51, time_per_epoch=42.58),
    #     Point(name="SE (d=64)",       dim=64,     peak_gpu_memory=0.74, time_per_epoch=24.80),
    #     Point(name="MR (d=500)",      dim=500,    peak_gpu_memory=5.51, time_per_epoch=42.01),
    #     Point(name="MR (d=64)",       dim=64,     peak_gpu_memory=0.74, time_per_epoch=24.25),
    #     Point(name="Hinge (d=500)",   dim=500,    peak_gpu_memory=5.51, time_per_epoch=41.95),
    #     Point(name="Hinge (d=64)",    dim=64,     peak_gpu_memory=0.74, time_per_epoch=15.35),
    #     Point(name="BCE (d=500)",     dim=500,    peak_gpu_memory=5.51, time_per_epoch=42.02),
    #     Point(name="BCE (d=64)",      dim=64,     peak_gpu_memory=0.74, time_per_epoch=23.94),
    #     Point(name="SANS (d=500)",    dim=500,    peak_gpu_memory=5.51, time_per_epoch=42.04),
    #     Point(name="SANS (d=64)",     dim=64,     peak_gpu_memory=0.74, time_per_epoch=15.12),
    #     Point(name="BPR (d=500)",     dim=500,    peak_gpu_memory=5.51, time_per_epoch=42.02),
    #     Point(name="BPR (d=64)",      dim=64,     peak_gpu_memory=0.74, time_per_epoch=24.00),
    #     Point(name="CE (d=500)",      dim=500,    peak_gpu_memory=5.51, time_per_epoch=41.97),
    #     Point(name="CE (d=64)",       dim=64,     peak_gpu_memory=0.76, time_per_epoch=13.35),
    #     Point(name="KGAU (d=64)",     dim=64,     peak_gpu_memory=0.62, time_per_epoch=27.70)
    # ]

    wn18rr_loss_points = [
        Point(name="Uniform (BCE, d=500)",      dim=500, peak_gpu_memory=1.85, time_per_epoch=90.0,  mrr=0.4383),
        Point(name="Bernoulli (BCE, d=500)",    dim=500, peak_gpu_memory=1.85, time_per_epoch=92.0,  mrr=0.4341),
        Point(name="SelfAdv (SANS, d=500)",     dim=500, peak_gpu_memory=1.85, time_per_epoch=92.0,  mrr=0.4433),
        Point(name="1vsAll (BCE, d=500)",       dim=500, peak_gpu_memory=1.27, time_per_epoch=60.0,  mrr=0.4363),
        Point(name="kvsAll (BCE, d=500)",       dim=500, peak_gpu_memory=1.27, time_per_epoch=89.0,  mrr=0.4390),
        Point(name="KGAU (d=500)",              dim=500, peak_gpu_memory=0.81, time_per_epoch=7.11,  mrr=0.4560),
        Point(name="KGAU (d=64)",               dim=64,  peak_gpu_memory=0.13, time_per_epoch=2.49,  mrr=0.4643),
    ]

    fb15k237_loss_points = [
        Point(name="Uniform (BCE, d=1000)",     dim=1000, peak_gpu_memory=5.37, time_per_epoch=104.0, mrr=0.2689),
        Point(name="Bernoulli (BCE, d=1000)",   dim=1000, peak_gpu_memory=5.37, time_per_epoch=102.0, mrr=0.2582),
        Point(name="SelfAdv (SANS, d=1000)",    dim=1000, peak_gpu_memory=5.37, time_per_epoch=104.0, mrr=0.3184),
        Point(name="1vsAll (BCE, d=1000)",      dim=1000, peak_gpu_memory=0.93, time_per_epoch=72.0,  mrr=0.2680),
        Point(name="KvsAll (BCE, d=1000)",      dim=1000, peak_gpu_memory=0.81, time_per_epoch=61.0,  mrr=0.2715),
        Point(name="KGAU (d=1000)",             dim=1000, peak_gpu_memory=0.68, time_per_epoch=46.46, mrr=0.3185),
        Point(name="KGAU (d=128)",              dim=128,  peak_gpu_memory=0.13, time_per_epoch=32.16, mrr=0.3080),
    ]
    
    draw_chart(
        points=wn18rr_loss_points,
        title="Peak GPU Memory and Time per Epoch of KGAU on WN18RR",
        x_title="Training Strategies",
        left_y_axis="peak_gpu_memory", left_y_color="blue", left_y_title="Peak GPU Memory (GB)",
        right_y_axis="time_per_epoch", right_y_color="green", right_y_title="Time per Epoch (s)",
        output_dir="visualization/outputs/charts",
        show_bar_values=True, show_line_values=True, show_line=True,
        line_y_axis="mrr", line_y_color="darkred", line_y_title="MRR",
        legend_loc="lower left"
    )
    
    draw_chart(
        points=fb15k237_loss_points,
        title="Peak GPU Memory and Time per Epoch of KGAU on FB15K-237",
        x_title="Training Strategies",
        left_y_axis="peak_gpu_memory", left_y_color="blue", left_y_title="Peak GPU Memory (GB)",
        right_y_axis="time_per_epoch", right_y_color="green", right_y_title="Time per Epoch (s)",
        output_dir="visualization/outputs/charts",
        show_bar_values=True, show_line_values=True, show_line=True,
        line_y_axis="mrr", line_y_color="darkred", line_y_title="MRR",
        legend_loc="lower left"
    )
