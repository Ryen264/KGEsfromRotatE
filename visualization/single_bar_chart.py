import matplotlib.pyplot as plt
import numpy as np
import os
import re
from datetime import datetime
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class GammaPoint:
    name: str           # Label displayed on the X-axis (e.g., "0" or "1")
    uniform_gamma_q: int    # 0 or 1
    uniform_gamma_y: int   # 0 or 1
    uniform_gamma_e: int   # 0 or 1 (X-axis value)
    MRR: float          # Validation metric (Y-axis value)


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT, path)


def build_output_path(output_dir, title):
    safe_title = re.sub(r'[^\w\-]+', '_', title.strip()).strip('_')
    filename = '{}_{}.png'.format(safe_title, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    return os.path.join(resolve_path(output_dir), filename)


def draw_chart(points: list[GammaPoint],
               title: str,
               y_axis: str, 
               y_color: str, 
               y_title: str,
               output_dir: str = "visualization/outputs/charts", 
               show_values: bool = True):
    """
    Draws a single bar chart for a subset of gamma configurations.
    """
    try:
        labels = [p.name for p in points]
        y_vals = [getattr(p, y_axis) for p in points]
    except AttributeError as e:
        print(f"Error: The provided axis attribute does not exist. {e}")
        return

    x = np.arange(len(labels))
    width = 0.4  # Clean width for isolated single bars

    fig, ax = plt.subplots(figsize=(8, 6))

    rects = ax.bar(
        x, y_vals, width,
        label=y_title, color=y_color, alpha=0.7, edgecolor='black', linewidth=1.2
    )

    ax.set_xlabel('Entity Gamma', fontsize=12, labelpad=10)
    ax.set_ylabel(y_title, color=y_color, fontsize=12)
    ax.tick_params(axis='y', labelcolor=y_color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)

    # Automatically set a clean Y-axis limit with some padding above the max value
    if y_vals:
        ax.set_ylim(0, max(y_vals) * 1.15)

    if show_values:
        # Added explicit float formatting (%.4f) to fit standard validation MRR representation
        ax.bar_label(rects, padding=5, color=y_color, fontweight='bold', fontsize=10, fmt='%.4f')

    plt.title(title, fontsize=13, pad=20, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()

    output_path = build_output_path(output_dir, title)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print('Chart saved to {}'.format(output_path))

    plt.show()
    plt.close()


if __name__ == "__main__":
    # Example dataset mapping all 8 possible combinations of your uniformity gammas
    # Replace these dummy MRR values with your actual experiment log metrics
    gamma_experimental_results = [
        # Case (Query=0, Target=0)
        GammaPoint(name="0", uniform_gamma_q=0, uniform_gamma_y=0, uniform_gamma_e=0, MRR=0.0088),
        GammaPoint(name="1", uniform_gamma_q=0, uniform_gamma_y=0, uniform_gamma_e=1, MRR=0.4650),
        
        # Case (Query=0, Target=1)
        GammaPoint(name="0", uniform_gamma_q=0, uniform_gamma_y=1, uniform_gamma_e=0, MRR=0.4551),
        GammaPoint(name="1", uniform_gamma_q=0, uniform_gamma_y=1, uniform_gamma_e=1, MRR=0.4636),
        
        # Case (Query=1, Target=0)
        GammaPoint(name="0", uniform_gamma_q=1, uniform_gamma_y=0, uniform_gamma_e=0, MRR=0.4573),
        GammaPoint(name="1", uniform_gamma_q=1, uniform_gamma_y=0, uniform_gamma_e=1, MRR=0.4642),

        # Case (Query=1, Target=1)
        GammaPoint(name="0", uniform_gamma_q=1, uniform_gamma_y=1, uniform_gamma_e=0, MRR=0.4555),
        GammaPoint(name="1", uniform_gamma_q=1, uniform_gamma_y=1, uniform_gamma_e=1, MRR=0.4658),
    ]
    
    # Define the 4 target combinations of (uniform_gamma_q, uniform_gamma_y)
    target_cases = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    # Palette configuration to distinguish the 4 charts visually
    colors = ["crimson", "teal", "royalblue", "darkorchid"]

    for idx, (uniform_gamma_q, uniform_gamma_y) in enumerate(target_cases):
        # 1. Filter points matching the precise (Query, Target) background context
        filtered_points = [
            p for p in gamma_experimental_results 
            if p.uniform_gamma_q == uniform_gamma_q and p.uniform_gamma_y == uniform_gamma_y
        ]
        
        # 2. Sort by entity gamma to guarantee chronological 0 -> 1 order on X axis
        filtered_points.sort(key=lambda p: p.uniform_gamma_e)
        
        # 3. Construct descriptive chart names
        chart_title = f"Query Gamma={uniform_gamma_q}, Target Gamma={uniform_gamma_y}"
        
        # 4. Render and commit image to disk
        draw_chart(
            points=filtered_points,
            title=chart_title,
            y_axis="MRR", 
            y_color=colors[idx], 
            y_title="MRR",
            output_dir="visualization/outputs/charts", 
            show_values=True
        )