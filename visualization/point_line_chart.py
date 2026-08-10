import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

import matplotlib.pyplot as plt

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
    uniform_t: int = 0
    peak_gpu_memory: float = 0.0
    time_per_epoch: float = 0.0

def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT, path)


def build_output_path(output_dir, title):
    safe_title = re.sub(r'[^\w\-]+', '_', title.strip()).strip('_')
    filename = '{}_{}.png'.format(safe_title, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    return os.path.join(resolve_path(output_dir), filename)


# Offset directions for labels when multiple points share the same coordinates.
LABEL_PLACEMENTS = [
    {'xytext': (10, 10), 'ha': 'left', 'va': 'bottom'},
    {'xytext': (-10, 10), 'ha': 'right', 'va': 'bottom'},
    {'xytext': (10, -10), 'ha': 'left', 'va': 'top'},
    {'xytext': (-10, -10), 'ha': 'right', 'va': 'top'},
    {'xytext': (14, 0), 'ha': 'left', 'va': 'center'},
    {'xytext': (-14, 0), 'ha': 'right', 'va': 'center'},
    {'xytext': (0, 12), 'ha': 'center', 'va': 'bottom'},
    {'xytext': (0, -12), 'ha': 'center', 'va': 'top'},
]


def label_placements_for_points(x_vals, y_vals):
    groups = defaultdict(list)
    for idx, (x_val, y_val) in enumerate(zip(x_vals, y_vals)):
        groups[(x_val, y_val)].append(idx)

    placements = [None] * len(x_vals)
    for indices in groups.values():
        for placement_idx, point_idx in enumerate(indices):
            placements[point_idx] = LABEL_PLACEMENTS[placement_idx % len(LABEL_PLACEMENTS)]
    return placements


def draw_chart(points: list[Point], x_axis: str, y_axis: str, color_axis: str,
    title: str, x_title: str="", y_title: str="", color_title: str="", color_map: str="Blues",
    output_dir: str="visualization/outputs/charts", plot_line: bool=False,
    x_scale_mode: str="linear", y_scale_mode: str="linear"):
    # Dynamically extract data based on the string arguments using getattr()
    try:
        x_vals = [getattr(p, x_axis) for p in points]
        y_vals = [getattr(p, y_axis) for p in points]
        color_vals = [getattr(p, color_axis) for p in points]
        names = [p.name for p in points]
    except AttributeError as e:
        print(f"Error: One of the provided axis or color attributes does not exist. {e}")
        return

    # Create the figure
    plt.figure(figsize=(10, 6))

    # Create a scatter plot
    # cmap='Blues' maps the lowest values to light blue and highest values to dark blue
    # s=150 sets the size of the dots, edgecolor makes them distinct
    scatter = plt.scatter(x_vals, y_vals, c=color_vals, cmap=color_map, s=150, edgecolor='black', zorder=3)

    if plot_line:
        plt.plot(x_vals, y_vals, color='black', linewidth=1)

    label_placements = label_placements_for_points(x_vals, y_vals)
    for i, name in enumerate(names):
        placement = label_placements[i]
        plt.annotate(
            name,
            (x_vals[i], y_vals[i]),
            xytext=placement['xytext'],
            textcoords='offset points',
            ha=placement['ha'],
            va=placement['va'],
            fontsize=10,
            fontweight='bold',
        )

    # Add a colorbar to explain the color mapping
    cbar = plt.colorbar(scatter)
    cbar.set_label(color_title, fontsize=12)

    # Labeling and formatting
    plt.xscale(x_scale_mode)
    plt.yscale(y_scale_mode)

    plt.xlabel(x_title, fontsize=12)
    plt.ylabel(y_title, fontsize=12)
    plt.title(title, fontsize=14, pad=15)

    # If the x-axis is 'year', force integer ticks so we don't get 2013.5
    if x_axis == "year" and x_scale_mode == "linear":
        plt.xticks(sorted(list(set(x_vals))))

    # Add a subtle grid
    plt.grid(True, linestyle='--', alpha=0.5, zorder=0)

    # Adjust layout, save, then display
    plt.tight_layout()

    output_path = build_output_path(output_dir, title)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print('Chart saved to {}'.format(output_path))

    plt.show()
    plt.close()

if __name__ == "__main__":
    # wn18rr_points = [
    #     Point(name="TransE",    year=2013, MRR=0.2260, dim=500,     batch_size=512,     negative_sample_size=1024),
    #     Point(name="DistMult",  year=2015, MRR=0.4300, dim=100,     batch_size=868,     negative_sample_size=1),
    #     Point(name="ComplEx",   year=2016, MRR=0.4400, dim=100,     batch_size=868,     negative_sample_size=1),
    #     Point(name="RotatE",    year=2019, MRR=0.4760, dim=500,     batch_size=512,     negative_sample_size=1024),
    #     Point(name="pRotatE",   year=2019, MRR=0.4620, dim=500,     batch_size=512,     negative_sample_size=1024),
    #     Point(name="TuckER",    year=2019, MRR=0.4700, dim=200,     batch_size=128,     negative_sample_size=40942),
    #     Point(name="SimKGC",    year=2022, MRR=0.6850, dim=768,     batch_size=1024,    negative_sample_size=3072),
    #     Point(name="TransERR",  year=2024, MRR=0.5010, dim=1000,    batch_size=2048,    negative_sample_size=128),
    #     Point(name="DaBR",      year=2025, MRR=0.5100, dim=500,     batch_size=100,     negative_sample_size=5)
    # ]

    # fb15k237_points = [
    #     Point(name="TransE",   year=2013, MRR=0.2940, dim=1000,     batch_size=1024,    negative_sample_size=256),
    #     Point(name="DistMult", year=2015, MRR=0.2410, dim=100,      batch_size=2721,    negative_sample_size=1),
    #     Point(name="ComplEx",  year=2016, MRR=0.2470, dim=100,      batch_size=2721,    negative_sample_size=1),
    #     Point(name="RotatE",   year=2019, MRR=0.3380, dim=1000,     batch_size=1024,    negative_sample_size=256),
    #     Point(name="pRotatE",  year=2019, MRR=0.3280, dim=1000,     batch_size=1024,    negative_sample_size=256),
    #     Point(name="TuckER",   year=2019, MRR=0.3580, dim=200,      batch_size=128,     negative_sample_size=14540),
    #     Point(name="SimKGC",   year=2022, MRR=0.3360, dim=768,      batch_size=1024,    negative_sample_size=3072),
    #     Point(name="TransERR", year=2024, MRR=0.3600, dim=1000,     batch_size=1000,    negative_sample_size=128),
    #     Point(name="DaBR",     year=2025, MRR=0.3730, dim=500,      batch_size=100,     negative_sample_size=10)
    # ]

    # uniform_t_points = [
    #     Point(name="t=2", uniform_t=2, MRR=0.3862, hit_1=0.3038),
    #     Point(name="t=3", uniform_t=3, MRR=0.4483, hit_1=0.3866),
    #     Point(name="t=4", uniform_t=4, MRR=0.4658, hit_1=0.4218),
    #     Point(name="t=5", uniform_t=5, MRR=0.4571, hit_1=0.4148),
    #     Point(name="t=6", uniform_t=6, MRR=0.4571, hit_1=0.4145)
    # ]

    # dim_points = [
    #     Point(name="d=32",  dim=32, MRR=0.4536, hit_1=0.4119),
    #     Point(name="d=64",  dim=64, MRR=0.4658, hit_1=0.4218),
    #     Point(name="d=128", dim=128, MRR=0.4678, hit_1=0.4210),
    #     Point(name="d=256", dim=256, MRR=0.4625, hit_1=0.4124),
    #     Point(name="d=500", dim=500, MRR=0.4597, hit_1=0.4075)
    # ]

    # batch_size_points = [
    #     Point(name="b=128", batch_size=128, MRR=0.4444, hit_1=0.3933),
    #     Point(name="b=256", batch_size=256, MRR=0.4545, hit_1=0.4049),
    #     Point(name="b=512", batch_size=512, MRR=0.4658, hit_1=0.4218),
    #     Point(name="b=1024", batch_size=1024, MRR=0.4695, hit_1=0.4247)
    # ]

    batch_size_points = [
        Point(name="b=128", batch_size=128, MRR=0.4459, hit_1=0.3949, peak_gpu_memory=0.12),
        Point(name="b=256", batch_size=256, MRR=0.4555, hit_1=0.4097, peak_gpu_memory=0.12),
        Point(name="b=512", batch_size=512, MRR=0.4639, hit_1=0.4205, peak_gpu_memory=0.13),
        Point(name="b=1024", batch_size=1024, MRR=0.4682, hit_1=0.4255, peak_gpu_memory=0.15),
        Point(name="b=2048", batch_size=2048, MRR=0.4704, hit_1=0.4284, peak_gpu_memory=0.23),
        Point(name="b=4096", batch_size=4096, MRR=0.4670, hit_1=0.4284, peak_gpu_memory=0.54),
        # Point(name="b=8192", batch_size=8192, MRR=0.0775, hit_1=0.0534, peak_gpu_memory=1.71),
        # Point(name="b=16384", batch_size=16384, MRR=0.0007, hit_1=0.0000, peak_gpu_memory=6.32)
    ]

    # dim_points = [
    #     Point(name="d=16", dim=16, MRR=0.4099, hit_1=0.3551, peak_gpu_memory=0.05),
    #     Point(name="d=32", dim=32, MRR=0.4538, hit_1=0.4100, peak_gpu_memory=0.08),
    #     Point(name="d=64", dim=64, MRR=0.4639, hit_1=0.4205, peak_gpu_memory=0.13),
    #     Point(name="d=128", dim=128, MRR=0.4618, hit_1=0.4142, peak_gpu_memory=0.23),
    #     Point(name="d=256", dim=256, MRR=0.4620, hit_1=0.4113, peak_gpu_memory=0.43),
    #     Point(name="d=500", dim=500, MRR=0.4581, hit_1=0.4087, peak_gpu_memory=0.81),
    #     Point(name="d=1000", dim=1000, MRR=0.4544, hit_1=0.4028, peak_gpu_memory=1.59),
    #     Point(name="d=1500", dim=1500, MRR=0.4460, hit_1=0.3926, peak_gpu_memory=2.38)
    # ]

    # draw_chart(
    #     points=wn18rr_points, 
    #     x_axis="year", x_title="Year",
    #     y_axis="dim", y_title="Embedding Dimension",
    #     color_axis="MRR", color_title="MRR", color_map="Blues",
    #     title="Overview of Embedding Dimensions of different KGE models over time on WN18RR",
    #     plot_line=False
    # )

    # draw_chart(
    #     points=fb15k237_points, 
    #     x_axis="year", x_title="Year",
    #     y_axis="dim", y_title="Embedding Dimension",
    #     color_axis="MRR", color_title="MRR", color_map="Blues",
    #     title="Overview of Embedding Dimensions of different KGE models over time on FB15k237",
    #     plot_line=False
    # )

    # draw_chart(
    #     points=uniform_t_points, 
    #     x_axis="uniform_t", x_title="Uniformity Temperature",
    #     y_axis="MRR", y_title="MRR",
    #     color_axis="hit_1", color_title="Hit@1", color_map="Blues",
    #     title="Analysis of ComplEx-AU over different Uniformity Temperatures on WN18RR",
    #     plot_line=True
    # )

    # draw_chart(
    #     points=dim_points, 
    #     x_axis="dim", x_title="Embedding Dimension",
    #     y_axis="MRR", y_title="MRR",
    #     color_axis="hit_1", color_title="Hit@1", color_map="Blues",
    #     title="Analysis of ComplEx-AU over different Embedding Dimensions on WN18RR",
    #     plot_line=True
    # )

    draw_chart(
        points=batch_size_points, 
        x_axis="batch_size", x_title="Batch Size",
        y_axis="MRR", y_title="MRR",
        color_axis="hit_1", color_title="Hit@1", color_map="Blues",
        title="Analysis of ComplEx-AU over different Batch Sizes on WN18RR",
        plot_line=True,
        x_scale_mode="log",
    )

    # draw_chart(
    #     points=wn18rr_points,
    #     x_axis="batch_size", x_title="Batch Size (log scale)",
    #     y_axis="negative_sample_size", y_title="Negative Sample Size (log scale)",
    #     color_axis="MRR", color_title="MRR", color_map="Blues",
    #     title="Overview of Batch Sizes and Negative Sample Sizes in of different KGE models on WN18RR",
    #     plot_line=False,
    #     x_scale_mode="log", y_scale_mode="log",
    # )

    # draw_chart(
    #     points=fb15k237_points,
    #     x_axis="batch_size", x_title="Batch Size (log scale)",
    #     y_axis="negative_sample_size", y_title="Negative Sample Size (log scale)",
    #     color_axis="MRR", color_title="MRR", color_map="Blues",
    #     title="Overview of Batch Sizes and Negative Sample Sizes of different KGE models on FB15k237",
    #     plot_line=False,
    #     x_scale_mode="log", y_scale_mode="log",
    # )
