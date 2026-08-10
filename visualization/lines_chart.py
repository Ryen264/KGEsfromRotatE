import os
import re
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


@dataclass
class LineSpec:
    '''One series plotted against x_axis. side is "left" or "right" (twin y-axis).'''
    y_axis: str
    color: str
    title: str = ""
    side: str = "left"


def resolve_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT, path)


def build_output_path(output_dir, title):
    safe_title = re.sub(r'[^\w\-]+', '_', title.strip()).strip('_')
    filename = '{}_{}.png'.format(safe_title, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    return os.path.join(resolve_path(output_dir), filename)


def _format_value(value):
    if isinstance(value, float):
        return f'{value:g}'
    return str(value)


# Offset directions when multiple labels collide in display space.
LABEL_PLACEMENTS = [
    {'xytext': (0, 10), 'ha': 'center', 'va': 'bottom'},
    {'xytext': (0, -10), 'ha': 'center', 'va': 'top'},
    {'xytext': (12, 0), 'ha': 'left', 'va': 'center'},
    {'xytext': (-12, 0), 'ha': 'right', 'va': 'center'},
    {'xytext': (10, 10), 'ha': 'left', 'va': 'bottom'},
    {'xytext': (-10, 10), 'ha': 'right', 'va': 'bottom'},
    {'xytext': (10, -10), 'ha': 'left', 'va': 'top'},
    {'xytext': (-10, -10), 'ha': 'right', 'va': 'top'},
]


def _display_xy(ax, x, y):
    return ax.transData.transform((x, y))


def _assign_label_placements(label_items, overlap_px=22.0):
    '''
    label_items: list of {'ax', 'x', 'y'}.
    Points close in display pixels share a collision group and get staggered placements.
    '''
    n = len(label_items)
    placements = [LABEL_PLACEMENTS[0]] * n
    if n == 0:
        return placements

    display_pts = [_display_xy(item['ax'], item['x'], item['y']) for item in label_items]
    assigned = [False] * n

    for i in range(n):
        if assigned[i]:
            continue
        group = [i]
        assigned[i] = True
        xi, yi = display_pts[i]
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            xj, yj = display_pts[j]
            if (xi - xj) ** 2 + (yi - yj) ** 2 <= overlap_px ** 2:
                group.append(j)
                assigned[j] = True
        for rank, idx in enumerate(group):
            placements[idx] = LABEL_PLACEMENTS[rank % len(LABEL_PLACEMENTS)]
    return placements


def _normalize_lines(lines):
    if not lines:
        raise ValueError('lines must contain at least one LineSpec')
    normalized = []
    for item in lines:
        if isinstance(item, LineSpec):
            spec = item
        elif isinstance(item, dict):
            spec = LineSpec(**item)
        else:
            raise TypeError('Each line must be a LineSpec or dict, got {}'.format(type(item)))
        if spec.side not in ('left', 'right'):
            raise ValueError("LineSpec.side must be 'left' or 'right', got {!r}".format(spec.side))
        if not spec.title:
            spec = LineSpec(
                y_axis=spec.y_axis, color=spec.color, title=spec.y_axis, side=spec.side,
            )
        normalized.append(spec)
    return normalized


def draw_chart(
    points: list[Point],
    x_axis: str,
    title: str,
    lines: list,
    x_title: str = "",
    left_y_title: str = "",
    right_y_title: str = "",
    output_dir: str = "visualization/outputs/charts",
    x_scale_mode: str = "linear",
    left_y_scale_mode: str = "linear",
    right_y_scale_mode: str = "linear",
    show_values: bool = True,
    legend_loc: str = "best",
):
    '''
    Plot one or more lines sharing the same x_axis.

    Lines on side="right" use a twin y-axis (needed when scales differ, e.g. MRR vs GPU GB).
    Value labels are drawn at the start of each segment (every point except the last is a
    segment start; the final point is also labeled so the last value remains visible).
    '''
    line_specs = _normalize_lines(lines)

    try:
        x_vals = [getattr(p, x_axis) for p in points]
        series = {
            spec.y_axis: [getattr(p, spec.y_axis) for p in points]
            for spec in line_specs
        }
    except AttributeError as e:
        print(f"Error: One of the provided axis attributes does not exist. {e}")
        return

    # Keep polyline order consistent along x.
    order = sorted(range(len(points)), key=lambda i: x_vals[i])
    x_vals = [x_vals[i] for i in order]
    series = {key: [vals[i] for i in order] for key, vals in series.items()}

    fig, ax_left = plt.subplots(figsize=(10, 6))
    ax_right = None
    if any(spec.side == 'right' for spec in line_specs):
        ax_right = ax_left.twinx()

    axes = {'left': ax_left, 'right': ax_right}
    handles = []
    label_items = []

    ax_left.set_xscale(x_scale_mode)
    ax_left.set_yscale(left_y_scale_mode)
    if ax_right is not None:
        ax_right.set_yscale(right_y_scale_mode)

    for spec in line_specs:
        ax = axes[spec.side]
        y_vals = series[spec.y_axis]
        line, = ax.plot(
            x_vals, y_vals,
            color=spec.color, marker='o', linewidth=2, markersize=7,
            label=spec.title, zorder=3,
        )
        handles.append(line)

        if show_values:
            for x_val, y_val in zip(x_vals, y_vals):
                label_items.append({
                    'ax': ax,
                    'x': x_val,
                    'y': y_val,
                    'text': _format_value(y_val),
                    'color': spec.color,
                })

    # Resolve display positions after scales/limits are set, then stagger collisions.
    if label_items:
        fig.canvas.draw()
        placements = _assign_label_placements(label_items)
        for item, placement in zip(label_items, placements):
            item['ax'].annotate(
                item['text'],
                (item['x'], item['y']),
                textcoords='offset points',
                xytext=placement['xytext'],
                ha=placement['ha'],
                va=placement['va'],
                fontsize=9, fontweight='bold', color=item['color'],
                zorder=4,
            )

    left_specs = [s for s in line_specs if s.side == 'left']
    right_specs = [s for s in line_specs if s.side == 'right']
    ax_left.set_xlabel(x_title, fontsize=12)
    ax_left.set_ylabel(left_y_title or ' / '.join(s.title for s in left_specs), fontsize=12)
    if left_specs:
        ax_left.tick_params(
            axis='y',
            labelcolor=left_specs[0].color if len(left_specs) == 1 else 'black',
        )
    if ax_right is not None:
        ax_right.set_ylabel(right_y_title or ' / '.join(s.title for s in right_specs), fontsize=12)
        if right_specs:
            ax_right.tick_params(
                axis='y',
                labelcolor=right_specs[0].color if len(right_specs) == 1 else 'black',
            )

    if x_axis == 'year' and x_scale_mode == 'linear':
        ax_left.set_xticks(sorted(set(x_vals)))

    ax_left.set_title(title, fontsize=14, pad=15)
    ax_left.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax_left.legend(handles=handles, loc=legend_loc, framealpha=0.95)

    fig.tight_layout()

    output_path = build_output_path(output_dir, title)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print('Chart saved to {}'.format(output_path))

    plt.show()
    plt.close(fig)


DEFAULT_LINES = [
    LineSpec(y_axis='MRR', color='crimson', title='MRR', side='left'),
    LineSpec(y_axis='peak_gpu_memory', color='#3568E8', title='Peak GPU Memory (GB)', side='right'),
]


if __name__ == "__main__":
    batch_size_points = [
        Point(name="b=128", batch_size=128, MRR=0.4459, hit_1=0.3949, peak_gpu_memory=0.12),
        Point(name="b=256", batch_size=256, MRR=0.4555, hit_1=0.4097, peak_gpu_memory=0.12),
        Point(name="b=512", batch_size=512, MRR=0.4639, hit_1=0.4205, peak_gpu_memory=0.13),
        Point(name="b=1024", batch_size=1024, MRR=0.4682, hit_1=0.4255, peak_gpu_memory=0.15),
        Point(name="b=2048", batch_size=2048, MRR=0.4704, hit_1=0.4284, peak_gpu_memory=0.23),
        Point(name="b=4096", batch_size=4096, MRR=0.4670, hit_1=0.4284, peak_gpu_memory=0.54),
        # Point(name="b=8192", batch_size=8192, MRR=0.0775, hit_1=0.0534, peak_gpu_memory=1.71),
        # Point(name="b=16384", batch_size=16384, MRR=0.0007, hit_1=0.0000, peak_gpu_memory=6.32),
    ]

    # dim_points = [
    #     Point(name="d=16", dim=16, MRR=0.4099, hit_1=0.3551, peak_gpu_memory=0.05),
    #     Point(name="d=32", dim=32, MRR=0.4538, hit_1=0.4100, peak_gpu_memory=0.08),
    #     Point(name="d=64", dim=64, MRR=0.4639, hit_1=0.4205, peak_gpu_memory=0.13),
    #     Point(name="d=128", dim=128, MRR=0.4618, hit_1=0.4142, peak_gpu_memory=0.23),
    #     Point(name="d=256", dim=256, MRR=0.4620, hit_1=0.4113, peak_gpu_memory=0.43),
    #     Point(name="d=500", dim=500, MRR=0.4581, hit_1=0.4087, peak_gpu_memory=0.81),
    #     Point(name="d=1000", dim=1000, MRR=0.4544, hit_1=0.4028, peak_gpu_memory=1.59),
    #     Point(name="d=1500", dim=1500, MRR=0.4460, hit_1=0.3926, peak_gpu_memory=2.38),
    # ]

    # draw_chart(
    #     points=dim_points,
    #     x_axis="dim", x_title="Embedding Dimension",
    #     title="Analysis of ComplEx-AU over different Embedding Dimensions on WN18RR",
    #     lines=DEFAULT_LINES,
    #     left_y_title="MRR",
    #     right_y_title="Peak GPU Memory (GB)",
    #     x_scale_mode="log",
    # )

    draw_chart(
        points=batch_size_points,
        x_axis="batch_size", x_title="Batch Size",
        title="Analysis of ComplEx-AU over different Batch Sizes on WN18RR",
        lines=DEFAULT_LINES,
        left_y_title="MRR",
        right_y_title="Peak GPU Memory (GB)",
        x_scale_mode="log",
    )
