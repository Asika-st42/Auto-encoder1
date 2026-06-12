#!/usr/bin/env python3
"""Inspect extracted cluster patches and save SVG figures."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


DEFAULT_INPUT = Path("results/clusters_5x5_event74_layer15.csv")
DEFAULT_OUTPUT_DIR = Path("results")
DEFAULT_NUM_PATCHES = 12
DEFAULT_BINS = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save heatmaps and ADC summary histograms for extracted clusters."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input clusters CSV.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Figure output folder.")
    parser.add_argument(
        "--num-patches",
        type=int,
        default=DEFAULT_NUM_PATCHES,
        help="Number of high-ADC cluster patches to show in the heatmap figure.",
    )
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS, help="Number of histogram bins.")
    args = parser.parse_args()
    if args.num_patches <= 0:
        parser.error("--num-patches must be positive")
    if args.bins <= 0:
        parser.error("--bins must be positive")
    return args


def read_clusters(input_path: Path) -> tuple[list[dict[str, str]], list[str], int]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    patch_columns = sorted(
        [name for name in fieldnames if name.startswith("p") and name[1:].isdigit()],
        key=lambda name: (int(name[1:-1]), int(name[-1])),
    )
    patch_size = int(math.sqrt(len(patch_columns)))
    if patch_size * patch_size != len(patch_columns):
        raise ValueError("Patch columns do not form a square image")
    if not rows:
        raise ValueError(f"No clusters found in {input_path}")

    return rows, patch_columns, patch_size


def patch_values(row: dict[str, str], patch_columns: list[str]) -> list[float]:
    return [float(row[column]) for column in patch_columns]


def cluster_summaries(
    rows: list[dict[str, str]], patch_columns: list[str]
) -> tuple[list[float], list[float], list[int]]:
    maxima: list[float] = []
    means: list[float] = []
    positive_counts: list[int] = []

    for row in rows:
        values = patch_values(row, patch_columns)
        maxima.append(max(values))
        means.append(sum(values) / len(values))
        positive_counts.append(sum(1 for value in values if value > 0))

    return maxima, means, positive_counts


def color_for_value(value: float, max_value: float) -> str:
    if max_value <= 0:
        ratio = 0.0
    else:
        ratio = max(0.0, min(1.0, value / max_value))

    # Dark blue to yellow, chosen to work on white backgrounds.
    red = round(25 + 225 * ratio)
    green = round(58 + 178 * ratio)
    blue = round(95 - 72 * ratio)
    return f"rgb({red},{green},{blue})"


def svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = "middle") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}" fill="#1f2933">{text}</text>'
    )


def save_patch_heatmaps(
    output_path: Path,
    rows: list[dict[str, str]],
    patch_columns: list[str],
    patch_size: int,
    num_patches: int,
) -> None:
    selected_rows = sorted(
        rows,
        key=lambda row: float(row.get("center_adc", "0")),
        reverse=True,
    )[:num_patches]

    cell_size = 24
    panel_gap = 24
    label_height = 38
    cols = min(4, len(selected_rows))
    rows_count = math.ceil(len(selected_rows) / cols)
    panel_size = patch_size * cell_size
    width = cols * panel_size + (cols + 1) * panel_gap
    height = rows_count * (panel_size + label_height) + (rows_count + 1) * panel_gap + 28

    all_values = [value for row in selected_rows for value in patch_values(row, patch_columns)]
    max_value = max(all_values) if all_values else 0.0
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 24, f"Top {len(selected_rows)} cluster patches by center ADC", 16),
    ]

    for index, row in enumerate(selected_rows):
        grid_col = index % cols
        grid_row = index // cols
        x0 = panel_gap + grid_col * (panel_size + panel_gap)
        y0 = panel_gap + 24 + grid_row * (panel_size + label_height + panel_gap)
        values = patch_values(row, patch_columns)

        elements.append(
            svg_text(
                x0 + panel_size / 2,
                y0 - 6,
                f"id {row.get('cluster_id')}  adc {float(row.get('center_adc', 0)):.0f}",
                11,
            )
        )
        for patch_index, value in enumerate(values):
            row_index = patch_index // patch_size
            col_index = patch_index % patch_size
            x = x0 + col_index * cell_size
            y = y0 + row_index * cell_size
            elements.append(
                f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" '
                f'fill="{color_for_value(value, max_value)}" stroke="#ffffff" stroke-width="1"/>'
            )

    elements.append("</svg>")
    output_path.write_text("\n".join(elements) + "\n")


def histogram(values: list[float], bins: int) -> tuple[list[tuple[float, float, int]], float, float]:
    min_value = min(values)
    max_value = max(values)
    if min_value == max_value:
        return [(min_value, max_value, len(values))], min_value, max_value

    bin_width = (max_value - min_value) / bins
    counts = [0 for _ in range(bins)]
    for value in values:
        index = min(bins - 1, int((value - min_value) / bin_width))
        counts[index] += 1

    return [
        (min_value + index * bin_width, min_value + (index + 1) * bin_width, count)
        for index, count in enumerate(counts)
    ], min_value, max_value


def save_histogram(output_path: Path, values: list[float], title: str, x_label: str, bins: int) -> None:
    hist, min_value, max_value = histogram(values, bins)
    width = 760
    height = 460
    margin_left = 72
    margin_right = 32
    margin_top = 54
    margin_bottom = 72
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_count = max(count for _, _, count in hist)
    bar_width = plot_width / len(hist)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 28, title, 18),
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}" stroke="#243b53"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#243b53"/>',
    ]

    for index, (_, _, count) in enumerate(hist):
        bar_height = 0 if max_count == 0 else (count / max_count) * plot_height
        x = margin_left + index * bar_width
        y = margin_top + plot_height - bar_height
        elements.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(1.0, bar_width - 1):.2f}" '
            f'height="{bar_height:.2f}" fill="#2f80ed"/>'
        )

    elements.extend(
        [
            svg_text(margin_left, margin_top + plot_height + 24, f"{min_value:.1f}", 12),
            svg_text(width - margin_right, margin_top + plot_height + 24, f"{max_value:.1f}", 12),
            svg_text(width / 2, height - 24, x_label, 13),
            svg_text(24, margin_top + plot_height / 2, "count", 13),
            svg_text(margin_left - 10, margin_top + 4, str(max_count), 12, "end"),
            svg_text(margin_left - 10, margin_top + plot_height + 4, "0", 12, "end"),
        ]
    )

    elements.append("</svg>")
    output_path.write_text("\n".join(elements) + "\n")


def main() -> None:
    args = parse_args()
    rows, patch_columns, patch_size = read_clusters(args.input)
    maxima, means, positive_counts = cluster_summaries(rows, patch_columns)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_patch_heatmaps(
        args.output_dir / "cluster_patch_heatmaps.svg",
        rows,
        patch_columns,
        patch_size,
        args.num_patches,
    )
    save_histogram(
        args.output_dir / "cluster_max_adc_histogram.svg",
        maxima,
        "Maximum ADC per 5x5 Patch",
        "maximum ADC",
        args.bins,
    )
    save_histogram(
        args.output_dir / "cluster_mean_adc_histogram.svg",
        means,
        "Mean ADC per 5x5 Patch",
        "mean ADC",
        args.bins,
    )
    save_histogram(
        args.output_dir / "cluster_positive_pixels_histogram.svg",
        [float(value) for value in positive_counts],
        "Pixels with ADC > 0 per 5x5 Patch",
        "positive pixels",
        min(args.bins, patch_size * patch_size),
    )

    print(f"Read {len(rows)} clusters from {args.input}")
    print(f"Saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
