#!/usr/bin/env python3
"""Classify high-error clusters using simple 5x5 patch shape features."""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import Counter
from pathlib import Path


DEFAULT_ANOMALOUS_CSV = Path("results/top_anomalous_clusters.csv")
DEFAULT_CLUSTERS_CSV = Path("results/clusters_5x5_event74_layer15.csv")
DEFAULT_OUTPUT_CSV = Path("results/anomalous_cluster_classification.csv")
DEFAULT_BAR_PLOT = Path("results/anomalous_cluster_type_bar.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify anomalous cluster patches into interpretable shape categories."
    )
    parser.add_argument("--anomalous-csv", type=Path, default=DEFAULT_ANOMALOUS_CSV)
    parser.add_argument("--clusters-csv", type=Path, default=DEFAULT_CLUSTERS_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--bar-plot", type=Path, default=DEFAULT_BAR_PLOT)
    parser.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="Classify only the first N anomalous rows. Use 0 to classify all rows.",
    )
    args = parser.parse_args()
    if args.top_n < 0:
        parser.error("--top-n must be non-negative")
    return args


def import_matplotlib():
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(Path("results/.matplotlib")))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        package = exc.name or "required package"
        raise SystemExit(f"Missing dependency: {package}. Install matplotlib before running.") from exc

    return plt


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader), reader.fieldnames or []


def patch_column_key(column: str) -> tuple[int, int]:
    return int(column[1:-1]), int(column[-1])


def get_patch_columns(fieldnames: list[str]) -> tuple[list[str], int]:
    patch_columns = sorted(
        [name for name in fieldnames if name.startswith("p") and name[1:].isdigit()],
        key=patch_column_key,
    )
    patch_size = int(math.sqrt(len(patch_columns)))
    if patch_size * patch_size != len(patch_columns):
        raise ValueError("Patch columns do not form a square image")
    return patch_columns, patch_size


def merge_with_cluster_metadata(
    anomalous_rows: list[dict[str, str]], cluster_rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    # The anomalous file already contains metadata in this project, but this merge
    # keeps the script robust if a future anomalous file only stores ids/errors.
    clusters_by_id = {row["cluster_id"]: row for row in cluster_rows if row.get("cluster_id")}
    merged_rows: list[dict[str, str]] = []

    for row in anomalous_rows:
        cluster_id = row.get("cluster_id", "")
        if cluster_id not in clusters_by_id:
            continue
        merged = dict(clusters_by_id[cluster_id])
        merged.update(row)
        merged_rows.append(merged)

    if not merged_rows:
        raise ValueError("No anomalous rows could be matched to cluster metadata")

    return merged_rows


def patch_matrix(row: dict[str, str], patch_columns: list[str], patch_size: int) -> list[list[float]]:
    values = [float(row[column]) for column in patch_columns]
    return [
        values[start : start + patch_size]
        for start in range(0, patch_size * patch_size, patch_size)
    ]


def local_peak_count(matrix: list[list[float]], min_fraction_of_max: float = 0.35) -> int:
    # Count strict 8-neighborhood local maxima above a fraction of the patch max.
    patch_size = len(matrix)
    max_adc = max(max(row) for row in matrix)
    if max_adc <= 0:
        return 0

    threshold = max_adc * min_fraction_of_max
    peaks = 0
    for row in range(patch_size):
        for col in range(patch_size):
            value = matrix[row][col]
            if value < threshold:
                continue

            is_peak = True
            for drow in (-1, 0, 1):
                for dcol in (-1, 0, 1):
                    if drow == 0 and dcol == 0:
                        continue
                    neighbor_row = row + drow
                    neighbor_col = col + dcol
                    if 0 <= neighbor_row < patch_size and 0 <= neighbor_col < patch_size:
                        if value <= matrix[neighbor_row][neighbor_col]:
                            is_peak = False
                            break
                if not is_peak:
                    break

            if is_peak:
                peaks += 1

    return peaks


def weighted_spread(matrix: list[list[float]]) -> tuple[float, float]:
    # Weighted RMS spread in x/y using ADC as the weight.
    total_adc = sum(sum(row) for row in matrix)
    if total_adc <= 0:
        return 0.0, 0.0

    weighted_row = 0.0
    weighted_col = 0.0
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            weighted_row += row_index * value
            weighted_col += col_index * value

    center_row = weighted_row / total_adc
    center_col = weighted_col / total_adc
    row_variance = 0.0
    col_variance = 0.0

    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            row_variance += ((row_index - center_row) ** 2) * value
            col_variance += ((col_index - center_col) ** 2) * value

    vertical_spread = math.sqrt(row_variance / total_adc)
    horizontal_spread = math.sqrt(col_variance / total_adc)
    return horizontal_spread, vertical_spread


def max_pixel_is_near_edge(matrix: list[list[float]]) -> bool:
    patch_size = len(matrix)
    max_adc = max(max(row) for row in matrix)
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            if value == max_adc:
                return (
                    row_index == 0
                    or col_index == 0
                    or row_index == patch_size - 1
                    or col_index == patch_size - 1
                )
    return False


def classify_cluster(
    peak_count: int,
    max_near_edge: bool,
    horizontal_spread: float,
    vertical_spread: float,
    nonzero_pixels: int,
) -> str:
    # Rule order matters: edge maxima are first because truncated patches are
    # physically different from central single/multi-peak shapes.
    if max_near_edge:
        return "edge_cluster"
    if peak_count >= 2:
        return "multi_peak"
    if max(horizontal_spread, vertical_spread) >= 1.35 and abs(horizontal_spread - vertical_spread) >= 0.45:
        return "elongated"
    if nonzero_pixels >= 12:
        return "diffuse_cluster"
    return "single_peak"


def classify_rows(
    rows: list[dict[str, str]], patch_columns: list[str], patch_size: int
) -> list[dict[str, str | int | float | bool]]:
    classified: list[dict[str, str | int | float | bool]] = []

    for row in rows:
        matrix = patch_matrix(row, patch_columns, patch_size)
        values = [value for matrix_row in matrix for value in matrix_row]
        max_adc = max(values)
        mean_adc = sum(values) / len(values)
        nonzero_pixels = sum(1 for value in values if value > 0)
        peak_count = local_peak_count(matrix)
        max_near_edge = max_pixel_is_near_edge(matrix)
        horizontal_spread, vertical_spread = weighted_spread(matrix)
        cluster_type = classify_cluster(
            peak_count,
            max_near_edge,
            horizontal_spread,
            vertical_spread,
            nonzero_pixels,
        )

        classified_row = dict(row)
        classified_row.update(
            {
                "max_adc": max_adc,
                "mean_adc": mean_adc,
                "nonzero_pixels": nonzero_pixels,
                "peak_count": peak_count,
                "max_pixel_near_edge": max_near_edge,
                "horizontal_spread": horizontal_spread,
                "vertical_spread": vertical_spread,
                "cluster_type": cluster_type,
            }
        )
        classified.append(classified_row)

    return classified


def save_classification(output_csv: Path, rows: list[dict[str, str | int | float | bool]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    preferred_fields = [
        "cluster_id",
        "reconstruction_mse",
        "cluster_type",
        "max_adc",
        "mean_adc",
        "nonzero_pixels",
        "peak_count",
        "max_pixel_near_edge",
        "horizontal_spread",
        "vertical_spread",
    ]
    remaining_fields = [field for field in rows[0].keys() if field not in preferred_fields]
    fieldnames = preferred_fields + remaining_fields

    with output_csv.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_type_bar_plot(plt, output_path: Path, type_counts: Counter[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["single_peak", "multi_peak", "elongated", "edge_cluster", "diffuse_cluster"]
    counts = [type_counts.get(label, 0) for label in labels]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(labels, counts, color="#2f80ed")
    ax.set_title("Anomalous Cluster Type Counts")
    ax.set_xlabel("cluster type")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count), ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def print_summary(type_counts: Counter[str], total: int) -> None:
    print("\nCluster type summary")
    print("--------------------")
    print(f"{'type':<18} {'count':>8} {'percent':>10}")
    for cluster_type, count in type_counts.most_common():
        percent = 100 * count / total
        print(f"{cluster_type:<18} {count:>8} {percent:>9.1f}%")


def main() -> None:
    args = parse_args()
    plt = import_matplotlib()

    anomalous_rows, anomalous_fields = read_rows(args.anomalous_csv)
    cluster_rows, cluster_fields = read_rows(args.clusters_csv)
    if args.top_n:
        anomalous_rows = anomalous_rows[: args.top_n]

    # Prefer patch columns from the anomalous file if present; otherwise use the
    # full cluster file after merging metadata.
    source_fields = anomalous_fields if any(field.startswith("p") for field in anomalous_fields) else cluster_fields
    patch_columns, patch_size = get_patch_columns(source_fields)
    merged_rows = merge_with_cluster_metadata(anomalous_rows, cluster_rows)
    classified_rows = classify_rows(merged_rows, patch_columns, patch_size)
    type_counts = Counter(str(row["cluster_type"]) for row in classified_rows)

    save_classification(args.output_csv, classified_rows)
    save_type_bar_plot(plt, args.bar_plot, type_counts)
    print_summary(type_counts, len(classified_rows))
    print(f"\nSaved classification to {args.output_csv}")
    print(f"Saved type bar plot to {args.bar_plot}")


if __name__ == "__main__":
    main()
