#!/usr/bin/env python3
"""Analyze autoencoder reconstruction error against cluster-level features."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path


DEFAULT_ERRORS_CSV = Path("results/cluster_ae_reconstruction_errors.csv")
DEFAULT_CLUSTERS_CSV = Path("results/clusters_5x5_event74_layer15.csv")
DEFAULT_OUTPUT_DIR = Path("results")
DEFAULT_ANALYSIS_CSV = Path("results/cluster_error_feature_analysis.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge cluster features with reconstruction errors and plot relationships."
    )
    parser.add_argument("--errors-csv", type=Path, default=DEFAULT_ERRORS_CSV)
    parser.add_argument("--clusters-csv", type=Path, default=DEFAULT_CLUSTERS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--analysis-csv", type=Path, default=DEFAULT_ANALYSIS_CSV)
    return parser.parse_args()


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


def patch_column_key(column: str) -> tuple[int, int]:
    return int(column[1:-1]), int(column[-1])


def read_errors(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return {
            row["cluster_id"]: float(row["reconstruction_mse"])
            for row in reader
            if row.get("cluster_id") and row.get("reconstruction_mse")
        }


def count_local_peaks(values: list[float], patch_size: int, min_fraction_of_max: float = 0.35) -> int:
    max_adc = max(values)
    if max_adc <= 0:
        return 0

    threshold = max_adc * min_fraction_of_max
    peaks = 0

    for row in range(patch_size):
        for col in range(patch_size):
            index = row * patch_size + col
            value = values[index]
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
                        neighbor = values[neighbor_row * patch_size + neighbor_col]
                        if value <= neighbor:
                            is_peak = False
                            break
                if not is_peak:
                    break

            if is_peak:
                peaks += 1

    return peaks


def build_feature_table(clusters_csv: Path, errors_by_cluster_id: dict[str, float]) -> list[dict[str, str | float | int]]:
    if not clusters_csv.exists():
        raise FileNotFoundError(f"Input file not found: {clusters_csv}")

    rows: list[dict[str, str | float | int]] = []

    with clusters_csv.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        patch_columns = sorted(
            [name for name in fieldnames if name.startswith("p") and name[1:].isdigit()],
            key=patch_column_key,
        )
        patch_size = int(math.sqrt(len(patch_columns)))
        if patch_size * patch_size != len(patch_columns):
            raise ValueError("Patch columns do not form a square image")

        for row in reader:
            cluster_id = row.get("cluster_id", "")
            if cluster_id not in errors_by_cluster_id:
                continue

            values = [float(row[column]) for column in patch_columns]
            total_adc = sum(values)
            feature_row: dict[str, str | float | int] = {
                "cluster_id": cluster_id,
                "reconstruction_mse": errors_by_cluster_id[cluster_id],
                "max_adc": max(values),
                "mean_adc": total_adc / len(values),
                "nonzero_pixels": sum(1 for value in values if value > 0),
                "total_adc": total_adc,
                "peak_count": count_local_peaks(values, patch_size),
            }

            for metadata_column in ("event", "layer", "center_phi", "center_tbin", "center_adc"):
                if metadata_column in row:
                    feature_row[metadata_column] = row[metadata_column]

            rows.append(feature_row)

    if not rows:
        raise ValueError("No clusters could be matched with reconstruction errors")

    return rows


def save_feature_table(path: Path, rows: list[dict[str, str | float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_scatter_plot(plt, rows: list[dict[str, str | float | int]], x_key: str, x_label: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x_values = [float(row[x_key]) for row in rows]
    y_values = [float(row["reconstruction_mse"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.scatter(x_values, y_values, s=12, alpha=0.55, edgecolors="none")
    ax.set_xlabel(x_label)
    ax.set_ylabel("reconstruction error (MSE)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plt = import_matplotlib()

    errors_by_cluster_id = read_errors(args.errors_csv)
    rows = build_feature_table(args.clusters_csv, errors_by_cluster_id)
    save_feature_table(args.analysis_csv, rows)

    save_scatter_plot(
        plt,
        rows,
        "max_adc",
        "max ADC",
        args.output_dir / "reconstruction_error_vs_max_adc.png",
    )
    save_scatter_plot(
        plt,
        rows,
        "mean_adc",
        "mean ADC",
        args.output_dir / "reconstruction_error_vs_mean_adc.png",
    )
    save_scatter_plot(
        plt,
        rows,
        "nonzero_pixels",
        "number of nonzero pixels",
        args.output_dir / "reconstruction_error_vs_nonzero_pixels.png",
    )
    save_scatter_plot(
        plt,
        rows,
        "peak_count",
        "peak count",
        args.output_dir / "reconstruction_error_vs_peak_count.png",
    )

    print(f"Saved merged feature table to {args.analysis_csv}")
    print(f"Saved 4 reconstruction error feature plots to {args.output_dir}")


if __name__ == "__main__":
    main()
