#!/usr/bin/env python3
"""Plot reconstruction error against simple cluster ADC summary features."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


DEFAULT_CLUSTERS_CSV = Path("results/clusters_5x5_event74_layer15.csv")
DEFAULT_ERRORS_CSV = Path("results/cluster_ae_reconstruction_errors.csv")
DEFAULT_OUTPUT_DIR = Path("results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot reconstruction error versus cluster ADC summary features."
    )
    parser.add_argument("--clusters-csv", type=Path, default=DEFAULT_CLUSTERS_CSV)
    parser.add_argument("--errors-csv", type=Path, default=DEFAULT_ERRORS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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


def read_errors(errors_csv: Path) -> dict[str, float]:
    if not errors_csv.exists():
        raise FileNotFoundError(f"Input file not found: {errors_csv}")

    with errors_csv.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return {
            row["cluster_id"]: float(row["reconstruction_mse"])
            for row in reader
            if row.get("cluster_id") and row.get("reconstruction_mse")
        }


def read_cluster_stats(
    clusters_csv: Path, errors_by_cluster_id: dict[str, float]
) -> tuple[list[float], list[float], list[float], list[int]]:
    if not clusters_csv.exists():
        raise FileNotFoundError(f"Input file not found: {clusters_csv}")

    reconstruction_errors: list[float] = []
    max_adcs: list[float] = []
    mean_adcs: list[float] = []
    nonzero_counts: list[int] = []

    with clusters_csv.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        patch_columns = sorted(
            [name for name in fieldnames if name.startswith("p") and name[1:].isdigit()],
            key=patch_column_key,
        )
        if not patch_columns:
            raise ValueError("No patch pixel columns found")

        for row in reader:
            cluster_id = row.get("cluster_id", "")
            if cluster_id not in errors_by_cluster_id:
                continue

            values = [float(row[column]) for column in patch_columns]
            reconstruction_errors.append(errors_by_cluster_id[cluster_id])
            max_adcs.append(max(values))
            mean_adcs.append(sum(values) / len(values))
            nonzero_counts.append(sum(1 for value in values if value > 0))

    if not reconstruction_errors:
        raise ValueError("No clusters could be matched with reconstruction errors")

    return reconstruction_errors, max_adcs, mean_adcs, nonzero_counts


def save_scatter(
    plt,
    x_values: list[float] | list[int],
    y_values: list[float],
    x_label: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.scatter(x_values, y_values, s=12, alpha=0.55, edgecolors="none")
    ax.set_xlabel(x_label)
    ax.set_ylabel("reconstruction error (MSE)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plt = import_matplotlib()
    errors_by_cluster_id = read_errors(args.errors_csv)
    reconstruction_errors, max_adcs, mean_adcs, nonzero_counts = read_cluster_stats(
        args.clusters_csv, errors_by_cluster_id
    )

    save_scatter(
        plt,
        max_adcs,
        reconstruction_errors,
        "max ADC",
        args.output_dir / "error_vs_max_adc.png",
    )
    save_scatter(
        plt,
        mean_adcs,
        reconstruction_errors,
        "mean ADC",
        args.output_dir / "error_vs_mean_adc.png",
    )
    save_scatter(
        plt,
        nonzero_counts,
        reconstruction_errors,
        "number of pixels with ADC > 0",
        args.output_dir / "error_vs_nonzero_pixels.png",
    )

    print(f"Plotted {len(reconstruction_errors)} clusters")
    print(f"Saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
