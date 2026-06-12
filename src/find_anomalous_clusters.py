#!/usr/bin/env python3
"""Rank clusters by autoencoder reconstruction error and plot examples."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path


DEFAULT_CLUSTERS_CSV = Path("results/clusters_5x5_event74_layer15.csv")
DEFAULT_ERRORS_CSV = Path("results/cluster_ae_reconstruction_errors.csv")
DEFAULT_OUTPUT_DIR = Path("results")
TOP_N = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find anomalous clusters and plot high-error and low-error heatmaps."
    )
    parser.add_argument("--clusters-csv", type=Path, default=DEFAULT_CLUSTERS_CSV)
    parser.add_argument("--errors-csv", type=Path, default=DEFAULT_ERRORS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-n", type=int, default=TOP_N)
    args = parser.parse_args()

    if args.top_n <= 0:
        parser.error("--top-n must be positive")

    return args


def import_plotting_dependencies():
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


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader), reader.fieldnames or []


def merge_clusters_with_errors(
    clusters_csv: Path, errors_csv: Path
) -> tuple[list[dict[str, str]], list[str], int]:
    cluster_rows, cluster_fields = read_csv_rows(clusters_csv)
    error_rows, _ = read_csv_rows(errors_csv)

    patch_columns = sorted(
        [name for name in cluster_fields if name.startswith("p") and name[1:].isdigit()],
        key=patch_column_key,
    )
    patch_size = int(math.sqrt(len(patch_columns)))
    if patch_size * patch_size != len(patch_columns):
        raise ValueError("Patch columns do not form a square image")

    errors_by_cluster_id = {
        row["cluster_id"]: row["reconstruction_mse"]
        for row in error_rows
        if row.get("cluster_id") and row.get("reconstruction_mse")
    }

    merged_rows: list[dict[str, str]] = []
    for row in cluster_rows:
        cluster_id = row.get("cluster_id", "")
        if cluster_id not in errors_by_cluster_id:
            continue
        merged_row = dict(row)
        merged_row["reconstruction_mse"] = errors_by_cluster_id[cluster_id]
        merged_rows.append(merged_row)

    if not merged_rows:
        raise ValueError("No clusters could be matched with reconstruction errors")

    merged_rows.sort(key=lambda row: float(row["reconstruction_mse"]), reverse=True)
    return merged_rows, patch_columns, patch_size


def save_ranked_clusters(output_path: Path, rows: list[dict[str, str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    if "reconstruction_mse" in fieldnames:
        fieldnames.remove("reconstruction_mse")
    fieldnames.insert(1, "reconstruction_mse")

    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def patch_matrix(row: dict[str, str], patch_columns: list[str], patch_size: int) -> list[list[float]]:
    values = [float(row[column]) for column in patch_columns]
    return [
        values[start : start + patch_size]
        for start in range(0, patch_size * patch_size, patch_size)
    ]


def save_heatmap_grid(
    plt,
    rows: list[dict[str, str]],
    patch_columns: list[str],
    patch_size: int,
    output_path: Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = len(rows)
    cols = 5
    subplot_rows = math.ceil(count / cols)
    fig, axes = plt.subplots(subplot_rows, cols, figsize=(cols * 2.1, subplot_rows * 2.35))

    if subplot_rows == 1:
        axes = [axes]

    max_adc = max(float(row[column]) for row in rows for column in patch_columns)

    for index in range(subplot_rows * cols):
        ax = axes[index // cols][index % cols]
        ax.set_xticks([])
        ax.set_yticks([])

        if index >= count:
            ax.axis("off")
            continue

        row = rows[index]
        image = patch_matrix(row, patch_columns, patch_size)
        ax.imshow(image, cmap="viridis", vmin=0.0, vmax=max_adc)
        ax.set_title(
            f"id {row['cluster_id']}\nerr {float(row['reconstruction_mse']):.5f}",
            fontsize=9,
        )

    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    plt = import_plotting_dependencies()

    merged_rows, patch_columns, patch_size = merge_clusters_with_errors(
        args.clusters_csv, args.errors_csv
    )
    output_csv = args.output_dir / "top_anomalous_clusters.csv"
    save_ranked_clusters(output_csv, merged_rows)

    high_error_rows = merged_rows[: args.top_n]
    low_error_rows = list(reversed(merged_rows[-args.top_n :]))

    save_heatmap_grid(
        plt,
        high_error_rows,
        patch_columns,
        patch_size,
        args.output_dir / "top_anomalous_clusters.png",
        f"Top {len(high_error_rows)} Highest-Error Clusters",
    )
    save_heatmap_grid(
        plt,
        low_error_rows,
        patch_columns,
        patch_size,
        args.output_dir / "top_normal_clusters.png",
        f"Top {len(low_error_rows)} Lowest-Error Clusters",
    )

    print(f"Saved sorted clusters to {output_csv}")
    print(f"Saved high-error heatmaps to {args.output_dir / 'top_anomalous_clusters.png'}")
    print(f"Saved low-error heatmaps to {args.output_dir / 'top_normal_clusters.png'}")


if __name__ == "__main__":
    main()
