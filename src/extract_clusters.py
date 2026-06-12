#!/usr/bin/env python3
"""Extract ADC patches around local maxima from a phi-tbin hit image."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_INPUT = Path("data/event74_hits.csv")
DEFAULT_OUTPUT = Path("results/clusters_5x5_event74_layer15.csv")
DEFAULT_EVENT = 74
DEFAULT_LAYER = 15
DEFAULT_PATCH_SIZE = 5
DEFAULT_MIN_ADC = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a phi-tbin ADC image and extract patches around local maxima."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input hits CSV.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output patches CSV.")
    parser.add_argument("--event", type=int, default=DEFAULT_EVENT, help="Event number to select.")
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER, help="Layer number to select.")
    parser.add_argument(
        "--patch-size",
        type=int,
        default=DEFAULT_PATCH_SIZE,
        help="Odd patch width and height centered on each local maximum.",
    )
    parser.add_argument(
        "--min-adc",
        type=float,
        default=DEFAULT_MIN_ADC,
        help="Minimum center-pixel ADC required for a local maximum.",
    )
    args = parser.parse_args()
    if args.patch_size <= 0 or args.patch_size % 2 == 0:
        parser.error("--patch-size must be a positive odd integer")
    return args


def read_filtered_hits(
    input_path: Path, event_id: int, layer_id: int
) -> tuple[list[tuple[float, int, float]], list[float], list[int]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    hits: list[tuple[float, int, float]] = []
    phis: set[float] = set()
    tbins: set[int] = set()

    with input_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"event", "layer", "phi", "tbin", "adc"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required column(s): {missing}")

        for row in reader:
            if int(row["event"]) != event_id or int(row["layer"]) != layer_id:
                continue

            phi = float(row["phi"])
            tbin = int(row["tbin"])
            adc = float(row["adc"])
            hits.append((phi, tbin, adc))
            phis.add(phi)
            tbins.add(tbin)

    if not hits:
        raise ValueError(f"No hits found for event={event_id}, layer={layer_id}")

    return hits, sorted(phis), sorted(tbins)


def build_adc_image(
    hits: list[tuple[float, int, float]], phis: list[float], tbins: list[int]
) -> list[list[float]]:
    phi_to_row = {phi: row_index for row_index, phi in enumerate(phis)}
    tbin_to_col = {tbin: col_index for col_index, tbin in enumerate(tbins)}
    image = [[0.0 for _ in tbins] for _ in phis]

    for phi, tbin, adc in hits:
        image[phi_to_row[phi]][tbin_to_col[tbin]] += adc

    return image


def is_local_maximum(
    image: list[list[float]], row: int, col: int, min_adc: float
) -> bool:
    center = image[row][col]
    if center < min_adc:
        return False

    row_count = len(image)
    col_count = len(image[0])

    for drow in (-1, 0, 1):
        for dcol in (-1, 0, 1):
            if drow == 0 and dcol == 0:
                continue

            neighbor_row = row + drow
            neighbor_col = col + dcol
            if 0 <= neighbor_row < row_count and 0 <= neighbor_col < col_count:
                if center <= image[neighbor_row][neighbor_col]:
                    return False

    return True


def extract_patch(
    image: list[list[float]], center_row: int, center_col: int, patch_size: int
) -> list[float]:
    row_count = len(image)
    col_count = len(image[0])
    patch_radius = patch_size // 2
    patch: list[float] = []

    for row in range(center_row - patch_radius, center_row + patch_radius + 1):
        for col in range(center_col - patch_radius, center_col + patch_radius + 1):
            if 0 <= row < row_count and 0 <= col < col_count:
                patch.append(image[row][col])
            else:
                patch.append(0.0)

    return patch


def find_clusters(
    image: list[list[float]],
    phis: list[float],
    tbins: list[int],
    event_id: int,
    layer_id: int,
    patch_size: int,
    min_adc: float,
) -> list[dict[str, float | int]]:
    clusters: list[dict[str, float | int]] = []

    for row_index, phi in enumerate(phis):
        for col_index, tbin in enumerate(tbins):
            if not is_local_maximum(image, row_index, col_index, min_adc):
                continue

            patch = extract_patch(image, row_index, col_index, patch_size)
            cluster: dict[str, float | int] = {
                "cluster_id": len(clusters),
                "event": event_id,
                "layer": layer_id,
                "center_phi": phi,
                "center_tbin": tbin,
                "center_adc": image[row_index][col_index],
                "row_index": row_index,
                "col_index": col_index,
            }
            for patch_index, value in enumerate(patch):
                patch_row = patch_index // patch_size
                patch_col = patch_index % patch_size
                cluster[f"p{patch_row}{patch_col}"] = value
            clusters.append(cluster)

    return clusters


def write_clusters(
    output_path: Path, clusters: list[dict[str, float | int]], patch_size: int
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "cluster_id",
        "event",
        "layer",
        "center_phi",
        "center_tbin",
        "center_adc",
        "row_index",
        "col_index",
    ] + [f"p{row}{col}" for row in range(patch_size) for col in range(patch_size)]

    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(clusters)


def main() -> None:
    args = parse_args()
    hits, phis, tbins = read_filtered_hits(args.input, args.event, args.layer)
    image = build_adc_image(hits, phis, tbins)
    clusters = find_clusters(
        image,
        phis,
        tbins,
        args.event,
        args.layer,
        args.patch_size,
        args.min_adc,
    )
    write_clusters(args.output, clusters, args.patch_size)
    print(
        f"Wrote {len(clusters)} clusters from event={args.event}, layer={args.layer}, "
        f"patch_size={args.patch_size}, min_adc={args.min_adc} to {args.output}"
    )


if __name__ == "__main__":
    main()
