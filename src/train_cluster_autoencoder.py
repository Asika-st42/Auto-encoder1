#!/usr/bin/env python3
"""Train a fully connected PyTorch autoencoder on 5x5 cluster ADC patches."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path


DEFAULT_INPUT = Path("results/clusters_5x5_event74_layer15.csv")
DEFAULT_LOSS_PLOT = Path("results/cluster_ae_loss.png")
DEFAULT_RECON_PLOT = Path("results/cluster_ae_reconstructions.png")
DEFAULT_ERROR_PLOT = Path("results/cluster_ae_reconstruction_error_hist.png")
DEFAULT_ERROR_CSV = Path("results/cluster_ae_reconstruction_errors.csv")
DEFAULT_MODEL_PATH = Path("models/cluster_autoencoder.pt")
INPUT_DIM = 25
DEFAULT_LATENT_DIM = 8
VALIDATION_FRACTION = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a 25-32-16-latent-16-32-25 autoencoder on cluster ADC patches."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT, help="Input cluster CSV.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--latent-dim", type=int, default=DEFAULT_LATENT_DIM, help="Latent dimension.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/validation split.")
    parser.add_argument("--show", action="store_true", help="Show plots interactively after saving them.")
    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.latent_dim <= 0:
        parser.error("--latent-dim must be positive")

    return args


def import_training_dependencies(show_plots: bool):
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(Path("results/.matplotlib")))
        import matplotlib

        if not show_plots:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset, random_split
    except ModuleNotFoundError as exc:
        package = exc.name or "required package"
        raise SystemExit(
            f"Missing dependency: {package}. Install PyTorch, matplotlib, and numpy-compatible "
            "scientific packages before running this training script."
        ) from exc

    return plt, torch, nn, DataLoader, TensorDataset, random_split


def patch_column_key(column: str) -> tuple[int, int]:
    return int(column[1:-1]), int(column[-1])


def read_adc_features(input_csv: Path) -> tuple[list[list[float]], list[str], list[str]]:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    with input_csv.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        patch_columns = sorted(
            [name for name in fieldnames if name.startswith("p") and name[1:].isdigit()],
            key=patch_column_key,
        )

        if len(patch_columns) != INPUT_DIM:
            raise ValueError(f"Expected {INPUT_DIM} patch columns, found {len(patch_columns)}")

        cluster_ids: list[str] = []
        features: list[list[float]] = []
        for row_index, row in enumerate(reader):
            cluster_ids.append(row.get("cluster_id", str(row_index)))
            features.append([float(row[column]) for column in patch_columns])

    if not features:
        raise ValueError(f"No rows found in {input_csv}")

    return features, patch_columns, cluster_ids


def normalize_features(features: list[list[float]]) -> tuple[list[list[float]], float]:
    log_features = [[math.log1p(value) for value in row] for row in features]
    max_log_adc = max(max(row) for row in log_features)
    if max_log_adc <= 0:
        return log_features, 1.0
    return [[value / max_log_adc for value in row] for row in log_features], max_log_adc


def make_autoencoder(nn, latent_dim: int):
    return nn.Sequential(
        nn.Linear(INPUT_DIM, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, latent_dim),
        nn.ReLU(),
        nn.Linear(latent_dim, 16),
        nn.ReLU(),
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, INPUT_DIM),
    )


def split_dataset(torch, random_split, dataset, seed: int):
    validation_size = max(1, round(len(dataset) * VALIDATION_FRACTION))
    train_size = len(dataset) - validation_size
    if train_size <= 0:
        raise ValueError("Train/validation split leaves no training samples")

    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, validation_size], generator=generator)


def train_model(
    torch,
    nn,
    DataLoader,
    TensorDataset,
    random_split,
    data,
    epochs: int,
    batch_size: int,
    lr: float,
    latent_dim: int,
    seed: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")

    tensor_data = torch.tensor(data, dtype=torch.float32)
    dataset = TensorDataset(tensor_data)
    train_dataset, validation_dataset = split_dataset(torch, random_split, dataset, seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    model = make_autoencoder(nn, latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses: list[float] = []
    validation_losses: list[float] = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            loss.backward()
            optimizer.step()

            batch_size_actual = batch.size(0)
            total_loss += loss.item() * batch_size_actual
            total_samples += batch_size_actual

        train_loss = total_loss / total_samples
        validation_loss = evaluate_loss(torch, model, criterion, validation_loader, device)
        train_losses.append(train_loss)
        validation_losses.append(validation_loss)
        print(
            f"epoch {epoch + 1:04d}/{epochs} "
            f"train_loss={train_loss:.8f} val_loss={validation_loss:.8f}"
        )

    return model, tensor_data, train_losses, validation_losses, device


def evaluate_loss(torch, model, criterion, loader, device) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            batch_size_actual = batch.size(0)
            total_loss += loss.item() * batch_size_actual
            total_samples += batch_size_actual

    return total_loss / total_samples


def save_loss_plot(
    plt, train_losses: list[float], validation_losses: list[float], output_path: Path, show_plot: bool
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, linewidth=2, label="train")
    ax.plot(epochs, validation_losses, linewidth=2, label="validation")
    ax.set_title("Cluster Autoencoder Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    if show_plot:
        plt.show()
    plt.close(fig)


def compute_reconstruction_errors(torch, model, tensor_data, device, batch_size: int) -> list[float]:
    model.eval()
    errors: list[float] = []

    with torch.no_grad():
        for start in range(0, tensor_data.size(0), batch_size):
            batch = tensor_data[start : start + batch_size].to(device)
            reconstruction = model(batch)
            batch_errors = torch.mean((reconstruction - batch) ** 2, dim=1).cpu().tolist()
            errors.extend(float(error) for error in batch_errors)

    return errors


def save_reconstruction_errors(
    output_path: Path, cluster_ids: list[str], errors: list[float]
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["cluster_id", "reconstruction_mse"])
        writer.writeheader()
        for cluster_id, error in zip(cluster_ids, errors):
            writer.writerow({"cluster_id": cluster_id, "reconstruction_mse": error})


def save_error_histogram(plt, errors: list[float], output_path: Path, show_plot: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(errors, bins=50, color="#2f80ed", edgecolor="#ffffff")
    ax.set_title("Cluster Reconstruction Error")
    ax.set_xlabel("Per-cluster MSE")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    if show_plot:
        plt.show()
    plt.close(fig)


def save_reconstruction_plot(
    plt, torch, model, tensor_data, device, output_path: Path, show_plot: bool, seed: int
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    example_count = min(8, tensor_data.size(0))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(tensor_data.size(0), generator=generator)[:example_count]
    originals = tensor_data[indices]

    with torch.no_grad():
        reconstructions = model(originals.to(device)).cpu()

    fig, axes = plt.subplots(2, example_count, figsize=(1.7 * example_count, 4))
    if example_count == 1:
        axes = [[axes[0]], [axes[1]]]

    for col in range(example_count):
        original = originals[col].reshape(5, 5)
        reconstructed = reconstructions[col].reshape(5, 5)

        axes[0][col].imshow(original, cmap="viridis", vmin=0.0, vmax=1.0)
        axes[0][col].set_title(f"orig {int(indices[col])}", fontsize=9)
        axes[1][col].imshow(reconstructed, cmap="viridis", vmin=0.0, vmax=1.0)
        axes[1][col].set_title("recon", fontsize=9)

        for row in range(2):
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])

    fig.suptitle("Original vs Reconstructed Cluster Patches", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)


def save_model(torch, model, model_path: Path, max_log_adc: float, args: argparse.Namespace) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": INPUT_DIM,
            "latent_dim": args.latent_dim,
            "normalization": "log1p_divide_by_max",
            "max_log_adc": max_log_adc,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "validation_fraction": VALIDATION_FRACTION,
            "seed": args.seed,
            "input_csv": str(args.input_csv),
            "architecture": f"25-32-16-{args.latent_dim}-16-32-25",
        },
        model_path,
    )


def main() -> None:
    args = parse_args()
    plt, torch, nn, DataLoader, TensorDataset, random_split = import_training_dependencies(args.show)

    features, _, cluster_ids = read_adc_features(args.input_csv)
    normalized_features, max_log_adc = normalize_features(features)
    model, tensor_data, train_losses, validation_losses, device = train_model(
        torch,
        nn,
        DataLoader,
        TensorDataset,
        random_split,
        normalized_features,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.latent_dim,
        args.seed,
    )

    reconstruction_errors = compute_reconstruction_errors(
        torch, model, tensor_data, device, args.batch_size
    )

    save_loss_plot(plt, train_losses, validation_losses, DEFAULT_LOSS_PLOT, args.show)
    save_reconstruction_plot(
        plt, torch, model, tensor_data, device, DEFAULT_RECON_PLOT, args.show, args.seed
    )
    save_reconstruction_errors(DEFAULT_ERROR_CSV, cluster_ids, reconstruction_errors)
    save_error_histogram(plt, reconstruction_errors, DEFAULT_ERROR_PLOT, args.show)
    save_model(torch, model, DEFAULT_MODEL_PATH, max_log_adc, args)

    print(f"Saved loss curve to {DEFAULT_LOSS_PLOT}")
    print(f"Saved reconstructions to {DEFAULT_RECON_PLOT}")
    print(f"Saved reconstruction errors to {DEFAULT_ERROR_CSV}")
    print(f"Saved reconstruction error histogram to {DEFAULT_ERROR_PLOT}")
    print(f"Saved model to {DEFAULT_MODEL_PATH}")


if __name__ == "__main__":
    main()
