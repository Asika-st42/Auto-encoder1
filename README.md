# TPC Cluster Autoencoder Project

## Project Goal

This project extracts local 5x5 ADC cluster images from TPC hit data and trains a fully connected autoencoder to identify unusual cluster shapes. The anomaly score is the reconstruction error: clusters that the autoencoder reconstructs poorly are treated as more anomalous.

## Data Description

The input hit file is expected at:

```text
data/event74_hits.csv
```

The hit CSV contains per-hit information such as:

- `event`
- `layer`
- `phi`
- `tbin`
- `adc`
- spatial coordinates and plotting coordinates

The main workflow uses `event=74` and `layer=15`. Hits are projected into a 2D `phi` by `tbin` ADC image. Local maxima in this image are used as cluster centers, and each cluster is represented as a 5x5 patch of ADC values.

Generated outputs are written to `results/`. Trained model files are written to `models/`.

## Workflow

1. Extract 5x5 clusters around local maxima from the hit image.
2. Inspect extracted clusters with heatmaps and ADC summary histograms.
3. Train a PyTorch autoencoder on the normalized 5x5 cluster patches.
4. Compute reconstruction error for every cluster.
5. Rank clusters by reconstruction error.
6. Plot high-error and low-error examples.
7. Analyze reconstruction error against cluster features.
8. Classify anomalous clusters into interpretable shape categories.

## How To Run Each Script

Create and use the project virtual environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch matplotlib numpy
```

Extract 5x5 clusters:

```bash
.venv/bin/python src/extract_clusters.py \
  --event 74 \
  --layer 15 \
  --patch-size 5 \
  --min-adc 0
```

Inspect extracted clusters:

```bash
.venv/bin/python src/inspect_clusters.py
```

Train the autoencoder:

```bash
.venv/bin/python src/train_cluster_autoencoder.py \
  --epochs 100 \
  --batch-size 128 \
  --learning-rate 0.001 \
  --latent-dim 8
```

Find highest-error and lowest-error clusters:

```bash
.venv/bin/python src/find_anomalous_clusters.py
```

Plot reconstruction error against ADC summary features:

```bash
.venv/bin/python src/analyze_reconstruction_error.py
```

Classify anomalous cluster shapes:

```bash
.venv/bin/python src/classify_anomalous_clusters.py
```

Optional older correlation plotting script:

```bash
.venv/bin/python src/plot_error_correlations.py
```

## Autoencoder Architecture

The training script uses log-normalized ADC inputs:

```text
X = log(1 + ADC)
X = X / max(X)
```

The default fully connected autoencoder is:

```text
input_dim = 25

encoder:
25 -> 32 -> 16 -> latent_dim

decoder:
latent_dim -> 16 -> 32 -> 25
```

The default latent dimension is `8`. The final decoder layer is linear, with no final sigmoid. Training uses MSE loss with an 80% train and 20% validation split.

## Results Summary

Important generated files include:

- `results/clusters_5x5_event74_layer15.csv`
- `results/cluster_ae_loss.png`
- `results/cluster_ae_reconstructions.png`
- `results/cluster_ae_reconstruction_errors.csv`
- `results/cluster_ae_reconstruction_error_hist.png`
- `results/top_anomalous_clusters.csv`
- `results/top_anomalous_clusters.png`
- `results/top_normal_clusters.png`
- `results/cluster_error_feature_analysis.csv`
- `results/anomalous_cluster_classification.csv`
- `results/anomalous_cluster_type_bar.png`
- `models/cluster_autoencoder.pt`

In the current run, the autoencoder was trained for 100 epochs. Both training and validation losses decreased over the run, and the reconstruction error histogram is concentrated at low error with a smaller high-error tail.

## Anomaly Detection Interpretation

The autoencoder learns the most common 5x5 cluster shapes. Low reconstruction error usually indicates a simple pattern that the model can reproduce well. High reconstruction error indicates a cluster shape that is less typical.

High reconstruction error clusters tend to be elongated, multi-peak, edge-localized, or diffuse. Low-error clusters are mostly centered single-peak clusters.

The classification script assigns each anomalous cluster to one of:

- `single_peak`
- `multi_peak`
- `elongated`
- `edge_cluster`
- `diffuse_cluster`

These labels are rule-based summaries of the cluster patch geometry. They are useful for interpretation, but the anomaly score itself comes from the autoencoder reconstruction error.
