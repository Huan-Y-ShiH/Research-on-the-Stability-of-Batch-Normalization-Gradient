# Batch Normalization Position and Gradient Stability Research

Research on the impact of batch normalization layer position on gradient stability in convolutional neural networks.

## Experiment Overview

- **Datasets:** CIFAR-10, SVHN
- **Architectures:** SimpleCNN, ResNet18Light
- **BN Configurations:** NoBN, BN_pre (Conv→BN→ReLU), BN_post (Conv→ReLU→BN)
- **Total Runs:** 4 scenarios × 3 configs × 5 seeds = 60 independent training runs
- **Platform:** NVIDIA Tesla T4 GPU, PyTorch 2.x, CUDA 11.8, AMP

## Code Structure

```
experiment/
├── models.py          # CNN model definitions (SimpleCNN + ResNet18Light)
├── trainer.py         # Training loop with gradient monitoring via PyTorch hooks
├── mediation.py       # Mediation analysis (Baron & Kenny + Bootstrap)
├── visualize.py       # Visualization module (7 figure types, SCI-journal quality)
├── run_all.py         # Main experiment orchestration
├── run_cloud.py       # Cloud deployment entry (GPU auto-detection)
└── requirements.txt   # Python dependencies
```

## Results

```
output/
├── summary.json                       # Cross-scenario summary
├── CIFAR10_simple_cnn/
│   ├── data/results.json              # Raw data + statistical tests
│   └── figures/                       # 7 figures (PNG + PDF)
├── CIFAR10_resnet18/
│   ├── data/results.json
│   └── figures/
├── SVHN_simple_cnn/
│   ├── data/results.json
│   └── figures/
└── SVHN_resnet18/
    ├── data/results.json
    └── figures/
```

## Key Findings

1. BN_pre (Conv→BN→ReLU) provides the most balanced gradient distribution across layers
2. BN_post (Conv→ReLU→BN) converges fastest (~27% acceleration) despite higher gradient disparity
3. Residual connections partially compensate for BN_post-induced gradient imbalance
4. Dataset difficulty amplifies BN position effects

## Reproduction

```bash
pip install -r experiment/requirements.txt
python experiment/run_cloud.py
```

CIFAR-10 data should be placed at `cifar-10-batches-py/` (not included in repo due to size).
SVHN is auto-downloaded via torchvision.

## License

This project is for academic research purposes.
