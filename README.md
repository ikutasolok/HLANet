# HLANet

Code for high- and low-rise specialized building-height estimation. The network has high-rise, low-rise, and height-category branches with a Height-Category Guidance Module (HCGM).

```bash
pip install -r requirements.txt
```

`model.py` defines the network, `dataset.py` loads patches, `prepare_samples.py` prepares high/low samples, `train.py` runs three-stage training, and `metrics.py`/`evaluate.py` report held-out height errors.

Each `.npz` patch contains `image` (`[6, H, W]`: Sentinel-2 B2/B3/B4/B8, then Sentinel-1 VV/VH) and `height` (`[H, W]`, metres). `valid` (`[H, W]`) is optional. Text manifests list one patch path per line; relative paths are resolved from the manifest's directory. Keep evaluation patches separate from training patches.

```bash
python train.py --low-manifest data/low.txt --high-manifest data/high.txt --out runs/hlanet
python evaluate.py --manifest data/test.txt --checkpoint runs/hlanet/05_high.pt
```

Run `python train.py --smoke --device cpu` to check the training stages without data. The repository does not include the original data or checkpoints. Results may differ with input data, preprocessing, sample selection, and train/test split.
