"""Prepare HLANet high/low sample manifests and five-point extreme crops."""
import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def read_paths(manifest):
    path = Path(manifest)
    return [Path(line) if Path(line).is_absolute() else path.parent / line
            for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def load_patch(path):
    with np.load(path) as patch:
        image = np.asarray(patch['image'], dtype=np.float32)
        height = np.asarray(patch['height'], dtype=np.float32)
        valid = np.asarray(patch['valid'], dtype=bool) if 'valid' in patch else np.isfinite(height)
    if image.ndim != 3 or image.shape[0] != 6 or height.shape != image.shape[1:]:
        raise ValueError(f'Expected image [6,H,W] and height [H,W]: {path}')
    return image, height, valid & np.isfinite(height)


def scene_features(height, valid):
    building = height[valid & (height > 0)]
    if not len(building):
        return np.zeros(6, dtype=np.float64)
    return np.array([building.mean(), building.var(), building.max(),
                     len(building) / max(int(valid.sum()), 1),
                     np.percentile(building, 75), np.percentile(building, 90)], dtype=np.float64)


def write_manifest(path, files):
    path.write_text(''.join(str(file.resolve()) + '\n' for file in files), encoding='utf-8')


def cluster(args):
    files = sorted(args.base_dir.glob('*.npz'))
    if len(files) < 2:
        raise ValueError('At least two .npz patches are needed for K=2')
    features = np.stack([scene_features(*load_patch(file)[1:]) for file in files])
    standardized = StandardScaler().fit_transform(features)
    labels = KMeans(n_clusters=2, random_state=args.seed, n_init=20).fit_predict(standardized)
    cluster_means = [features[labels == label, 0].mean() for label in (0, 1)]
    high_label = int(np.argmax(cluster_means))
    high = [file for file, label, row in zip(files, labels, features) if label == high_label and row[3] >= 0.15]
    low = [file for file, label in zip(files, labels) if label != high_label]
    if args.extreme_manifest:
        for file in read_paths(args.extreme_manifest):
            _, height, valid = load_patch(file)
            if scene_features(height, valid)[3] >= 0.15:
                high.append(file)
    high = list(dict.fromkeys(high))
    if not high or not low:
        raise ValueError('Empty high or low set; inspect the scene distribution')
    args.out.mkdir(parents=True, exist_ok=True)
    write_manifest(args.out / 'high.txt', high)
    write_manifest(args.out / 'low.txt', low)
    print(f'high={len(high)} low={len(low)}; manifests in {args.out}')


def extreme(args):
    image, height, valid = load_patch(args.mosaic)
    size = args.size
    offsets = ((size // 2, size // 2), (5, 5), (5, size - 6),
               (size - 6, 5), (size - 6, size - 6))
    args.out.mkdir(parents=True, exist_ok=True)
    saved = []
    with args.centers.open(newline='', encoding='utf-8') as stream:
        centers = list(csv.DictReader(stream))[:100]
    for number, center in enumerate(centers):
        row, col = int(center['row']), int(center['col'])
        for position, (row_offset, col_offset) in enumerate(offsets):
            top, left = row - row_offset, col - col_offset
            if top < 0 or left < 0 or top + size > height.shape[0] or left + size > height.shape[1]:
                continue
            destination = args.out / f'extreme_{number:03d}_{position}.npz'
            np.savez_compressed(destination, image=image[:, top:top + size, left:left + size],
                                height=height[top:top + size, left:left + size],
                                valid=valid[top:top + size, left:left + size])
            saved.append(destination)
    write_manifest(args.out / 'extreme.txt', saved)
    print(f'saved {len(saved)} extreme crops in {args.out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('cluster')
    p.add_argument('--base-dir', type=Path, required=True)
    p.add_argument('--extreme-manifest', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    p.set_defaults(func=cluster)
    p = commands.add_parser('extreme')
    p.add_argument('--mosaic', type=Path, required=True)
    p.add_argument('--centers', type=Path, required=True, help='CSV of top-100 building center pixels, columns row,col')
    p.add_argument('--size', type=int, default=256)
    p.add_argument('--out', type=Path, required=True)
    p.set_defaults(func=extreme)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
