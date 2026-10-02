"""Three-stage HLANet training on prepared six-channel .npz patches."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import ConcatDataset, DataLoader

from dataset import PatchDataset, SyntheticDataset
from model import HLANet

BINS = (3.0, 12.0, 21.0, 30.0, 60.0, 90.0)


def interval_weights(dataset):
    counts = torch.zeros(7, dtype=torch.float64)
    boundaries = torch.tensor(BINS)
    for item in dataset:
        heights = item['height'][item['valid']]
        counts += torch.bincount(torch.bucketize(heights, boundaries, right=True), minlength=7).double()
    # Square-root inverse frequency; rescaling leaves relative bin weights unchanged.
    weights = counts.clamp_min(1).rsqrt()
    return (weights / weights.mean()).float()


def weighted_mse(prediction, target, valid, weights):
    bins = torch.bucketize(target, torch.tensor(BINS, device=target.device), right=True)
    error = (prediction - target).square() * weights[bins]
    return error[valid].mean()


def segmentation_loss(logits, target, valid):
    valid_pixels = valid[:, 0]
    ce = F.cross_entropy(logits, target, reduction='none')[valid_pixels].mean()
    probabilities = logits.softmax(1)
    one_hot = F.one_hot(target, 3).permute(0, 3, 1, 2).float()
    mask = valid.float()
    intersection = (probabilities * one_hot * mask).sum((0, 2, 3))
    denominator = ((probabilities + one_hot) * mask).sum((0, 2, 3))
    dice = 1 - ((2 * intersection + 1) / (denominator + 1)).mean()
    return 0.5 * ce + 0.5 * dice


def train_stage(model, log_variance, dataset, stage, epochs, batch_size, lr, device, high_weights, low_weights, max_steps=None):
    model.set_stage(stage)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(params + [log_variance], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    for epoch in range(epochs):
        model.train()
        # Frozen branch batch-normalization statistics must remain fixed too.
        if stage == 'low':
            model.high_decoder.eval()
        elif stage == 'high':
            model.low_decoder.eval()
        running = 0.0
        steps = 0
        for batch in loader:
            image = batch['image'].to(device)
            height = batch['height'].to(device)
            valid = batch['valid'].to(device)
            labels = batch['class'].to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(image)
            seg = segmentation_loss(output['class_logits'], labels, valid)
            class_term = torch.exp(-log_variance) * seg + log_variance
            loss = class_term
            if stage in ('warmup', 'high'):
                loss = loss + weighted_mse(output['high'], height, valid, high_weights)
            if stage in ('warmup', 'low'):
                loss = loss + weighted_mse(output['low'], height, valid, low_weights)
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        scheduler.step()
        print(f'{stage} epoch {epoch + 1}/{epochs}: loss={running / steps:.4f}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--low-manifest', type=Path)
    parser.add_argument('--high-manifest', type=Path)
    parser.add_argument('--out', type=Path, default=Path('runs/hlanet'))
    parser.add_argument('--warmup-epochs', type=int, default=30)
    parser.add_argument('--specialist-epochs', type=int, default=30)
    parser.add_argument('--cycles', type=int, default=2, help='Low/high alternating cycles; manuscript leaves stopping count unspecified')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--smoke', action='store_true', help='Run one batch in each stage on synthetic data')
    args = parser.parse_args()
    if not args.smoke and (args.low_manifest is None or args.high_manifest is None):
        parser.error('--low-manifest and --high-manifest are required outside --smoke')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu' if args.device == 'auto' else args.device)
    low = SyntheticDataset(False) if args.smoke else PatchDataset(args.low_manifest)
    high = SyntheticDataset(True) if args.smoke else PatchDataset(args.high_manifest)
    model = HLANet().to(device)
    log_variance = nn.Parameter(torch.zeros((), device=device))
    high_weights = interval_weights(high).to(device)
    low_weights = interval_weights(low).to(device)
    args.out.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update({'low_interval_weights': low_weights.tolist(), 'high_interval_weights': high_weights.tolist(), 'input_order': ['B2', 'B3', 'B4', 'B8', 'VV', 'VH']})
    args.out.joinpath('config.json').write_text(json.dumps(config, indent=2, default=str), encoding='utf-8')
    warm_epochs = 1 if args.smoke else args.warmup_epochs
    specialist_epochs = 1 if args.smoke else args.specialist_epochs
    cycles = 1 if args.smoke else args.cycles
    max_steps = 1 if args.smoke else None
    plan = [('warmup', ConcatDataset([low, high]), warm_epochs)]
    for _ in range(cycles):
        plan += [('low', low, specialist_epochs), ('high', high, specialist_epochs)]
    for number, (stage, dataset, epochs) in enumerate(plan, 1):
        train_stage(model, log_variance, dataset, stage, epochs, args.batch_size if not args.smoke else 1,
                    args.lr, device, high_weights, low_weights, max_steps=max_steps)
        torch.save({'model': model.state_dict(), 'log_variance': float(log_variance.detach()),
                    'stage': stage, 'stage_index': number, 'config': config}, args.out / f'{number:02d}_{stage}.pt')
    model.eval()
    with torch.no_grad():
        image = low[0]['image'][None].to(device)
        output = model(image)
    assert output['fused'].shape == (1, 1, *image.shape[-2:])
    print(f'completed {len(plan)} stages; checkpoint directory: {args.out}')


if __name__ == '__main__':
    main()
