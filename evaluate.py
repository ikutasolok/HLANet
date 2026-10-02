"""Evaluate a trained HLANet checkpoint on a held-out patch manifest."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import PatchDataset
from metrics import RegressionMeter
from model import HLANet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    model = HLANet().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    meters = {branch: {'building': RegressionMeter(), 'high_rise': RegressionMeter()}
              for branch in ('high', 'low', 'fused')}
    loader = DataLoader(PatchDataset(args.manifest), batch_size=args.batch_size)
    with torch.no_grad():
        for batch in loader:
            image = batch['image'].to(device)
            target = batch['height'].to(device)
            valid = batch['valid'].to(device)
            output = model(image)
            for branch, groups in meters.items():
                groups['building'].update(output[branch], target, valid & (target > 0))
                groups['high_rise'].update(output[branch], target, valid & (target >= 30))
    print(json.dumps({branch: {name: meter.result() for name, meter in groups.items()}
                      for branch, groups in meters.items()}, indent=2))


if __name__ == '__main__':
    main()
