"""
Read and pretty-print contents of a CARI4D .pth file.

Usage:
    python read_pth.py --pth output/opt/.../Date03_Sub01_gas_wild002.pth
    python read_pth.py --pth output/opt/.../Date03_Sub01_gas_wild002.pth --source pr --frames 798 850
"""

import argparse
import torch
import numpy as np


def print_tensor(name, t, sample_frames=None):
    if not isinstance(t, torch.Tensor):
        print(f"    {name}: {type(t).__name__} = {t}")
        return
    print(f"    {name}: shape={list(t.shape)} dtype={t.dtype}")
    print(f"      min={t.float().min().item():.4f}  max={t.float().max().item():.4f}  mean={t.float().mean().item():.4f}")
    if sample_frames:
        for f in sample_frames:
            if f < t.shape[0]:
                print(f"      frame {f}: {t[f].numpy()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth', required=True)
    parser.add_argument('--source', default=None, choices=['pr', 'gt', 'in'],
                        help='Print only this source. If omitted, prints all.')
    parser.add_argument('--frames', type=int, nargs='+', default=[0, 100, 500, 800],
                        help='Sample frames to print values for (default: 0 100 500 800)')
    args = parser.parse_args()

    print(f"Loading: {args.pth}\n")
    d = torch.load(args.pth, map_location='cpu', weights_only=False)

    sources = [args.source] if args.source else list(d.keys())

    for src in sources:
        if src not in d:
            print(f"Source '{src}' not found. Available: {list(d.keys())}")
            continue
        print(f"{'='*60}")
        print(f"  SOURCE: {src}")
        print(f"{'='*60}")
        v = d[src]
        if isinstance(v, dict):
            for k, val in v.items():
                if isinstance(val, torch.Tensor):
                    print_tensor(k, val, sample_frames=args.frames)
                elif isinstance(val, list):
                    print(f"    {k}: list of {len(val)} items")
                else:
                    print(f"    {k}: {type(val).__name__} = {val}")
        else:
            print(f"  {type(v)}: {v}")
        print()


if __name__ == '__main__':
    main()