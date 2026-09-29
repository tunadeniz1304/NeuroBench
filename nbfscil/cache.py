"""Decode MSWC FSCIL audio once and cache the S2S spike encoding on local disk.

The official harness encodes raw 48 kHz audio with neurobench's S2SPreProcessor on
every forward pass. Opus decoding costs ~17 ms/sample on our 2-core machine, so we
run exactly the same encoder once per split and store the int8 result. Feeding the
cached tensors to the model is equivalent to running the encoder inline.

Usage: python -m nbfscil.cache [--splits base_train base_val base_test evaluation]
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from neurobench.datasets import MSWC
from neurobench.datasets.MSWC_IncrementalLoader import IncrementalFewShot
from neurobench.processors.preprocessors import S2SPreProcessor

from nbfscil.paths import DATA_ROOT, CACHE_DIR

EVAL_LANGUAGES = IncrementalFewShot.__init__.__defaults__[0]  # upstream session languages


def make_encoder(device):
    # identical to examples/mswc_fscil/mswc_fscil.py (SPIKING branch)
    enc = S2SPreProcessor(device, transpose=True)
    enc.configure(threshold=1.0, sample_rate=48000, hop_length=240)
    return enc


class _EvalItems(Dataset):
    """Evaluation-language samples in upstream index order, keeping class order."""

    def __init__(self, lang):
        self.ds = MSWC(root=DATA_ROOT, subset="evaluation", language=lang)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        data, target, _, _ = self.ds[i]
        return data, target


def encode_split(name, dataset, device, batch_size=250, workers=2):
    enc = make_encoder(device)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, shuffle=False)
    xs, ys = [], []
    for data, target in tqdm(loader, desc=name):
        spikes, target = enc((data.to(device), target.to(device)))
        if spikes.dim() == 2:  # batch of one squeezed away
            spikes = spikes.unsqueeze(0)
        assert spikes.abs().max() <= 127
        xs.append(spikes.to(torch.int8).cpu())
        ys.append(target.cpu())
    return torch.cat(xs), torch.cat(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+",
                    default=["base_train", "base_val", "base_test", "evaluation"])
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CACHE_DIR, exist_ok=True)

    for split in args.splits:
        out = os.path.join(CACHE_DIR, f"{split}.pt")
        if os.path.exists(out):
            print(f"{out} exists, skipping")
            continue
        if split.startswith("base_"):
            proc = {"base_train": "training", "base_val": "validation", "base_test": "testing"}[split]
            x, y = encode_split(split, MSWC(root=DATA_ROOT, subset="base", procedure=proc), device)
            torch.save({"x": x, "y": y}, out)
        else:
            # per language, samples are ordered class by class, 200 per class
            # (first 100 = support pool, last 100 = query pool, as in upstream)
            per_lang = {}
            for lang in EVAL_LANGUAGES:
                x, y = encode_split(f"evaluation/{lang}", _EvalItems(lang), device)
                per_lang[lang] = {"x": x, "y": y}
            torch.save(per_lang, out)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
