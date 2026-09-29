"""Spike-train augmentation for base training (GPU, fixed shapes, CUDA-graph safe).

Inputs are S2S event tensors (B, T, C) with values in {-1, 0, 1}.
"""
import torch


class SpikeAugment:
    def __init__(self, time_masks=0, time_mask_len=20, chan_masks=0, chan_mask_len=3,
                 max_shift=0, spike_drop=0.0):
        self.tm, self.tml = time_masks, time_mask_len
        self.cm, self.cml = chan_masks, chan_mask_len
        self.shift, self.drop = max_shift, spike_drop

    def __call__(self, x):
        B, T, C = x.shape
        dev = x.device
        keep = torch.ones(B, T, C, device=dev)
        t = torch.arange(T, device=dev)
        c = torch.arange(C, device=dev)
        for _ in range(self.tm):  # SpecAugment-style time masks
            start = torch.rand(B, 1, device=dev) * T
            length = torch.rand(B, 1, device=dev) * self.tml
            keep = keep * (~((t >= start) & (t < start + length))).float()[:, :, None]
        for _ in range(self.cm):  # frequency-channel masks
            start = torch.rand(B, 1, device=dev) * C
            length = torch.rand(B, 1, device=dev) * self.cml
            keep = keep * (~((c >= start) & (c < start + length))).float()[:, None, :]
        if self.drop:
            keep = keep * (torch.rand(B, T, C, device=dev) >= self.drop).float()
        x = x * keep
        if self.shift:  # random circular time shift with zero fill of the wrapped part
            s = (torch.rand(B, device=dev) * (2 * self.shift + 1)).long() - self.shift
            idx = (t[None, :] - s[:, None])
            valid = ((idx >= 0) & (idx < T)).float()[:, :, None]
            x = torch.gather(x, 1, idx.clamp(0, T - 1)[:, :, None].expand(B, T, C)) * valid
        return x


def build_augment(acfg):
    if not acfg:
        return lambda x: x
    return SpikeAugment(**acfg)
