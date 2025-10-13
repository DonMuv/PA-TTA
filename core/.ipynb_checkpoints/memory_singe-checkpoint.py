import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


class MemoryBank(nn.Module):
    def __init__(self, capacity, dim_feature):
        super().__init__()
        self.capacity = capacity
        self.dim_feature = dim_feature
        self.queue = torch.randn(capacity, dim_feature)
        self.queue = F.normalize(self.queue, dim=1)
        self.ptr = 0

    @torch.no_grad()
    def push(self, features):
        assert features.size(1) == self.dim_feature
        n = features.size(0)
        if self.ptr + n > self.capacity:
            # Calculate the split point
            split_point = self.capacity - self.ptr
            # Update the remaining space in the queue
            self.queue[self.ptr:self.capacity, :] = features[:split_point, :]
            # Update the beginning of the queue
            self.queue[:n - split_point, :] = features[split_point:, :]
            # Reset the pointer
            self.ptr = n - split_point
        else:
            self.queue[self.ptr:self.ptr + n, :] = features
            self.ptr = (self.ptr + n) % self.capacity

    def pull(self):
        return self.queue


class BlockMemoryBank(nn.Module):
    def __init__(self, h_blocks, w_blocks, capacity, dim_feature):
        super().__init__()
        self.h_blocks = h_blocks
        self.w_blocks = w_blocks
        self.capacity = capacity
        self.dim_feature = dim_feature

        self.banks = nn.ModuleDict({
            f"{i}_{j}": MemoryBank(capacity, dim_feature)
            for i in range(h_blocks)
            for j in range(w_blocks)
        })

    @torch.no_grad()
    def push(self, features, block_indices):
        """
        features: [N, C]
        block_indices: list of (i, j) with length N
        """
        assert features.dim() == 2
        assert len(block_indices) == features.size(0)

        grouped = defaultdict(list)
        for idx, (i, j) in enumerate(block_indices):
            grouped[f"{i}_{j}"].append(features[idx].unsqueeze(0))

        for key, feats in grouped.items():
            feat_tensor = torch.cat(feats, dim=0)  # [M, C]
            self.banks[key].push(feat_tensor)

    def pull(self, i, j):
        return self.banks[f"{i}_{j}"].pull()

    def pull_all(self):
        """Return: dict of {(i,j): Tensor [capacity, C]}"""
        return {
            (int(i), int(j)): bank.pull()
            for key, bank in self.banks.items()
            for i, j in [map(int, key.split('_'))]
        }

