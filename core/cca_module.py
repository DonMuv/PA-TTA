import torch
import clip
import cv2
import os
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange
from .memory_singe import MemoryBank, BlockMemoryBank
        
class CCA(nn.Module):
    def __init__(self, backbone, capacity=64, threshold=0.5, stop_attn_grad=True, l2norm=False):
        super(CCA, self).__init__()
        self.CDmodel = backbone
        self.capacity = capacity
        self.threshold = threshold
        self.stop_attn_grad = stop_attn_grad
        self.l2norm = l2norm

        self.queue = MemoryBank(capacity, 32)
        self.window_size = 8
        self.block_queue = BlockMemoryBank(h_blocks=8, w_blocks=8, capacity=64, dim_feature=32)
        
        self.attention = nn.MultiheadAttention(32, num_heads=1)
        self.attn_norm = nn.LayerNorm(32)
        self.local_attention = nn.MultiheadAttention(32, num_heads=1)

    def local_mem_attention(self, feat):
        B, C, H, W = feat.shape
        ws = self.window_size
        h_blocks = H // ws
        w_blocks = W // ws
    
        # 划窗
        patch_feats = F.unfold(feat, kernel_size=ws, stride=ws)  # [B, C*ws*ws, N]
        patch_feats = patch_feats.view(B, C, ws * ws, -1).mean(dim=2)  # [B, C, N]
        patch_feats = patch_feats.permute(0, 2, 1)  # [B, N, C]
        N = patch_feats.shape[1]
    
        block_ids = [(i // w_blocks, i % w_blocks) for i in range(N)]
    
        outputs = []
        for b in range(B):
            patch_output = []
            for idx, (i, j) in enumerate(block_ids):
                q = patch_feats[b, idx].unsqueeze(0).unsqueeze(0)  # [1, 1, C]
                mem = self.block_queue.pull(i, j).to(feat.device)  # [K, C]
                if mem.size(0) == 0:
                    patch_output.append(q.squeeze(0))  # fallback to self
                    continue
                mem = mem.unsqueeze(1)  # [K, 1, C]
                out, _ = self.local_attention(q, mem, mem)  # [1, 1, C]
                patch_output.append(out.squeeze(0))  # [1, C]
            outputs.append(torch.stack(patch_output, dim=0))  # [N, C]
    
        attn_out = torch.stack(outputs, dim=0)  # [B, N, C]
        attn_out = attn_out.view(B, h_blocks, w_blocks, C).permute(0, 3, 1, 2)  # [B, C, h_blk, w_blk]
        attn_out = F.interpolate(attn_out, size=(H, W), mode='bilinear', align_corners=False)
        return attn_out
    
    def mem_and_atten(self, feat):
        q = feat.flatten(2).permute(2, 0, 1)
        # Flatten and permute features for attention
        # Get features from the memory bank
        mem = self.queue.pull().to("cuda")
        mem = mem.unsqueeze(1).expand(mem.size(0), q.size(1), mem.size(1))
        # Compute attention
        attn, _ = self.attention(q, mem, mem)      
        attn = self.attn_norm(attn)
        attn = attn.permute(1, 2, 0).view_as(feat)
        local_atten = self.local_mem_attention(feat)
        # Combine attention features with original features
        attn_feat = attn + feat + local_atten
        return attn_feat
        

    def forward(self, x1, x2):
            
        # Extract features from the backbone
        featA, featB = self.CDmodel.forward_feats(x1, x2)

        attn_featA = self.mem_and_atten(featA)
        attn_featB = self.mem_and_atten(featB)

        # Compute segmentation mask
        attn_segA, attn_segB = self.CDmodel.forward_decoder(attn_featA, attn_featB, is_seg=False)
        atten_cp = self.CDmodel.forward_diff(attn_segA, attn_segB)

        # Update memory bank if in training mode
        if self.training:
            self.update_memory_bank(featA, attn_segA)
            self.update_memory_bank(featB, attn_segB)
            self.update_blockmemory_bank(featA, attn_segA)
            self.update_blockmemory_bank(featB, attn_segB)

        return atten_cp

    @torch.no_grad()
    def update_memory_bank(self, feat, conf):
        # Compute confidence scores (e.g., softmax of segmentation output)
        conf = torch.sigmoid(conf)
        conf = conf.max(1, keepdim=True)[0]  # Use max confidence for simplicity

        # Thresholding to select confident regions
        mask = (conf > self.threshold).float()
        feat = feat * mask

        # Flatten and push to memory bank
        feat = feat.flatten(2).permute(2, 0, 1)
        feat = feat.mean(dim=0)  # Average over spatial dimensions
        self.queue.push(feat)

    @torch.no_grad()
    def update_blockmemory_bank(self, feat, conf):
        B, C, H, W = feat.shape
        ws = self.window_size
        conf = torch.sigmoid(conf).max(1, keepdim=True)[0]
        mask = (conf > self.threshold).float()
        feat = feat * mask  # [B, C, H, W]
    
        # unfold + average pooling
        patch_feats = F.unfold(feat, kernel_size=ws, stride=ws)  # [B, C*ws*ws, N]
        patch_feats = patch_feats.view(B, C, ws*ws, -1).mean(dim=2)  # [B, C, N]
        patch_feats = patch_feats.permute(0, 2, 1)  # [B, N, C]
    
        h_blocks = H // ws
        w_blocks = W // ws
        N = h_blocks * w_blocks
    
        block_ids = [(i // w_blocks, i % w_blocks) for i in range(N)]
    
        for b in range(B):
            for idx, (i, j) in enumerate(block_ids):
                patch_feat = patch_feats[b, idx]  # [C]
                self.block_queue.push(patch_feat.unsqueeze(0), [(i, j)])


