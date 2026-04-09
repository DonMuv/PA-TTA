
import math
import cv2
import os
import PIL
import torch
import torch.nn as nn
import torch.jit
import numpy as np
from einops import rearrange
from copy import deepcopy
import torch.nn.functional as F
from core.sam_guide import merge_batch_masks_by_overlap, edge_to_area_ratio_tensor
from core.BIT_CD.models.losses import cross_entropy
from torchvision import utils
    
def softmax_cross_entropy(logits: torch.Tensor, target_logits: torch.Tensor, weights=None) -> torch.Tensor:
    # logits: (B, C, H, W)
    logits = logits.contiguous()
    target_logits = target_logits.contiguous()
    flattened_logits = rearrange(logits, "b c h w -> (b h w) c")
    flattened_target = rearrange(target_logits, "b c h w -> (b h w) c")
    entropy_map = torch.sum(-flattened_target.softmax(1) * flattened_logits.log_softmax(1), dim=1)
    if weights is None:
        entropy_map = entropy_map
    else:
        weights = rearrange(weights, "b h w -> (b h w)")
        entropy_map = entropy_map * weights
    return torch.mean(entropy_map)
    
        
class PaTTA(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)


    def forward(self, x1, x2, weight_mask=None, epoch=None, is_train=True, name=None):
        for _ in range(self.steps):
            outputs = self.forward_and_update(x1, x2, weight_mask, epoch, is_train, name)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        # Use this line to also restore the teacher model                         
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward_and_update(self, x1, x2, sam_mask, epoch, is_train, name):
        with torch.no_grad():
            target = self.model.CDmodel.forward(x1, x2)

        if not is_train:
            seg_logit = self.model.CDmodel.forward(x1, x2)
            return seg_logit

        logits = self.model(x1, x2)

        merged_mask = merge_batch_masks_by_overlap(sam_mask, logits.argmax(dim=1), iou_threshold=0.7)

        loss = 0
        if epoch > 10:
            ratios_pred = edge_to_area_ratio_tensor(logits.argmax(dim=1))
            ratios_sam = edge_to_area_ratio_tensor(merged_mask)
            use_sam = ratios_sam > ratios_pred
            fused = logits.argmax(dim=1).clone()
            for i in range(logits.size(0)):
                if use_sam[i]:
                    fused[i] = sam_mask[i]

            loss = cross_entropy(logits, fused.to('cuda'))
        else:
            loss = cross_entropy(logits, merged_mask.to('cuda'))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss    

def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor
    

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    model.train()
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
