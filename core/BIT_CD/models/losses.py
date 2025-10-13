import torch
import torch.nn.functional as F
from einops import rearrange


def cross_entropy(input, target, weight=None, reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)


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
