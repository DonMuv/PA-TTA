import cv2
import torch
import numpy as np
from scipy.ndimage import label

def edge_to_area_ratio_tensor(masks: torch.Tensor) -> torch.Tensor:

    masks_np = masks.detach().cpu().numpy().astype(np.uint8)
    B = masks_np.shape[0]
    ratios = []

    for i in range(B):
        mask = masks_np[i]

        # 面积：像素为1的数量
        area = np.sum(mask)
        if area == 0:
            ratios.append(0.0)
            continue

        # 提取轮廓并计算周长
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter = sum([cv2.arcLength(cnt, True) for cnt in contours])

        ratio = perimeter / (area + 1e-6)
        ratios.append(ratio)

    return torch.tensor(ratios, dtype=torch.float32)

    
def edge_complexity_fourier_tensor(masks: torch.Tensor, top_k_ratio=0.1) -> torch.Tensor:

    B, H, W = masks.shape
    masks_np = masks.detach().cpu().numpy().astype(np.uint8)  # 转 numpy uint8

    scores = []

    for i in range(B):
        mask = masks_np[i]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) == 0:
            scores.append(0.0)
            continue

        contour = max(contours, key=cv2.contourArea).squeeze()

        if contour.ndim != 2 or contour.shape[0] < 10:
            scores.append(0.0)
            continue

        curve = contour[:, 0] + 1j * contour[:, 1]  # 复数序列
        fft_result = np.fft.fft(curve)
        fft_magnitude = np.abs(fft_result)
        fft_magnitude[0] = 0  # 去除DC分量

        sorted_mag = np.sort(fft_magnitude)[::-1]
        top_k = int(len(sorted_mag) * top_k_ratio)
        top_energy = np.sum(sorted_mag[:top_k])
        total_energy = np.sum(sorted_mag)

        score = float(top_energy / (total_energy + 1e-6))
        scores.append(score)

    return torch.tensor(scores, dtype=torch.float32)
    
def compute_pixel_diff_ratio(mask1, mask2):
    """
    mask1, mask2: numpy bool arrays of shape [H, W]
    return: float, ratio of differing pixels to total number of pixels
    """
    diff = mask1 ^ mask2  # 像素不同的地方为 1
    H, W = mask1.shape
    total_pixels = H * W
    diff_pixels = diff.sum()

    return diff_pixels / total_pixels
    
def compute_iou(mask1, mask2):
    intersection = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()
    return intersection / union if union != 0 else 0

def merge_single_mask(sam_mask, model_mask, iou_threshold=0.5, max_area_ratio=2.0):
    """
    sam_mask, model_mask: numpy bool array, shape [H, W]
    """
    model_labeled, num_model = label(model_mask)
    sam_labeled, num_sam = label(sam_mask)

    merged = np.zeros_like(model_mask, dtype=bool)

    for model_id in range(1, num_model + 1):
        model_region = model_labeled == model_id
        best_iou = 0
        best_sam_region = None
        best_sam_area = 0

        for sam_id in range(1, num_sam + 1):
            sam_region = sam_labeled == sam_id
            iou = compute_iou(model_region, sam_region)

            if iou > best_iou:
                best_iou = iou
                best_sam_region = sam_region
                best_sam_area = sam_region.sum()

        model_area = model_region.sum()

        # 面积比例检查
        if (
            best_iou >= iou_threshold
            and best_sam_region is not None
            and (1 / max_area_ratio) <= (best_sam_area / model_area) <= max_area_ratio
        ):
            merged |= best_sam_region
        else:
            merged |= model_region

    return merged.astype(np.uint8)

def merge_batch_masks_by_overlap(sam_masks, model_masks, iou_threshold=0.5):
    """
    sam_masks: [B, H, W] torch.Tensor
    model_masks: [B, H, W] torch.Tensor
    return: merged_masks: [B, H, W] torch.uint8
    """
    B, H, W = sam_masks.shape
    merged_batch = []

    for i in range(B):
        sam = sam_masks[i].bool().cpu().numpy()
        model = model_masks[i].bool().cpu().numpy()

        pixel_diff_ratio = compute_pixel_diff_ratio(sam, model)
        if pixel_diff_ratio > 0.5:
            # print(f"[{i}] 🚫 Discard SAM: pixel_diff_ratio={pixel_diff_ratio:.2f}")
            merged = model.astype(np.uint8)
        else:
            merged = merge_single_mask(sam, model, iou_threshold)
            
        # merged = merge_single_mask(sam, model, iou_threshold)
        merged_batch.append(torch.from_numpy(merged))

    return torch.stack(merged_batch, dim=0)  # [B, H, W]
