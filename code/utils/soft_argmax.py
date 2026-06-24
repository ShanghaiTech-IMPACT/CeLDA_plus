"""
Soft-Argmax coordinate decoding for differentiable landmark detection
"""
import numpy as np
import torch


def soft_argmax_2d(heatmaps, temperature=1.0):
    """
    Differentiable soft-argmax for 2D coordinate prediction.
    
    Args:
        heatmaps: torch.Tensor of shape [B, K, H, W] or numpy array [K, H, W]
        temperature: softmax temperature (lower = sharper distribution)
    
    Returns:
        coords: [B, K, 2] or [K, 2] tensor/array of (x, y) coordinates
    """
    is_numpy = isinstance(heatmaps, np.ndarray)
    if is_numpy:
        heatmaps = torch.from_numpy(heatmaps).float()
        if heatmaps.dim() == 3:
            heatmaps = heatmaps.unsqueeze(0)
    
    B, K, H, W = heatmaps.shape
    device = heatmaps.device
    
    # Create coordinate grids
    y_coords = torch.arange(H, device=device, dtype=heatmaps.dtype).view(1, 1, H, 1).expand(B, K, H, W)
    x_coords = torch.arange(W, device=device, dtype=heatmaps.dtype).view(1, 1, 1, W).expand(B, K, H, W)
    
    # Apply softmax over spatial dimensions
    heatmaps_flat = heatmaps.view(B, K, -1)
    weights = torch.softmax(heatmaps_flat / temperature, dim=-1)
    weights = weights.view(B, K, H, W)
    
    # Compute weighted average of coordinates
    x_pred = (weights * x_coords).sum(dim=(-2, -1))
    y_pred = (weights * y_coords).sum(dim=(-2, -1))
    
    coords = torch.stack([x_pred, y_pred], dim=-1)
    
    if is_numpy:
        coords = coords.squeeze(0).numpy()
    
    return coords


def decode_prediction_points(similarity_maps, percentile=99.95):
    """
    Decode prediction points using percentile-based peak detection.
    
    Args:
        similarity_maps: [K, H, W] array of similarity maps
        percentile: threshold percentile for peak detection
    
    Returns:
        pred_points: [K, 2] array of (x, y) coordinates
    """
    pred_points = []
    for k in range(similarity_maps.shape[0]):
        sim_k = similarity_maps[k]
        thr = np.percentile(sim_k, percentile)
        mask = sim_k >= thr
        coords = np.argwhere(mask)
        if coords.shape[0] == 0:
            yx = np.unravel_index(np.argmax(sim_k), sim_k.shape)
            y_mean, x_mean = float(yx[0]), float(yx[1])
        else:
            y_mean, x_mean = coords.mean(axis=0)
        pred_points.append([x_mean, y_mean])
    return np.array(pred_points, dtype=float)
