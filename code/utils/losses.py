import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class WingLoss(nn.Module):
    def __init__(self, omega=10, epsilon=2):
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)

    def forward(self, pred, target):
        delta = (target - pred).abs()
        loss_small = self.omega * torch.log(1 + delta / self.epsilon)
        loss_large = delta - self.C
        loss = torch.where(delta < self.omega, loss_small, loss_large)
        return loss.mean()


class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        delta = (target - pred).abs()

        A = (
            self.omega
            * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - target)))
            * (self.alpha - target)
            * torch.pow(self.theta / self.epsilon, self.alpha - target - 1)
            / self.epsilon
        )
        C = self.theta * A - self.omega * torch.log(
            1 + torch.pow(self.theta / self.epsilon, self.alpha - target)
        )

        loss_small = self.omega * torch.log(1 + torch.pow(delta / self.epsilon, self.alpha - target))
        loss_large = A * delta - C
        loss = torch.where(delta < self.theta, loss_small, loss_large)
        return loss.mean()


class CombinedLoss(nn.Module):
    def __init__(self, use_wing_loss=True, wing_omega=10, wing_epsilon=2, lambda_coord=0.1, lambda_struct=0.05):
        super().__init__()
        self.heatmap_loss = WingLoss(omega=wing_omega, epsilon=wing_epsilon) if use_wing_loss else nn.MSELoss()
        self.lambda_coord = lambda_coord
        self.lambda_struct = lambda_struct

    def extract_keypoints_from_heatmap(self, heatmap):
        B, K, _H, W = heatmap.shape
        heatmap_flat = heatmap.view(B, K, -1)
        max_indices = torch.argmax(heatmap_flat, dim=2)
        y_coords = (max_indices // W).float()
        x_coords = (max_indices % W).float()
        return torch.stack([x_coords, y_coords], dim=2)

    def coordinate_loss(self, pred_heatmap, gt_coords):
        pred_coords = self.extract_keypoints_from_heatmap(pred_heatmap)
        return F.l1_loss(pred_coords, gt_coords)

    def structural_loss(self, pred_coords, gt_coords):
        pred_expanded = pred_coords.unsqueeze(2)
        gt_expanded = gt_coords.unsqueeze(2)
        pred_dists = torch.norm(pred_coords.unsqueeze(1) - pred_expanded, dim=3)
        gt_dists = torch.norm(gt_coords.unsqueeze(1) - gt_expanded, dim=3)
        return F.l1_loss(pred_dists, gt_dists)

    def forward(self, pred_heatmap, gt_heatmap, gt_coords=None):
        loss_heatmap = self.heatmap_loss(pred_heatmap, gt_heatmap)
        total_loss = loss_heatmap
        loss_dict = {"heatmap": loss_heatmap.item()}

        if gt_coords is not None and self.lambda_coord > 0:
            loss_coord = self.coordinate_loss(pred_heatmap, gt_coords)
            total_loss = total_loss + self.lambda_coord * loss_coord
            loss_dict["coord"] = loss_coord.item()

            if self.lambda_struct > 0:
                pred_coords = self.extract_keypoints_from_heatmap(pred_heatmap)
                loss_struct = self.structural_loss(pred_coords, gt_coords)
                total_loss = total_loss + self.lambda_struct * loss_struct
                loss_dict["struct"] = loss_struct.item()

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict


def get_loss_function(loss_type="wing", **kwargs):
    if loss_type == "mse":
        return nn.MSELoss()
    if loss_type == "wing":
        return WingLoss(omega=kwargs.get("omega", 10), epsilon=kwargs.get("epsilon", 2))
    if loss_type == "adaptive_wing":
        return AdaptiveWingLoss(
            omega=kwargs.get("omega", 14),
            theta=kwargs.get("theta", 0.5),
            epsilon=kwargs.get("epsilon", 1),
            alpha=kwargs.get("alpha", 2.1),
        )
    if loss_type == "combined":
        return CombinedLoss(
            use_wing_loss=kwargs.get("use_wing_loss", True),
            wing_omega=kwargs.get("wing_omega", 10),
            wing_epsilon=kwargs.get("wing_epsilon", 2),
            lambda_coord=kwargs.get("lambda_coord", 0.1),
            lambda_struct=kwargs.get("lambda_struct", 0.05),
        )
    raise ValueError(f"Unknown loss type: {loss_type}")
