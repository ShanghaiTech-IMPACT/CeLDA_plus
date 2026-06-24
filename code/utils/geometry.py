import torch


def get_mask_num(num_keypoints, mask_ratio):
    mask_num = int(num_keypoints * mask_ratio)
    if mask_num <= 0:
        mask_num = 1
    if mask_num >= num_keypoints:
        mask_num = num_keypoints - 1
    return mask_num


def pairwise_l2(x, eps=1e-8):
    x = x.float()
    if x.dim() == 2:
        diff = x.unsqueeze(0) - x.unsqueeze(1)
        dist = torch.sqrt((diff**2).sum(-1) + eps)
    elif x.dim() == 3:
        diff = x.unsqueeze(2) - x.unsqueeze(1)
        dist = torch.sqrt((diff**2).sum(-1) + eps)
    else:
        raise ValueError(f"pairwise_l2 expects [K,D] or [B,K,D], got {tuple(x.shape)}")
    return dist


def pairwise_euclidean_dist(proto, gt_xy, use_minmax_norm=False, eps=1e-8):
    dist_proto = pairwise_l2(proto, eps=eps)
    dist_gt = pairwise_l2(gt_xy, eps=eps)

    if use_minmax_norm:
        if dist_proto.dim() == 3:
            p_min = dist_proto.amin(dim=(1, 2), keepdim=True)
            p_max = dist_proto.amax(dim=(1, 2), keepdim=True)
            dist_proto = (dist_proto - p_min) / (p_max - p_min + eps)

            g_min = dist_gt.amin(dim=(1, 2), keepdim=True)
            g_max = dist_gt.amax(dim=(1, 2), keepdim=True)
            dist_gt = (dist_gt - g_min) / (g_max - g_min + eps)
        else:
            p_min, p_max = dist_proto.min(), dist_proto.max()
            g_min, g_max = dist_gt.min(), dist_gt.max()
            dist_proto = (dist_proto - p_min) / (p_max - p_min + eps)
            dist_gt = (dist_gt - g_min) / (g_max - g_min + eps)
    return dist_proto, dist_gt
