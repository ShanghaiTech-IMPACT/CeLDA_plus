import logging
import os
import time
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    class SummaryWriter:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def close(self):
            pass

from dataloaders.landmark_dataset import CL_Landmark_Mix
from networks.CeLDA_Plus import CeLDAPlus

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKDIR = PROJECT_ROOT / "workdir"
RUNS_DIR = WORKDIR


def set_global_seed(seed, deterministic=True):
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_logger(args, resultdir, mode="full"):
    logdir = os.path.join(resultdir, "logs")
    savedir = os.path.join(resultdir, "checkpoints")
    shotdir = os.path.join(resultdir, "snapshot")

    print(
        "Result path: {}\nLogs path: {}\nCheckpoints path: {}\nSnapshot path: {}".format(
            resultdir, logdir, savedir, shotdir
        )
    )

    os.makedirs(logdir, exist_ok=True)
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(shotdir, exist_ok=True)

    writer = SummaryWriter(logdir)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    if mode == "full":
        formatter = logging.Formatter(
            "%(asctime)s %(filename)s %(funcName)s [line:%(lineno)d] %(levelname)s %(message)s"
        )
    elif mode == "simple":
        formatter = logging.Formatter("%(asctime)s %(message)s")
    else:
        formatter = logging.Formatter("%(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(os.path.join(shotdir, "snapshot.log"), encoding="utf8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logging.info(str(args))
    return logger, writer


def get_eta_time(start_time, iter_num, total_iter):
    elapsed_time = time.time() - start_time
    estimated_time = (elapsed_time / max(1, iter_num)) * max(0, total_iter - iter_num)
    hours, rem = divmod(estimated_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:02d}:{:02d}:{:02d}".format(int(hours), int(minutes), int(seconds))


def create_Gaussian(Gaussian_size=7, sigma=3, peak=1):
    kernel = np.zeros((Gaussian_size, Gaussian_size))
    center = Gaussian_size // 2
    for x in range(Gaussian_size):
        for y in range(Gaussian_size):
            dist_sq = (x - center) ** 2 + (y - center) ** 2
            kernel[x, y] = np.exp(-dist_sq / (2 * sigma**2))
    kernel /= 2 * np.pi * sigma**2
    kernel *= peak / kernel.max()
    return kernel


def generate_heatmap(Gaussian_size=11, sigma=6, peak=1, keypoints=None, image=None):
    B, _C, H, W = image.shape
    heatmap = np.zeros((B, keypoints.shape[1], H, W))
    kernel = create_Gaussian(Gaussian_size, sigma, peak)

    for i in range(keypoints.shape[1]):
        for j in range(B):
            x, y = keypoints[j, i]
            x, y = int(x), int(y)

            if x < -Gaussian_size // 2 or x >= W + Gaussian_size // 2:
                continue
            if y < -Gaussian_size // 2 or y >= H + Gaussian_size // 2:
                continue

            x1, x2 = max(0, x - Gaussian_size // 2), min(W, x + Gaussian_size // 2 + 1)
            y1, y2 = max(0, y - Gaussian_size // 2), min(H, y + Gaussian_size // 2 + 1)

            if x2 <= x1 or y2 <= y1:
                continue

            kx1, ky1 = max(0, Gaussian_size // 2 - x), max(0, Gaussian_size // 2 - y)
            kx2 = Gaussian_size - max(0, x + Gaussian_size // 2 + 1 - W)
            ky2 = Gaussian_size - max(0, y + Gaussian_size // 2 + 1 - H)

            if kx2 <= kx1 or ky2 <= ky1:
                continue

            adjusted_kernel = kernel[ky1:ky2, kx1:kx2]
            heatmap[j, i, y1:y2, x1:x2] += adjusted_kernel

    return heatmap


def get_train_augmentation(
    image_size=(512, 512),
    rotation_limit=10,
    scale_limit=0.1,
    shift_limit=0.1,
    brightness_limit=0.2,
    contrast_limit=0.2,
    use_elastic=False,
    use_flip=False,
):
    transforms = [
        A.Resize(image_size[0], image_size[1]),
        A.ShiftScaleRotate(
            shift_limit=shift_limit,
            scale_limit=scale_limit,
            rotate_limit=rotation_limit,
            interpolation=1,
            border_mode=0,
            value=0,
            p=0.7,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=0.5,
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
        A.MotionBlur(blur_limit=5, p=0.1),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
    ]

    if use_flip:
        transforms.insert(1, A.HorizontalFlip(p=0.5))

    return A.Compose(
        transforms,
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def get_val_augmentation(image_size=(512, 512)):
    return A.Compose(
        [A.Resize(image_size[0], image_size[1])],
        keypoint_params=A.KeypointParams(format="xy"),
    )


def get_augmentations(
    image_size=(512, 512), rotation_limit=10, use_geometric=True, use_elastic=True, use_flip=False
):
    if not use_geometric:
        train_aug = A.Compose(
            [
                A.Resize(image_size[0], image_size[1]),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.2),
            ],
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )
    else:
        train_aug = get_train_augmentation(
            image_size=image_size,
            rotation_limit=rotation_limit,
            use_elastic=use_elastic,
            use_flip=use_flip,
        )

    val_aug = get_val_augmentation(image_size=image_size)
    return {"train": train_aug, "val": val_aug}


def resolve_run_dir(run_dir="", save_root=None, exp="celda_plus_train"):
    if save_root is None:
        save_root = RUNS_DIR
    if run_dir:
        run_dir = Path(run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
    else:
        exp_dir = Path(save_root) / exp
        if not exp_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")
        candidates = [p for p in exp_dir.iterdir() if p.is_dir()]
        if not candidates:
            raise FileNotFoundError(f"No run folder found in: {exp_dir}")
        run_dir = sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]

    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"checkpoints directory not found: {ckpt_dir}")
    return run_dir


def load_model_from_run_dir(run_dir, checkpoint_tag, number_of_keypoints, device):
    ckpt_dir = Path(run_dir) / "checkpoints"
    model_path = ckpt_dir / f"model_{checkpoint_tag}.pth"
    proto_path = ckpt_dir / f"prototype_{checkpoint_tag}.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not proto_path.exists():
        raise FileNotFoundError(f"Prototype checkpoint not found: {proto_path}")

    net = CeLDAPlus(in_channels=3, landmark_num=number_of_keypoints).to(device)
    state_dict = torch.load(model_path, map_location=device)
    net.load_state_dict(state_dict, strict=False)
    prototype = torch.load(proto_path, map_location=device)
    if isinstance(prototype, torch.nn.Parameter):
        prototype = prototype.data
    if not isinstance(prototype, torch.Tensor):
        prototype = torch.tensor(prototype, dtype=torch.float32, device=device)
    with torch.no_grad():
        net.prototype.copy_(prototype.to(device).float())
    net.eval()
    return net


def build_eval_loader(data_path, split, image_size, batch_size=1, num_workers=2):
    aug = A.Compose([A.Resize(image_size, image_size)], keypoint_params=A.KeypointParams(format="xy"))
    dataset = CL_Landmark_Mix(base_dir=data_path, splits=split, augmentation=aug)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )


def map_points_512_to_raw(points_512, raw_shape, image_size=512):
    if isinstance(points_512, torch.Tensor):
        points_512 = points_512.detach().cpu().numpy()

    pts_list = []
    if isinstance(points_512, np.ndarray):
        for p in points_512:
            pts_list.append([float(p[0]), float(p[1])])
    elif isinstance(points_512, (list, tuple)):
        for p in points_512:
            if isinstance(p, torch.Tensor):
                p = p.detach().cpu().numpy()
            if isinstance(p, np.ndarray):
                p = p.tolist()
            pts_list.append([float(p[0]), float(p[1])])

    points_512 = np.array(pts_list, dtype=float)
    h_raw = float(raw_shape[0])
    w_raw = float(raw_shape[1])

    points_raw = points_512.copy()
    points_raw[:, 0] = points_raw[:, 0] * w_raw / float(image_size)
    points_raw[:, 1] = points_raw[:, 1] * h_raw / float(image_size)
    return points_raw


def map_back_to_raw(pred_512, gt_512, raw_shape, image_size=512):
    pred_raw = map_points_512_to_raw(pred_512, raw_shape, image_size=image_size)
    gt_raw = map_points_512_to_raw(gt_512, raw_shape, image_size=image_size)
    return np.array(pred_raw), np.array(gt_raw)


def map_raw_to_long_edge_2048(pred_raw, gt_raw, raw_shape):
    pred_raw = np.asarray(pred_raw, dtype=float)
    gt_raw = np.asarray(gt_raw, dtype=float)
    if pred_raw.shape != gt_raw.shape:
        raise ValueError(f"pred and gt shape mismatch: {pred_raw.shape} vs {gt_raw.shape}")

    h_raw = float(raw_shape[0])
    w_raw = float(raw_shape[1])
    max_edge = max(h_raw, w_raw)
    if max_edge <= 0:
        raise ValueError(f"Invalid raw shape: {raw_shape}")

    scale = 2048.0 / max_edge
    return pred_raw * scale, gt_raw * scale


def compute_error_mm(pred_raw, gt_raw, raw_shape):
    pred_2048, gt_2048 = map_raw_to_long_edge_2048(pred_raw, gt_raw, raw_shape)
    return np.linalg.norm(pred_2048 - gt_2048, axis=1) * 0.1


def profile_model_efficiency(
    net,
    image_size,
    device,
    warmup=10,
    iters=50,
    input_tensor=None,
    compute_flops=True,
):
    params_m = sum(p.numel() for p in net.parameters()) / 1e6
    flops_g = None
    if compute_flops:
        try:
            from thop import profile as thop_profile

            if input_tensor is None:
                flops_input = torch.randn(1, 3, image_size, image_size, device=device)
            else:
                flops_input = input_tensor.detach().to(device, non_blocking=True).float()
            macs, _ = thop_profile(net, inputs=(flops_input,), verbose=False)
            flops_g = macs / 1e9
        except Exception:
            flops_g = None

    if input_tensor is None:
        prof_input = torch.randn(1, 3, image_size, image_size, device=device)
    else:
        prof_input = input_tensor.detach().to(device, non_blocking=True).float()

    times_ms = []
    peak_gpu_g = 0.0
    net.eval()
    with torch.no_grad():
        if device.type == "cuda":
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            for _ in range(max(1, warmup)):
                _ = net(prof_input)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device=device)
            for _ in range(max(1, iters)):
                starter.record()
                _ = net(prof_input)
                ender.record()
                torch.cuda.synchronize()
                times_ms.append(starter.elapsed_time(ender))
            peak_gpu_g = float(torch.cuda.max_memory_allocated(device=device) / (1024 ** 3))
        else:
            for _ in range(max(1, warmup)):
                _ = net(prof_input)
            for _ in range(max(1, iters)):
                t0 = time.perf_counter()
                _ = net(prof_input)
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000.0)

    return params_m, flops_g, float(np.mean(times_ms)), float(np.std(times_ms)), peak_gpu_g
