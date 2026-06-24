import argparse
import os
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

from networks.CeLDA_Plus import CeLDAPlus
from networks.Masked_Modeling import Masked_Modeling
from utils import (
    PROJECT_ROOT,
    RUNS_DIR,
    WingLoss,
    build_mix_dataloaders,
    compute_error_mm,
    get_eta_time,
    get_mask_num,
    get_scheduler,
    map_back_to_raw,
    pairwise_euclidean_dist,
    set_global_seed,
    set_logger,
    soft_argmax_2d,
)


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register()

    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default=str(PROJECT_ROOT / "data" / "CephaAdoAdu46"))
    parser.add_argument("--save_path", type=str, default=str(RUNS_DIR))
    parser.add_argument("--exp", type=str, default="celda_plus_train")

    parser.add_argument("--number_of_keypoints", type=int, default=46)
    parser.add_argument("--image_size", type=int, nargs=2, default=[512, 512], metavar=("H", "W"))

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--base_lr", type=float, default=0.001)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--eval_epoch", type=int, default=1)

    parser.add_argument("--lambda_m", type=float, default=2.0, help="Weight of L_sem")
    parser.add_argument("--lambda_s", type=float, default=2.0, help="Weight of L_shape")
    parser.add_argument("--shape_norm", type=int, default=1, choices=[0, 1])

    parser.add_argument("--coord_loss_weight", type=float, default=2.0)
    parser.add_argument("--coord_loss_type", type=str, default="wing", choices=["wing", "l1", "smooth_l1"])
    parser.add_argument("--softargmax_temp", type=float, default=0.5)

    parser.add_argument("--mask_ratio", type=float, default=0.6)
    parser.add_argument("--theta", type=float, default=1.0)

    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "poly"])
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.99)

    parser.add_argument("--rotation_limit", type=int, default=10)
    parser.add_argument("--use_elastic", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use_flip", type=int, default=0, choices=[0, 1])

    parser.add_argument("--use_ema", type=int, default=1, choices=[0, 1])
    parser.add_argument("--ema_decay", type=float, default=0.999)

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--deterministic", type=int, default=1, choices=[0, 1])
    parser.add_argument("--use_amp", type=int, default=1, choices=[0, 1])
    return parser.parse_args()


def parse_raw_shape(raw_shapes_b, b):
    if isinstance(raw_shapes_b, (list, tuple)):
        if len(raw_shapes_b) == 2:
            h_seq, w_seq = raw_shapes_b[0], raw_shapes_b[1]
            try:
                return (int(h_seq[b]), int(w_seq[b]))
            except Exception:
                pass
        if b < len(raw_shapes_b):
            rs = raw_shapes_b[b]
            if isinstance(rs, (list, tuple)) and len(rs) >= 2:
                return (int(rs[0]), int(rs[1]))
            if torch.is_tensor(rs) and rs.numel() >= 2:
                return (int(rs[0].item()), int(rs[1].item()))

    if torch.is_tensor(raw_shapes_b):
        if raw_shapes_b.ndim == 2 and raw_shapes_b.shape[1] >= 2:
            return (int(raw_shapes_b[b, 0].item()), int(raw_shapes_b[b, 1].item()))
        if raw_shapes_b.ndim == 1 and raw_shapes_b.numel() >= 2:
            return (int(raw_shapes_b[0].item()), int(raw_shapes_b[1].item()))

    raise ValueError(f"Unsupported raw_shape format at index {b}: {type(raw_shapes_b)}")


def get_coord_criterion(loss_type):
    if loss_type == "wing":
        return WingLoss(omega=10, epsilon=2)
    if loss_type == "l1":
        return torch.nn.L1Loss()
    if loss_type == "smooth_l1":
        return torch.nn.SmoothL1Loss()
    raise ValueError(f"Unsupported coord loss type: {loss_type}")


def build_similarity(net, image):
    feature_outputs = net(image)
    feature_outputs = [
        F.interpolate(feat, size=image.shape[-2:], mode="bilinear", align_corners=True) for feat in feature_outputs
    ]
    features_cat = torch.cat(feature_outputs, dim=1)
    similarity = torch.einsum("kd,bdwh->bkwh", net.prototype, features_cat.float())
    return similarity


def forward_and_losses(batch, net, modeling_net, coord_criterion, args, mask_num, device):
    image = batch["image"].float().to(device, non_blocking=True)
    gt_xy = batch["keypoints"].float().to(device, non_blocking=True)

    similarity = build_similarity(net, image)
    pred_xy = soft_argmax_2d(similarity, temperature=args.softargmax_temp)

    loss_coord = coord_criterion(pred_xy, gt_xy)

    keypoints_norm = gt_xy / float(args.image_size[0])
    keypoints_features = net.prototype.unsqueeze(1).repeat(1, image.size(0), 1)
    feat_pred, feat_gt, _, _ = modeling_net(
        keypoints_features,
        return_mask=True,
        xy_embed=keypoints_norm,
        mask_num=mask_num,
    )
    feat_pred = feat_pred.transpose(0, 1)
    feat_gt = feat_gt.transpose(0, 1)
    loss_sem = F.mse_loss(feat_pred, feat_gt)

    proto_feat = net.prototype_mlp(net.prototype)
    proto_feat = proto_feat.unsqueeze(0).expand(gt_xy.shape[0], -1, -1)
    dist_proto, dist_gt = pairwise_euclidean_dist(
        proto_feat,
        gt_xy,
        use_minmax_norm=bool(args.shape_norm),
    )
    loss_shape = F.mse_loss(dist_proto, dist_gt)

    loss_total = args.coord_loss_weight * loss_coord + args.lambda_s * loss_shape + args.lambda_m * loss_sem
    return loss_total, loss_coord, loss_shape, loss_sem


def train_epoch(loader, net, modeling_net, coord_criterion, optimizer, scheduler, scaler, ema, args, mask_num, device):
    net.train()
    modeling_net.train()
    sums = {"loss_total": 0.0, "loss_coord": 0.0, "loss_shape": 0.0, "loss_sem": 0.0}
    use_amp = bool(args.use_amp)

    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=use_amp):
            loss_total, loss_coord, loss_shape, loss_sem = forward_and_losses(
                batch=batch,
                net=net,
                modeling_net=modeling_net,
                coord_criterion=coord_criterion,
                args=args,
                mask_num=mask_num,
                device=device,
            )

        if scaler is not None:
            scaler.scale(loss_total).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_norm_(modeling_net.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_total.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_norm_(modeling_net.parameters(), args.grad_clip)
            optimizer.step()

        scheduler.step()
        if ema is not None:
            ema.update()

        sums["loss_total"] += loss_total.item()
        sums["loss_coord"] += loss_coord.item()
        sums["loss_shape"] += loss_shape.item()
        sums["loss_sem"] += loss_sem.item()

    n = len(loader)
    return {k: v / n for k, v in sums.items()}


@torch.no_grad()
def validate_epoch(loader, net, coord_criterion, args, device):
    net.eval()
    total_coord = 0.0
    all_errors = []

    for batch in loader:
        image = batch["image"].float().to(device, non_blocking=True)
        gt_xy = batch["keypoints"].float().to(device, non_blocking=True)
        raw_shapes_b = batch["raw_shape"]

        similarity = build_similarity(net, image)
        pred_xy = soft_argmax_2d(similarity, temperature=args.softargmax_temp)

        loss_coord = coord_criterion(pred_xy, gt_xy)
        total_coord += loss_coord.item()

        pred_np = pred_xy.detach().cpu().numpy()
        gt_np = gt_xy.detach().cpu().numpy()

        for b in range(image.shape[0]):
            raw_shape = parse_raw_shape(raw_shapes_b, b)
            pred_raw, gt_raw = map_back_to_raw(pred_np[b], gt_np[b], raw_shape, image_size=args.image_size[0])
            errors = compute_error_mm(pred_raw, gt_raw, raw_shape)
            all_errors.extend(errors.tolist())

    mean_coord = total_coord / len(loader)
    mre = float(sum(all_errors) / max(1, len(all_errors)))
    return {"loss_coord": mean_coord, "mre": mre}


def save_model_group(net, modeling_net, ckpt_dir, tag):
    torch.save(net.state_dict(), os.path.join(ckpt_dir, f"model_{tag}.pth"))
    torch.save(net.prototype.detach().clone(), os.path.join(ckpt_dir, f"prototype_{tag}.pth"))
    torch.save(modeling_net.state_dict(), os.path.join(ckpt_dir, f"modeling_{tag}.pth"))


def main():
    args = parse_args()
    args.image_size = tuple(args.image_size)
    set_global_seed(args.seed, deterministic=bool(args.deterministic))

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    mask_num = get_mask_num(args.number_of_keypoints, args.mask_ratio)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    resultdir = os.path.join(
        args.save_path,
        args.exp,
        f"k{args.number_of_keypoints}_bs{args.batch_size}_lr{args.base_lr}_"
        f"ls{args.lambda_s}_lm{args.lambda_m}_mr{args.mask_ratio}_{args.scheduler}_{timestamp}",
    )
    ckpt_dir = os.path.join(resultdir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    logger, writer = set_logger(args, resultdir, mode="simple")
    logger.info("=" * 100)
    logger.info("Start CeLDA+ training (EMA + soft-argmax + L_sem + L_shape)")
    logger.info(f"Device: {device}")
    logger.info(f"Dataset: {args.data_path}")
    logger.info(f"lambda_s={args.lambda_s}, lambda_m={args.lambda_m}, coord_weight={args.coord_loss_weight}")
    logger.info(
        f"Keypoints={args.number_of_keypoints}, image_size={args.image_size}, "
        f"batch_size={args.batch_size}, mask_num={mask_num}, use_ema={args.use_ema}"
    )
    logger.info("=" * 100)

    trainloader, valloader = build_mix_dataloaders(
        data_path=args.data_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        rotation_limit=args.rotation_limit,
        use_elastic=bool(args.use_elastic),
        use_flip=bool(args.use_flip),
    )
    logger.info(f"Train samples: {len(trainloader.dataset)}, Val samples: {len(valloader.dataset)}")

    net = CeLDAPlus(in_channels=3, landmark_num=args.number_of_keypoints).to(device)
    modeling_net = Masked_Modeling(landmark_num=args.number_of_keypoints, feature_dim=224).to(device)

    params = [
        {"params": list(net.parameters()), "lr": args.base_lr, "lr_mult": 1.0},
        {"params": list(modeling_net.parameters()), "lr": args.base_lr * args.theta, "lr_mult": args.theta},
    ]
    optimizer = optim.SGD(params, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = get_scheduler(
        optimizer,
        scheduler_type=args.scheduler,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        initial_lr=args.base_lr,
        min_lr=args.min_lr,
        warmup_start_lr=0,
        steps_per_epoch=len(trainloader),
    )

    coord_criterion = get_coord_criterion(args.coord_loss_type)
    scaler = GradScaler() if bool(args.use_amp) else None
    ema = EMA(net, decay=args.ema_decay) if bool(args.use_ema) else None

    best_mre = float("inf")
    start_time = time.time()

    for epoch in range(args.max_epochs):
        t0 = time.time()
        train_stats = train_epoch(
            trainloader,
            net,
            modeling_net,
            coord_criterion,
            optimizer,
            scheduler,
            scaler,
            ema,
            args,
            mask_num,
            device,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        eta = get_eta_time(start_time, epoch + 1, args.max_epochs)
        logger.info(
            f"Epoch [{epoch + 1:03d}/{args.max_epochs}] Train | "
            f"L_total: {train_stats['loss_total']:.4f}, L_coord: {train_stats['loss_coord']:.4f}, "
            f"L_shape: {train_stats['loss_shape']:.4f}, L_sem: {train_stats['loss_sem']:.4f} | "
            f"LR: {current_lr:.6f} | Time: {time.time() - t0:.1f}s | ETA: {eta}"
        )

        writer.add_scalar("train/L_total", train_stats["loss_total"], epoch)
        writer.add_scalar("train/L_coord", train_stats["loss_coord"], epoch)
        writer.add_scalar("train/L_shape", train_stats["loss_shape"], epoch)
        writer.add_scalar("train/L_sem", train_stats["loss_sem"], epoch)
        writer.add_scalar("train/lr", current_lr, epoch)

        if epoch % args.eval_epoch == 0:
            if ema is not None:
                ema.apply_shadow()

            with torch.no_grad():
                val_stats = validate_epoch(valloader, net, coord_criterion, args, device)

            is_best = val_stats["mre"] < best_mre
            if is_best:
                best_mre = val_stats["mre"]
                tag = "best_ema" if ema is not None else "best"
                save_model_group(net, modeling_net, ckpt_dir, tag)

            logger.info(
                f"Epoch [{epoch + 1:03d}/{args.max_epochs}] Val   | "
                f"L_coord: {val_stats['loss_coord']:.4f}, MRE: {val_stats['mre']:.3f} mm | "
                f"Best MRE: {best_mre:.3f} mm"
            )
            writer.add_scalar("val/L_coord", val_stats["loss_coord"], epoch)
            writer.add_scalar("val/MRE", val_stats["mre"], epoch)

            if ema is not None:
                ema.restore()

    if ema is not None:
        ema.apply_shadow()
        save_model_group(net, modeling_net, ckpt_dir, "final_ema")
        ema.restore()
    else:
        save_model_group(net, modeling_net, ckpt_dir, "final")

    total_hours = (time.time() - start_time) / 3600.0
    logger.info("=" * 100)
    logger.info("Training completed.")
    logger.info(f"Best val MRE: {best_mre:.3f} mm")
    logger.info(f"Total time: {total_hours:.2f} hours")
    logger.info(f"Saved to: {resultdir}")
    logger.info("=" * 100)
    writer.close()


if __name__ == "__main__":
    main()
