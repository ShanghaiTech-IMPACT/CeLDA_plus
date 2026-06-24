import argparse
import json
from pathlib import Path

import numpy as np
import tabulate
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import (
    build_eval_loader,
    compute_error_mm,
    decode_prediction_points,
    load_model_from_run_dir,
    map_back_to_raw,
    profile_model_efficiency,
    resolve_run_dir,
    soft_argmax_2d,
    PROJECT_ROOT,
    RUNS_DIR,
)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def r2(x):
    return round(float(x), 2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="", help="Path to the exp run dir.")
    parser.add_argument("--save_root", type=str, default=str(RUNS_DIR))
    parser.add_argument("--exp", type=str, default="46_celda_plus")
    parser.add_argument("--checkpoint", type=str, default="best")

    parser.add_argument("--data_path", type=str, default=str(PROJECT_ROOT / "data" / "CephaAdoAdu46"))
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="mix",
        choices=["mix", "adult", "adolescent"],
        help="Evaluation subset mode based on age_group key.",
    )
    parser.add_argument("--number_of_keypoints", type=int, default=46)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--peak_percentile", type=float, default=99.95)
    parser.add_argument(
        "--decode_method",
        type=str,
        default="softargmax",
        choices=["percentile", "softargmax"],
        help="Coordinate decode method from similarity map.",
    )
    parser.add_argument("--softargmax_temp", type=float, default=0.5)
    parser.add_argument("--profile_warmup", type=int, default=0)
    parser.add_argument("--profile_iters", type=int, default=1)

    parser.add_argument("--output_dir", type=str, default="")
    return parser.parse_args()


def summarize_metrics(results_by_landmark):
    headers = ["index", "Mean", "SDR@1", "SDR@2", "SDR@3", "SDR@4"]
    table, all_err = [], []
    for i in range(len(results_by_landmark)):
        err_i = np.asarray(results_by_landmark[i], dtype=float)
        if err_i.size > 0:
            all_err.extend(err_i.tolist())
        table.append(
            [
                str(i),
                f"{r2(err_i.mean()) if err_i.size > 0 else 0.00:.2f}",
                f"{r2((err_i < 1).mean() * 100) if err_i.size > 0 else 0.00:.2f}",
                f"{r2((err_i < 2).mean() * 100) if err_i.size > 0 else 0.00:.2f}",
                f"{r2((err_i < 3).mean() * 100) if err_i.size > 0 else 0.00:.2f}",
                f"{r2((err_i < 4).mean() * 100) if err_i.size > 0 else 0.00:.2f}",
            ]
        )
    all_err = np.asarray(all_err, dtype=float)
    if all_err.size == 0:
        all_err = np.array([0.0], dtype=float)
    table.append(
        [
            "All",
            f"{r2(all_err.mean()):.2f}",
            f"{r2((all_err < 1).mean() * 100):.2f}",
            f"{r2((all_err < 2).mean() * 100):.2f}",
            f"{r2((all_err < 3).mean() * 100):.2f}",
            f"{r2((all_err < 4).mean() * 100):.2f}",
        ]
    )
    return headers, table, all_err


def normalize_age_group_label(x):
    if x is None:
        return "mix"
    s = str(x).strip().lower()
    if s in {"adult", "adults"}:
        return "adult"
    if s in {"adolescent", "adolescents", "children", "child", "kids", "kid"}:
        return "adolescent"
    return "mix"


def allow_sample(eval_mode, age_group):
    if eval_mode == "mix":
        return True
    return normalize_age_group_label(age_group) == eval_mode


def parse_raw_shape(raw_shapes_b, b):
    # Default DataLoader collate for a tuple (H, W) becomes [tensor([H...]), tensor([W...])].
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

    raise ValueError(
        f"Unsupported raw_shape batch format at index {b}: "
        f"type={type(raw_shapes_b)}, value={raw_shapes_b}"
    )


def decode_coords(similarity_b, method, peak_percentile, softargmax_temp):
    if method == "softargmax":
        pred_512 = soft_argmax_2d(similarity_b.unsqueeze(0), temperature=softargmax_temp)
        return pred_512[0].detach().cpu().numpy()
    return decode_prediction_points(similarity_b.detach().cpu().numpy(), percentile=peak_percentile)


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    run_dir = resolve_run_dir(
        run_dir=args.run_dir,
        save_root=args.save_root,
        exp=args.exp,

    )

    test_split = "test"
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / f"eval_{test_split}_{args.checkpoint}_{args.eval_mode}"
    output_dir.mkdir(parents=True, exist_ok=True)

    net = load_model_from_run_dir(
        run_dir=run_dir,
        checkpoint_tag=args.checkpoint,
        number_of_keypoints=args.number_of_keypoints,
        device=device,
    )
    params_m = sum(p.numel() for p in net.parameters()) / 1e6
    flops_g = None

    loader = build_eval_loader(
        data_path=args.data_path,
        split=test_split,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    results_by_landmark = {i: [] for i in range(args.number_of_keypoints)}
    per_image_results = {}
    per_sample_forward_ms = []
    per_sample_peak_gpu_g = []
    num_total_seen = 0
    num_selected = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            images = batch["image"].to(device, non_blocking=True)
            keypoints_b = batch["keypoints"].cpu().numpy()
            raw_shapes_b = batch["raw_shape"]
            names_b = batch["file_name"]
            age_groups_b = batch["age_group"] if "age_group" in batch else ["mix"] * images.shape[0]
            num_total_seen += images.shape[0]

            feature_outputs = net(images)
            feature_outputs = [F.interpolate(feat, size=images.shape[-2:], mode="bilinear", align_corners=True) for feat in feature_outputs]
            features_cat = torch.cat(feature_outputs, dim=1)
            similarity = torch.einsum("kd,bdwh->bkwh", net.prototype.detach().float(), features_cat.float())

            similarity_np = similarity.cpu().numpy()
            bsz = images.shape[0]
            for b in range(bsz):
                gt_512 = keypoints_b[b]
                raw_shape = parse_raw_shape(raw_shapes_b, b)
                name = names_b[b]
                age_group = age_groups_b[b]
                if not allow_sample(args.eval_mode, age_group):
                    continue
                num_selected += 1

                if images.shape[0] != 1:
                    raise ValueError("This test script now enforces batch_size=1 for per-sample forward profiling.")
                _params_m, sample_flops, sample_ms, _sample_std, sample_peak = profile_model_efficiency(
                    net=net,
                    image_size=args.image_size,
                    device=device,
                    warmup=args.profile_warmup,
                    iters=args.profile_iters,
                    input_tensor=images[b : b + 1],
                    compute_flops=(flops_g is None),
                )
                if flops_g is None:
                    flops_g = sample_flops
                per_sample_forward_ms.append(float(sample_ms))
                per_sample_peak_gpu_g.append(float(sample_peak))

                pred_512 = decode_coords(
                    similarity_b=similarity[b],
                    method=args.decode_method,
                    peak_percentile=args.peak_percentile,
                    softargmax_temp=args.softargmax_temp,
                )
                pred_raw, gt_raw = map_back_to_raw(pred_512=pred_512, gt_512=gt_512, raw_shape=raw_shape, image_size=args.image_size)
                err_mm = compute_error_mm(pred_raw, gt_raw, raw_shape)

                for i, e in enumerate(err_mm):
                    results_by_landmark[i].append(float(e))
                per_image_results[name] = {
                    "pred": pred_raw,
                    "gt": gt_raw,
                    "error": err_mm,
                    "age_group": normalize_age_group_label(age_group),
                }

    headers, table, all_err = summarize_metrics(results_by_landmark)
    report_str = tabulate.tabulate(table, headers=headers)
    per_sample_ms_mean = float(np.mean(per_sample_forward_ms)) if per_sample_forward_ms else 0.0
    per_sample_ms_std = float(np.std(per_sample_forward_ms)) if per_sample_forward_ms else 0.0
    per_sample_gpu_mean = float(np.mean(per_sample_peak_gpu_g)) if per_sample_peak_gpu_g else 0.0
    print(f"Run directory : {run_dir}")
    print(f"Checkpoint    : {args.checkpoint}")
    print(f"Eval mode     : {args.eval_mode}")
    print(f"Decode method : {args.decode_method}")
    print(f"Samples used  : {num_selected}/{num_total_seen}")
    print(f"Output dir    : {output_dir}")
    if flops_g is None:
        print(
            f"Efficiency | Params: {params_m:.2f} M | FLOP: N/A | "
            f"Per-sample Forward Time: {per_sample_ms_mean:.2f} ms (std {per_sample_ms_std:.2f}) | "
            f"Per-sample Peak GPU: {per_sample_gpu_mean:.2f} G"
        )
    else:
        print(
            f"Efficiency | Params: {params_m:.2f} M | FLOP: {flops_g:.2f} G | "
            f"Per-sample Forward Time: {per_sample_ms_mean:.2f} ms (std {per_sample_ms_std:.2f}) | "
            f"Per-sample Peak GPU: {per_sample_gpu_mean:.2f} G"
        )
    print(report_str)
    print(f"Mean Error (All landmarks): {r2(all_err.mean()):.2f}")
    print(
        "Per-sample single forward | "
        f"Time mean/std: {per_sample_ms_mean:.2f}/{per_sample_ms_std:.2f} ms | "
        f"Peak GPU mean: {per_sample_gpu_mean:.2f} G"
    )

    with open(output_dir / "metrics_table.txt", "w", encoding="utf-8") as f:
        f.write(report_str + "\n")
        f.write(f"\nMean Error (All landmarks): {r2(all_err.mean()):.2f}\n")
        f.write(f"Decode method: {args.decode_method}\n")
        if args.decode_method == "softargmax":
            f.write(f"Softargmax temperature: {args.softargmax_temp}\n")
        else:
            f.write(f"Peak percentile: {args.peak_percentile}\n")

    with open(output_dir / "test_prediction_results.json", "w", encoding="utf-8") as f:
        json.dump(per_image_results, f, cls=NpEncoder, indent=2)

    with open(output_dir / "efficiency_statistics.txt", "w", encoding="utf-8") as f:
        f.write(f"Run directory: {run_dir}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Eval mode: {args.eval_mode}\n")
        f.write(f"Decode method: {args.decode_method}\n")
        if args.decode_method == "softargmax":
            f.write(f"Softargmax temperature: {args.softargmax_temp}\n")
        else:
            f.write(f"Peak percentile: {args.peak_percentile}\n")
        f.write(f"Samples used: {num_selected}/{num_total_seen}\n")
        f.write(f"Image size: {args.image_size}\n")
        f.write(f"Params (M): {params_m:.2f}\n")
        f.write("FLOP (G): N/A (thop unavailable or profiling failed)\n" if flops_g is None else f"FLOP (G): {flops_g:.2f}\n")
        f.write(f"Per-sample Forward Time (ms, mean): {per_sample_ms_mean:.2f}\n")
        f.write(f"Per-sample Forward Time (ms, std): {per_sample_ms_std:.2f}\n")
        f.write(f"Per-sample Peak GPU (G, mean): {per_sample_gpu_mean:.2f}\n")

if __name__ == "__main__":
    main()
