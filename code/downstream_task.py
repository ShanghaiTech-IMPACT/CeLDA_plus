#!/usr/bin/env python3
import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:
    cKDTree = None


ANGLE_INDICES_46 = {
    "ANB": [0, 10, 19],
    "SNB": [11, 10, 19],
    "SNA": [11, 10, 0],
    "ODI": [0, 19, 20, 21],
    "APDI": [0, 10, 19, 20],
    "FMA": [11, 10, 21, 23],
}

ANGLE_INDICES_201 = {
    "ANB": [6, 3, 10],
    "SNB": [19, 3, 10],
    "SNA": [19, 3, 6],
    "ODI": [6, 10, 11, 13],
    "APDI": [6, 3, 10, 11],
    "FMA": [19, 3, 13, 14],
}

CLASS_METRICS = {
    "ANB": {1: [2.2, 6.7], 2: [6.7, 360], 3: [0, 2.2]},
    "SNB": {1: [74.6, 78.7], 2: [0, 74.6], 3: [78.7, 360]},
    "SNA": {1: [79.4, 83.2], 2: [83.2, 360], 3: [0, 79.4]},
    "ODI": {1: [78.4, 80.5], 2: [80.5, 360], 3: [0, 68.4]},
    "APDI": {1: [77.6, 85.2], 2: [0, 77.6], 3: [85.2, 360]},
    "FMA": {1: [26.8, 31.4], 2: [31.4, 360], 3: [0, 26.8]},
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_points(x):
    if x is None:
        return None
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    return arr[:, :2]


def extract_pred_points(entry):
    if isinstance(entry, list):
        pts = to_points(entry)
        if pts is not None:
            return pts

    if not isinstance(entry, dict):
        return None

    for key in ["pred_final", "pred", "keypoint", "landmarks", "points"]:
        if key in entry:
            pts = to_points(entry[key])
            if pts is not None:
                return pts
    return None


def extract_gt_points(entry):
    if not isinstance(entry, dict):
        return None
    for key in ["gt_raw", "gt", "keypoint", "landmarks", "points"]:
        if key in entry:
            pts = to_points(entry[key])
            if pts is not None:
                return pts
    return None


def build_landmark_maps(gt_json: dict, pred_json: dict, num_points: int):
    gt_map: Dict[str, np.ndarray] = {}
    pred_map: Dict[str, np.ndarray] = {}

    for sid, v in gt_json.items():
        pts = extract_gt_points(v)
        if pts is not None and pts.shape[0] == num_points:
            gt_map[str(sid)] = pts

    for sid, v in pred_json.items():
        pts = extract_pred_points(v)
        if pts is not None and pts.shape[0] == num_points:
            pred_map[str(sid)] = pts

    common = sorted(set(gt_map.keys()) & set(pred_map.keys()))
    if not common:
        raise ValueError("No matched sample IDs between gt and pred with valid point count")

    return gt_map, pred_map, common


def calculate_angle(points):
    if len(points) == 3:
        p1, p2, p3 = points
        v1 = p1 - p2
        v2 = p3 - p2
    elif len(points) == 4:
        p1, p2, p3, p4 = points
        v1 = p2 - p1
        v2 = p4 - p3
    else:
        raise ValueError("calculate_angle expects 3 or 4 points")

    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom < 1e-12:
        return np.nan
    cosv = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosv)))


def get_class_from_angle(angle, metric_bins):
    if angle is None or np.isnan(angle):
        return None
    for cls_id, (lo, hi) in metric_bins.items():
        if lo <= angle <= hi:
            return cls_id
    return None


def compute_scr(gt_map: Dict[str, np.ndarray], pred_map: Dict[str, np.ndarray], common_ids: List[str], angle_indices):
    metrics = list(angle_indices.keys())
    class_ids = [1, 2, 3]

    correct = {m: [] for m in metrics}
    per_class = {m: {c: {"correct": 0, "total": 0} for c in class_ids} for m in metrics}
    used, skipped = 0, 0

    for sid in common_ids:
        gt_pts = gt_map[sid]
        pred_pts = pred_map[sid]

        valid = True
        for _metric, idxs in angle_indices.items():
            if max(idxs) >= pred_pts.shape[0] or max(idxs) >= gt_pts.shape[0]:
                valid = False
                break
        if not valid:
            skipped += 1
            continue

        used += 1
        for metric, idxs in angle_indices.items():
            pred_ang = calculate_angle([pred_pts[i] for i in idxs])
            gt_ang = calculate_angle([gt_pts[i] for i in idxs])
            pred_cls = get_class_from_angle(pred_ang, CLASS_METRICS[metric])
            gt_cls = get_class_from_angle(gt_ang, CLASS_METRICS[metric])

            correct[metric].append(pred_cls == gt_cls)
            if gt_cls in class_ids:
                per_class[metric][gt_cls]["total"] += 1
                if pred_cls == gt_cls:
                    per_class[metric][gt_cls]["correct"] += 1

    scr = {m: (float(np.mean(correct[m])) if len(correct[m]) > 0 else None) for m in metrics}
    scr_per_class = {
        m: {
            c: (
                float(per_class[m][c]["correct"] / per_class[m][c]["total"])
                if per_class[m][c]["total"] > 0
                else None
            )
            for c in class_ids
        }
        for m in metrics
    }
    support = {m: {c: per_class[m][c]["total"] for c in class_ids} for m in metrics}

    return {
        "num_images_used": used,
        "num_images_skipped": skipped,
        "SCR": scr,
        "SCR_per_class": scr_per_class,
        "class_support": support,
    }


def load_line_sequences(py_path: Path) -> List[List[int]]:
    spec = importlib.util.spec_from_file_location("line_index", str(py_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load line definitions from {py_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "MULTI_INTERPOLATION_SEQUENCES"):
        raise ValueError("MULTI_INTERPOLATION_SEQUENCES not found in line definition file")

    sequences = getattr(module, "MULTI_INTERPOLATION_SEQUENCES")
    if not isinstance(sequences, list) or not sequences:
        raise ValueError("MULTI_INTERPOLATION_SEQUENCES must be a non-empty list")

    parsed: List[List[int]] = []
    for i, seq in enumerate(sequences):
        if not isinstance(seq, list) or len(seq) < 2:
            raise ValueError(f"Invalid line sequence at index {i}: {seq}")
        parsed.append([int(x) for x in seq])
    return parsed


def build_polyline(points: np.ndarray, sample_step: float = 1.0) -> np.ndarray:
    if points.shape[0] < 2:
        raise ValueError("A line requires at least 2 points")

    sampled_parts: List[np.ndarray] = []
    for i in range(points.shape[0] - 1):
        p0, p1 = points[i], points[i + 1]
        seg = p1 - p0
        seg_len = float(np.linalg.norm(seg))

        if seg_len == 0.0:
            sampled_parts.append(p0.reshape(1, 2))
            continue

        n_steps = max(1, int(math.ceil(seg_len / sample_step)))
        t = np.linspace(0.0, 1.0, n_steps, endpoint=False).reshape(-1, 1)
        sampled_parts.append(p0.reshape(1, 2) + t * seg.reshape(1, 2))

    sampled_parts.append(points[-1].reshape(1, 2))
    return np.vstack(sampled_parts)


def nearest_distances(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    if cKDTree is not None:
        tree = cKDTree(dst)
        distances, _ = tree.query(src, k=1)
        return distances.astype(float)

    diff = src[:, None, :] - dst[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=2))
    return np.min(d, axis=1)


def compute_asd(points_a: np.ndarray, points_b: np.ndarray) -> float:
    d_ab = nearest_distances(points_a, points_b)
    d_ba = nearest_distances(points_b, points_a)
    return float(0.5 * (np.mean(d_ab) + np.mean(d_ba)))


def compute_hausdorff(points_a: np.ndarray, points_b: np.ndarray) -> float:
    d_ab = nearest_distances(points_a, points_b)
    d_ba = nearest_distances(points_b, points_a)
    return float(max(np.max(d_ab), np.max(d_ba)))


def evaluate_line_metrics(
    gt_map: Dict[str, np.ndarray],
    pred_map: Dict[str, np.ndarray],
    common_ids: Sequence[str],
    sequences: Sequence[Sequence[int]],
    sample_step: float,
):
    detailed_rows: List[dict] = []

    for sid in common_ids:
        gt_points = gt_map[sid]
        pred_points = pred_map[sid]

        for line_idx, seq in enumerate(sequences):
            idx = np.asarray(seq, dtype=int)
            gt_line = build_polyline(gt_points[idx], sample_step=sample_step)
            pred_line = build_polyline(pred_points[idx], sample_step=sample_step)
            asd = compute_asd(pred_line, gt_line)
            hd = compute_hausdorff(pred_line, gt_line)
            detailed_rows.append(
                {
                    "sample_id": sid,
                    "line_index": line_idx,
                    "asd": asd,
                    "hausdorff": hd,
                }
            )

    asd_vals = np.asarray([r["asd"] for r in detailed_rows], dtype=float)
    hd_vals = np.asarray([r["hausdorff"] for r in detailed_rows], dtype=float)

    return detailed_rows, {
        "asd_mean": float(np.mean(asd_vals)),
        "asd_std": float(np.std(asd_vals)),
        "hausdorff_mean": float(np.mean(hd_vals)),
        "hausdorff_std": float(np.std(hd_vals)),
        "num_images": int(len(common_ids)),
        "num_lines": int(len(sequences)),
        "num_line_image_pairs": int(len(detailed_rows)),
    }


def format_scr_percent(scr_result):
    out = {
        "num_images_used": int(scr_result["num_images_used"]),
        "num_images_skipped": int(scr_result["num_images_skipped"]),
        "SCR": {},
        "SCR_per_class": {},
        "class_support": scr_result["class_support"],
    }
    for metric, val in scr_result["SCR"].items():
        out["SCR"][metric] = None if val is None else round(float(val * 100.0), 2)
    for metric, d in scr_result["SCR_per_class"].items():
        out["SCR_per_class"][metric] = {
            str(c): (None if v is None else round(float(v * 100.0), 2)) for c, v in d.items()
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="Run downstream tasks: bone classification + line ASD/HD")
    parser.add_argument("--number_of_keypoints", type=int, required=True, choices=[46, 201])
    parser.add_argument("--gt_path", type=str, required=True)
    parser.add_argument("--pred_path", type=str, required=True)
    parser.add_argument("--line_index_py", type=str, required=True, help="Path to line index python file")
    parser.add_argument("--sample_step", type=float, default=1.0)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory. If empty, save to <run_dir>/downstream_statistics inferred from pred_path.",
    )
    args = parser.parse_args()

    gt_path = Path(args.gt_path)
    pred_path = Path(args.pred_path)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        pred_parent = pred_path.parent
        if pred_parent.name.startswith("eval_"):
            run_dir = pred_parent.parent
        else:
            run_dir = pred_parent
        output_dir = run_dir / "downstream_statistics"
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_json = load_json(gt_path)
    pred_json = load_json(pred_path)

    gt_map, pred_map, common_ids = build_landmark_maps(gt_json, pred_json, num_points=args.number_of_keypoints)

    angle_indices = ANGLE_INDICES_46 if args.number_of_keypoints == 46 else ANGLE_INDICES_201
    metrics_order = ["ANB", "SNB", "SNA"]
    angle_indices = {k: angle_indices[k] for k in metrics_order}

    scr_raw = compute_scr(gt_map, pred_map, common_ids, angle_indices)
    scr_percent = format_scr_percent(scr_raw)

    sequences = load_line_sequences(Path(args.line_index_py))
    max_idx = max(max(seq) for seq in sequences)
    if max_idx >= args.number_of_keypoints:
        raise ValueError(
            f"Line index out of range: max sequence index {max_idx} but num_keypoints={args.number_of_keypoints}"
        )

    line_details, line_summary = evaluate_line_metrics(
        gt_map=gt_map,
        pred_map=pred_map,
        common_ids=common_ids,
        sequences=sequences,
        sample_step=args.sample_step,
    )

    # Save outputs
    per_image = {}
    for row in line_details:
        sid = row["sample_id"]
        if sid not in per_image:
            per_image[sid] = {"asd_values": [], "hausdorff_values": [], "lines": []}
        per_image[sid]["asd_values"].append(float(row["asd"]))
        per_image[sid]["hausdorff_values"].append(float(row["hausdorff"]))
        per_image[sid]["lines"].append(
            {
                "line_index": int(row["line_index"]),
                "asd": float(row["asd"]),
                "hausdorff": float(row["hausdorff"]),
            }
        )

    per_image_summary = {}
    for sid, val in per_image.items():
        asd_arr = np.asarray(val["asd_values"], dtype=float)
        hd_arr = np.asarray(val["hausdorff_values"], dtype=float)
        per_image_summary[sid] = {
            "asd_mean": float(np.mean(asd_arr)),
            "asd_std": float(np.std(asd_arr)),
            "hausdorff_mean": float(np.mean(hd_arr)),
            "hausdorff_std": float(np.std(hd_arr)),
            "num_lines": int(len(val["lines"])),
            "lines": sorted(val["lines"], key=lambda x: x["line_index"]),
        }

    line_metrics_json = {
        "meta": {
            "number_of_keypoints": args.number_of_keypoints,
            "gt_path": str(gt_path),
            "pred_path": str(pred_path),
            "line_index_py": args.line_index_py,
            "sample_step": args.sample_step,
            "num_common_samples": len(common_ids),
        },
        "overall": {
            "asd_mean": round(line_summary["asd_mean"], 4),
            "asd_std": round(line_summary["asd_std"], 4),
            "hausdorff_mean": round(line_summary["hausdorff_mean"], 4),
            "hausdorff_std": round(line_summary["hausdorff_std"], 4),
            "num_images": int(line_summary["num_images"]),
            "num_lines": int(line_summary["num_lines"]),
            "num_line_image_pairs": int(line_summary["num_line_image_pairs"]),
        },
        "per_image": per_image_summary,
    }
    (output_dir / "line_metrics.json").write_text(
        json.dumps(line_metrics_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with (output_dir / f"bone_classification_table_{args.number_of_keypoints}.txt").open("w", encoding="utf-8") as f:
        headers = ["Metric", "Class_I", "Class_II", "Class_III", "Overall", "Support", "N_Used"]
        f.write("{:<8}{:<10}{:<10}{:<10}{:<10}{:<18}{:<8}\n".format(*headers))
        for metric in metrics_order:
            c = scr_percent["SCR_per_class"][metric]
            s = scr_percent["class_support"][metric]
            o = scr_percent["SCR"][metric]
            f.write(
                "{:<8}{:<10}{:<10}{:<10}{:<10}{:<18}{:<8}\n".format(
                    metric,
                    "" if c["1"] is None else f"{c['1']:.2f}",
                    "" if c["2"] is None else f"{c['2']:.2f}",
                    "" if c["3"] is None else f"{c['3']:.2f}",
                    "" if o is None else f"{o:.2f}",
                    f"{s[1]}/{s[2]}/{s[3]}",
                    str(scr_percent["num_images_used"]),
                )
            )

    print(f"[Done] Output directory: {output_dir}")
    print(
        "[Bone] "
        + ", ".join(
            [f"{m}={scr_percent['SCR'][m]:.2f}%" for m in metrics_order if scr_percent["SCR"][m] is not None]
        )
    )
    print(
        f"[Line] ASD={line_summary['asd_mean']:.4f}, HD={line_summary['hausdorff_mean']:.4f}, "
        f"Pairs={line_summary['num_line_image_pairs']}"
    )


if __name__ == "__main__":
    main()
