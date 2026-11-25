"""Evaluate a BubbleUNet checkpoint or TorchScript model on a labeled dataset.

Usage:
    uv run eval.py [checkpoint] [--dataset datasets/synthetic] [--aux] [--json-out metrics.json]

Expected evaluation dataset layout:
    <dataset>/
    ├── images/         # Required: page images (.jpg, .jpeg, .png, .webp), nested subfolders allowed
    ├── masks/          # Required: 8-bit grayscale PNGs (> 0 = foreground) matching images/ paths & HxW
    └── labels_yolo/    # Optional: per-instance polygons (.txt) matching images/ relative paths

Label format (labels_yolo/<rel_stem>.txt):
    One bubble instance per non-empty line with class id 0, >= 3 normalized
    polygon vertices in [0, 1], and a trailing confidence score in [0, 1]:

        0 x1 y1 x2 y2 ... xn yn score

    Used to preserve touching-bubble boundary walls, extract per-instance narrow
    protrusions (tails) for instances with score >= 0.50, and evaluate confidence
    calibration. When labels_yolo/ is omitted, evaluation falls back to masks/.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    BubbleEvalDataset,
    resolve_dataset_path,
)
from model import BubbleUNet, load_model


VALIDATION_THRESHOLD = 0.50
BOUNDARY_TOLERANCE_PX = 1
THIN_NEGATIVE_RADIUS = 4
CONFIDENCE_BACKGROUND_WEIGHT = 0.25
SCORE_EPS = 1e-4


def binary_mask_contour(binary_mask: torch.Tensor) -> torch.Tensor:
    """Two-sided 1-px binary contour matching validation.py."""
    dilated = F.max_pool2d(binary_mask, kernel_size=3, stride=1, padding=1)
    eroded = 1.0 - F.max_pool2d(1.0 - binary_mask, kernel_size=3, stride=1, padding=1)
    return ((dilated - eroded) > 0).float()


def focused_region_scores(
    predicted_mask: torch.Tensor,
    focus_target: torch.Tensor,
    ground_truth: torch.Tensor,
    radius: int = THIN_NEGATIVE_RADIUS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample precision/recall/F1 in the symmetric local neighborhood."""
    target_count = focus_target.sum(dim=(1, 2, 3))
    true_positive = (predicted_mask * focus_target).sum(dim=(1, 2, 3))
    recall = true_positive / target_count.clamp(min=1.0)
    neighborhood = F.max_pool2d(
        focus_target,
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    )
    local_negative = neighborhood * (1.0 - ground_truth)
    evaluation_region = (focus_target + local_negative).clamp(max=1.0)
    prediction_count = (predicted_mask * evaluation_region).sum(dim=(1, 2, 3))
    precision = true_positive / prediction_count.clamp(min=1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp(min=1e-6)
    valid = (target_count > 0).float()
    return precision, recall, f1, valid


def compute_teacher_fidelity_score(metrics: dict[str, float]) -> float:
    """Geometric-mean checkpoint ranking score matching validation.py."""
    factors = [
        max(metrics["dice"], 0.0),
        max(metrics["mask_boundary_precision"], 0.0),
        max(metrics["mask_boundary_recall"], 0.0),
    ]
    if metrics["narrow_valid_samples"] > 0:
        factors.extend(
            (
                max(metrics["narrow_precision"], 0.0),
                max(metrics["narrow_recall"], 0.0),
            )
        )
    mean_log = sum(math.log(value + SCORE_EPS) for value in factors) / len(factors)
    return max(math.exp(mean_log) - SCORE_EPS, 0.0)


def load_eval_model(
    checkpoint_path: Path,
    device: torch.device,
    weights: str = "ema",
    require_aux: bool = False,
) -> tuple[torch.nn.Module, dict]:
    """Load either a training .pth checkpoint or a TorchScript .pt export."""
    try:
        model, checkpoint = load_model(checkpoint_path)
        if weights == "raw":
            if "raw_model_state_dict" not in checkpoint:
                raise ValueError(f"{checkpoint_path} does not contain raw_model_state_dict")
            model.load_state_dict(checkpoint["raw_model_state_dict"], strict=True)
        model = model.to(device)
        model = model.to(memory_format=torch.channels_last)
        info = {
            "format": "checkpoint",
            "weights": weights,
            "variant": checkpoint.get("variant", model.variant),
            "epoch": checkpoint.get("epoch"),
            "input_height": int(checkpoint.get("input_height", DEFAULT_HEIGHT)),
            "input_width": int(checkpoint.get("input_width", DEFAULT_WIDTH)),
        }
        return model.eval(), info
    except Exception as ckpt_err:
        if weights == "raw":
            raise ValueError(
                f"--weights raw requires a training .pth checkpoint: {ckpt_err}"
            ) from ckpt_err
        if require_aux:
            raise ValueError(
                f"--aux requires a BubbleUNet training .pth checkpoint: {ckpt_err}"
            ) from ckpt_err
        try:
            scripted = torch.jit.load(str(checkpoint_path), map_location=device).eval()
        except Exception as jit_err:
            raise RuntimeError(
                f"Failed to load {checkpoint_path} as checkpoint ({ckpt_err}) "
                f"or TorchScript ({jit_err})"
            ) from jit_err
        return scripted, {
            "format": "torchscript",
            "weights": "exported",
            "variant": None,
            "epoch": None,
            "input_height": DEFAULT_HEIGHT,
            "input_width": DEFAULT_WIDTH,
        }


class ProgressBar:
    """Zero-dependency stderr progress indicator without GPU synchronization."""

    def __init__(self, total: int, enabled: bool = True, bar_width: int = 24) -> None:
        self.total = max(int(total), 1)
        self.enabled = enabled
        self.bar_width = bar_width
        self.is_tty = enabled and sys.stderr.isatty()
        self.start = time.perf_counter()
        self.last_len = 0
        self.last_log_time = 0.0
        self.done = 0

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        seconds = max(0, int(round(seconds)))
        mins, secs = divmod(seconds, 60)
        hours, mins = divmod(mins, 60)
        if hours:
            return f"{hours}h{mins:02d}m{secs:02d}s"
        if mins:
            return f"{mins}m{secs:02d}s"
        return f"{secs}s"

    def update(self, step: int) -> None:
        if not self.enabled:
            return
        self.done = min(self.done + step, self.total)
        now = time.perf_counter()
        finished = self.done >= self.total
        if not self.is_tty and not finished and (now - self.last_log_time) < 2.0:
            return
        self.last_log_time = now

        elapsed = max(now - self.start, 1e-6)
        rate = self.done / elapsed
        eta = (self.total - self.done) / rate if rate > 0 else 0.0
        frac = self.done / self.total
        filled = int(round(self.bar_width * frac))
        bar = "#" * filled + "-" * (self.bar_width - filled)
        msg = (
            f"Eval [{bar}] {self.done}/{self.total} ({100.0 * frac:5.1f}%) | "
            f"{rate:.1f} pages/s | ETA {self._fmt_duration(eta)}"
        )
        if self.is_tty:
            pad = " " * max(self.last_len - len(msg), 0)
            sys.stderr.write(f"\r{msg}{pad}")
            sys.stderr.flush()
            self.last_len = len(msg)
        else:
            sys.stderr.write(f"{msg}\n")
            sys.stderr.flush()

    def close(self) -> None:
        if self.is_tty and self.last_len > 0:
            sys.stderr.write("\r" + " " * self.last_len + "\r")
            sys.stderr.flush()
            self.last_len = 0


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    threshold: float = VALIDATION_THRESHOLD,
    boundary_tolerance_px: int = BOUNDARY_TOLERANCE_PX,
    use_amp: bool = True,
    eval_aux: bool = False,
    show_progress: bool = True,
) -> dict[str, float]:
    """Run batched evaluation with device-side accumulation (zero per-batch syncs)."""
    model.eval()
    amp_enabled = use_amp and device.type == "cuda"
    tolerance_kernel = 2 * boundary_tolerance_px + 1

    # Single device-side float64 accumulator avoids per-batch CPU-GPU sync stalls.
    acc = torch.zeros(37, dtype=torch.float64, device=device)
    progress = ProgressBar(len(loader.dataset), enabled=show_progress)

    try:
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            if isinstance(model, BubbleUNet):
                image = image.contiguous(memory_format=torch.channels_last)
            gt_mask = batch["mask"].to(device, non_blocking=True)
            gt_boundary = batch["boundary"].to(device, non_blocking=True)
            gt_thin = batch["thin"].to(device, non_blocking=True)
            gt_narrow = batch["narrow"].to(device, non_blocking=True)
            gt_conf = batch["conf"].to(device, non_blocking=True)
            gt_distance = (
                batch["distance"].to(device, non_blocking=True) if eval_aux else None
            )

            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=amp_enabled
            ):
                if eval_aux:
                    assert isinstance(model, BubbleUNet)
                    mask_h, boundary_h, tail_half, distance_half, conf_h = (
                        model._forward_half(image)
                    )
                    logits = model._upsample(
                        torch.cat((mask_h, boundary_h, conf_h), dim=1),
                        image.shape[-2],
                        image.shape[-1],
                    )
                else:
                    logits = model(image)
                    tail_half = None
                    distance_half = None

            pred_prob = torch.sigmoid(logits[:, 0:1].float())
            boundary_prob = torch.sigmoid(logits[:, 1:2].float())
            conf_prob = torch.sigmoid(logits[:, 2:3].float())

            predicted_mask = (pred_prob > threshold).float()
            pred_boundary = (boundary_prob > threshold).float()

            # Global mask metrics
            intersection = (predicted_mask * gt_mask).sum(dim=(1, 2, 3))
            predicted_sum = predicted_mask.sum(dim=(1, 2, 3))
            target_sum = gt_mask.sum(dim=(1, 2, 3))

            # Auxiliary boundary head (channel 1)
            b_inter = (pred_boundary * gt_boundary).sum(dim=(1, 2, 3))
            b_union = (
                pred_boundary.sum(dim=(1, 2, 3)) + gt_boundary.sum(dim=(1, 2, 3)) - b_inter
            )
            dilated_gt_b = F.max_pool2d(
                gt_boundary,
                kernel_size=tolerance_kernel,
                stride=1,
                padding=boundary_tolerance_px,
            )
            dilated_pred_b = F.max_pool2d(
                pred_boundary,
                kernel_size=tolerance_kernel,
                stride=1,
                padding=boundary_tolerance_px,
            )
            aux_matched_pred = (pred_boundary * dilated_gt_b).sum(dim=(1, 2, 3))
            aux_matched_gt = (gt_boundary * dilated_pred_b).sum(dim=(1, 2, 3))

            # Final-mask contour metrics
            pred_contour = binary_mask_contour(predicted_mask)
            gt_contour = binary_mask_contour(gt_mask)
            dilated_gt_c = F.max_pool2d(
                gt_contour,
                kernel_size=tolerance_kernel,
                stride=1,
                padding=boundary_tolerance_px,
            )
            dilated_pred_c = F.max_pool2d(
                pred_contour,
                kernel_size=tolerance_kernel,
                stride=1,
                padding=boundary_tolerance_px,
            )
            mask_matched_pred = (pred_contour * dilated_gt_c).sum(dim=(1, 2, 3))
            mask_matched_gt = (gt_contour * dilated_pred_c).sum(dim=(1, 2, 3))

            # Thin & narrow focused scores
            thin_p, thin_r, thin_f1, thin_valid = focused_region_scores(
                predicted_mask, gt_thin, gt_mask, THIN_NEGATIVE_RADIUS
            )
            narrow_p, narrow_r, narrow_f1, narrow_valid = focused_region_scores(
                predicted_mask, gt_narrow, gt_mask, THIN_NEGATIVE_RADIUS
            )

            # Confidence head calibration
            conf_error = (conf_prob - gt_conf).abs()
            conf_fg = ((gt_conf > 0) | (gt_mask > 0)).float()
            conf_bg = 1.0 - conf_fg
            conf_fg_cnt = conf_fg.sum(dim=(1, 2, 3))
            conf_bg_cnt = conf_bg.sum(dim=(1, 2, 3))
            conf_fg_mae = (conf_error * conf_fg).sum(dim=(1, 2, 3)) / conf_fg_cnt.clamp(
                min=1.0
            )
            conf_bg_mae = (conf_error * conf_bg).sum(dim=(1, 2, 3)) / conf_bg_cnt.clamp(
                min=1.0
            )
            has_fg = (conf_fg_cnt > 0).float()
            has_bg = (conf_bg_cnt > 0).float()
            conf_mae = (
                has_fg * conf_fg_mae + CONFIDENCE_BACKGROUND_WEIGHT * has_bg * conf_bg_mae
            ) / (has_fg + CONFIDENCE_BACKGROUND_WEIGHT * has_bg).clamp(min=1e-6)

            background = 1.0 - gt_mask
            empty_pages = target_sum == 0

            acc[0] += intersection.sum()
            acc[1] += predicted_sum.sum()
            acc[2] += target_sum.sum()
            acc[3] += b_inter.sum()
            acc[4] += b_union.sum()
            acc[5] += aux_matched_pred.sum()
            acc[6] += pred_boundary.sum()
            acc[7] += aux_matched_gt.sum()
            acc[8] += gt_boundary.sum()
            acc[9] += mask_matched_pred.sum()
            acc[10] += pred_contour.sum()
            acc[11] += mask_matched_gt.sum()
            acc[12] += gt_contour.sum()
            acc[13] += (thin_r * thin_valid).sum()
            acc[14] += (thin_p * thin_valid).sum()
            acc[15] += (thin_f1 * thin_valid).sum()
            acc[16] += thin_valid.sum()
            acc[17] += (narrow_r * narrow_valid).sum()
            acc[18] += (narrow_p * narrow_valid).sum()
            acc[19] += (narrow_f1 * narrow_valid).sum()
            acc[20] += narrow_valid.sum()
            acc[21] += conf_mae.sum()
            acc[22] += conf_bg_mae.sum()
            acc[23] += (pred_prob * gt_mask).sum()
            acc[24] += gt_mask.sum()
            acc[25] += (pred_prob * background).sum()
            acc[26] += background.sum()
            acc[27] += predicted_sum.sum()
            acc[28] += predicted_mask.numel()
            acc[29] += empty_pages.sum()
            acc[30] += (empty_pages & (predicted_sum > 0)).sum()
            acc[31] += image.shape[0]

            if (
                eval_aux
                and tail_half is not None
                and distance_half is not None
                and gt_distance is not None
            ):
                tail_prob = torch.sigmoid(tail_half.float())
                pred_tail = (tail_prob > threshold).float()
                tail_target = F.adaptive_max_pool2d(gt_narrow, tail_prob.shape[-2:])
                dist_pred = distance_half.float()
                dist_target = F.interpolate(
                    gt_distance.float(),
                    size=dist_pred.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                dist_band = (dist_target.abs() < 0.999).float()
                acc[32] += (pred_tail * tail_target).sum()
                acc[33] += pred_tail.sum()
                acc[34] += tail_target.sum()
                acc[35] += ((dist_pred - dist_target).abs() * dist_band).sum()
                acc[36] += dist_band.sum()

            progress.update(int(image.shape[0]))
    finally:
        progress.close()

    vals = acc.tolist()
    dist_band_pixels = vals[36]

    def safe_ratio(num: float, den: float) -> float:
        return num / den if den > 0.0 else 0.0

    mask_inter, mask_pred, mask_gt = vals[0], vals[1], vals[2]
    aux_b_inter, aux_b_union = vals[3], vals[4]
    aux_b_mp, aux_b_p, aux_b_mt, aux_b_t = vals[5], vals[6], vals[7], vals[8]
    mask_c_mp, mask_c_p, mask_c_mt, mask_c_t = vals[9], vals[10], vals[11], vals[12]
    thin_r_s, thin_p_s, thin_f1_s, thin_n = vals[13], vals[14], vals[15], vals[16]
    nar_r_s, nar_p_s, nar_f1_s, nar_n = vals[17], vals[18], vals[19], vals[20]
    conf_mae_s, conf_bg_mae_s = vals[21], vals[22]
    fg_prob_s, fg_pix, bg_prob_s, bg_pix = vals[23], vals[24], vals[25], vals[26]
    pred_fg_pix, total_pix = vals[27], vals[28]
    empty_cnt, empty_fp_cnt, sample_cnt = vals[29], vals[30], vals[31]

    aux_b_prec = safe_ratio(aux_b_mp, aux_b_p)
    aux_b_rec = safe_ratio(aux_b_mt, aux_b_t)
    mask_c_prec = safe_ratio(mask_c_mp, mask_c_p)
    mask_c_rec = safe_ratio(mask_c_mt, mask_c_t)

    metrics: dict[str, float] = {
        "samples": int(sample_cnt),
        "dice": safe_ratio(2.0 * mask_inter, mask_pred + mask_gt),
        "iou": safe_ratio(mask_inter, mask_pred + mask_gt - mask_inter),
        "precision": safe_ratio(mask_inter, mask_pred),
        "recall": safe_ratio(mask_inter, mask_gt),
        "mask_boundary_precision": mask_c_prec,
        "mask_boundary_recall": mask_c_rec,
        "mask_boundary_f1": safe_ratio(
            2.0 * mask_c_prec * mask_c_rec, mask_c_prec + mask_c_rec
        ),
        "spurious_boundary_rate": safe_ratio(mask_c_p - mask_c_mp, mask_c_p),
        "thin_precision": safe_ratio(thin_p_s, thin_n),
        "thin_recall": safe_ratio(thin_r_s, thin_n),
        "thin_f1": safe_ratio(thin_f1_s, thin_n),
        "narrow_precision": safe_ratio(nar_p_s, nar_n),
        "narrow_recall": safe_ratio(nar_r_s, nar_n),
        "narrow_f1": safe_ratio(nar_f1_s, nar_n),
        "narrow_valid_samples": int(nar_n),
        "aux_boundary_iou": safe_ratio(aux_b_inter, aux_b_union),
        "aux_boundary_f1": safe_ratio(
            2.0 * aux_b_prec * aux_b_rec, aux_b_prec + aux_b_rec
        ),
        "conf_mae": safe_ratio(conf_mae_s, sample_cnt),
        "conf_background_mae": safe_ratio(conf_bg_mae_s, sample_cnt),
        "foreground_probability": safe_ratio(fg_prob_s, fg_pix),
        "background_probability": safe_ratio(bg_prob_s, bg_pix),
        "predicted_foreground_rate": safe_ratio(pred_fg_pix, total_pix),
        "empty_page_false_positive_rate": safe_ratio(empty_fp_cnt, empty_cnt),
        "empty_page_count": int(empty_cnt),
    }

    if eval_aux:
        tail_inter, tail_pred, tail_gt, dist_err = vals[32], vals[33], vals[34], vals[35]
        tail_p = safe_ratio(tail_inter, tail_pred)
        tail_r = safe_ratio(tail_inter, tail_gt)
        metrics["aux_tail_precision"] = tail_p
        metrics["aux_tail_recall"] = tail_r
        metrics["aux_tail_f1"] = safe_ratio(2.0 * tail_p * tail_r, tail_p + tail_r)
        metrics["distance_band_mae"] = safe_ratio(dist_err, dist_band_pixels)

    metrics["teacher_fidelity_score"] = compute_teacher_fidelity_score(metrics)
    return metrics


def _worker_init(_worker_id: int) -> None:
    cv2.setNumThreads(1)


def format_report(
    checkpoint_path: Path,
    dataset_path: Path,
    device: torch.device,
    model_info: dict,
    metrics: dict[str, float],
    elapsed_sec: float,
) -> str:
    throughput = metrics["samples"] / max(elapsed_sec, 1e-6)
    meta = f"format={model_info['format']} | weights={model_info['weights']}"
    if model_info.get("variant"):
        meta += f" | variant={model_info['variant']}"
    if model_info.get("epoch") is not None:
        meta += f" | epoch={model_info['epoch']}"
    lines = [
        f"Checkpoint : {checkpoint_path} ({meta})",
        f"Dataset    : {dataset_path} ({metrics['samples']} pages, "
        f"{model_info['input_height']}x{model_info['input_width']})",
        f"Score      : TeacherFidelity={metrics['teacher_fidelity_score']:.4f}",
        f"Mask       : Dice={metrics['dice']:.4f} | IoU={metrics['iou']:.4f} | "
        f"P={metrics['precision']:.4f} | R={metrics['recall']:.4f}",
        f"Contour    : P={metrics['mask_boundary_precision']:.4f} | "
        f"R={metrics['mask_boundary_recall']:.4f} | "
        f"F1={metrics['mask_boundary_f1']:.4f} | "
        f"Spurious={metrics['spurious_boundary_rate']:.4f}",
        f"Thin       : P={metrics['thin_precision']:.4f} | "
        f"R={metrics['thin_recall']:.4f} | F1={metrics['thin_f1']:.4f}",
        f"Narrow     : P={metrics['narrow_precision']:.4f} | "
        f"R={metrics['narrow_recall']:.4f} | F1={metrics['narrow_f1']:.4f} "
        f"(valid={metrics['narrow_valid_samples']})",
        f"Aux/Conf   : AuxBndIoU={metrics['aux_boundary_iou']:.4f} | "
        f"AuxBndF1={metrics['aux_boundary_f1']:.4f} | "
        f"ConfMAE={metrics['conf_mae']:.4f} | ConfBgMAE={metrics['conf_background_mae']:.4f}",
        f"Sanity     : FgProb={metrics['foreground_probability']:.4f} | "
        f"BgProb={metrics['background_probability']:.4f} | "
        f"PredFg={metrics['predicted_foreground_rate']:.4f} | "
        f"EmptyPageFP={metrics['empty_page_false_positive_rate']:.4f} "
        f"(empty={metrics['empty_page_count']})",
    ]
    if "aux_tail_f1" in metrics:
        lines.append(
            f"AuxHalf    : TailP={metrics['aux_tail_precision']:.4f} | "
            f"TailR={metrics['aux_tail_recall']:.4f} | "
            f"TailF1={metrics['aux_tail_f1']:.4f} | "
            f"DistBandMAE={metrics['distance_band_mae']:.4f}"
        )
    lines.append(
        f"Elapsed    : {elapsed_sec:.2f}s ({throughput:.1f} pages/s, device={device})"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        nargs="?",
        default=Path(__file__).with_name("model.pt"),
        help="Path to training .pth checkpoint or TorchScript .pt model (default: public/model.pt)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("datasets/synthetic"),
        help="Dataset directory containing images/ and masks/ (default: datasets/synthetic)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Evaluation batch size (default: 8 on CUDA, 4 on CPU)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader worker processes (default: 2 on CUDA, 0 on CPU)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Evaluation device: 'cuda' or 'cpu' (default: cuda if available)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=VALIDATION_THRESHOLD,
        help=f"Probability threshold for mask binarization (default: {VALIDATION_THRESHOLD})",
    )
    parser.add_argument(
        "--weights",
        choices=("ema", "raw"),
        default="ema",
        help="Checkpoint weight set to evaluate for .pth files (default: ema)",
    )
    parser.add_argument(
        "--aux",
        action="store_true",
        help="Also evaluate internal half-resolution tail/distance heads (.pth only)",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable FP16 autocast on CUDA",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable evaluation progress output on stderr",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to save full metrics JSON",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    batch_size = (
        args.batch_size if args.batch_size is not None else (8 if device.type == "cuda" else 4)
    )
    num_workers = (
        args.num_workers if args.num_workers is not None else (2 if device.type == "cuda" else 0)
    )
    dataset_path = resolve_dataset_path(args.dataset)
    model, model_info = load_eval_model(
        args.checkpoint,
        device=device,
        weights=args.weights,
        require_aux=args.aux,
    )

    dataset = BubbleEvalDataset(
        dataset_path,
        input_height=model_info["input_height"],
        input_width=model_info["input_width"],
        compute_distance=args.aux,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=_worker_init if num_workers > 0 else None,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    start = time.perf_counter()
    metrics = evaluate(
        model,
        loader,
        device,
        threshold=args.threshold,
        use_amp=not args.no_amp,
        eval_aux=args.aux,
        show_progress=not args.no_progress,
    )
    elapsed = time.perf_counter() - start

    print(format_report(args.checkpoint, dataset_path, device, model_info, metrics, elapsed))

    if args.json_out is not None:
        payload = {
            "checkpoint": str(args.checkpoint),
            "dataset": str(dataset_path),
            "device": str(device),
            "threshold": args.threshold,
            "model": model_info,
            "elapsed_sec": elapsed,
            "metrics": metrics,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
