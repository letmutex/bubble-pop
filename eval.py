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
import os
import sys
import time
from collections import deque
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import (
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    NORM_MEAN,
    NORM_STD,
    BubbleEvalDataset,
    resolve_dataset_path,
)
from model import BubbleUNet, load_model


VALIDATION_THRESHOLD = 0.50
BOUNDARY_TOLERANCE_PX = 1
THIN_NEGATIVE_RADIUS = 4
CONFIDENCE_BACKGROUND_WEIGHT = 0.25
SCORE_EPS = 1e-4


def _separable_max_pool2d(tensor: torch.Tensor, radius: int) -> torch.Tensor:
    """Exact square (2*radius+1) max-pool decomposed into 1D passes for radius > 1."""
    if radius <= 0:
        return tensor
    k = 2 * radius + 1
    if radius == 1:
        return F.max_pool2d(tensor, kernel_size=3, stride=1, padding=1)
    return F.max_pool2d(
        F.max_pool2d(tensor, kernel_size=(k, 1), stride=1, padding=(radius, 0)),
        kernel_size=(1, k),
        stride=1,
        padding=(0, radius),
    )


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
    target_count = focus_target.sum(dim=(1, 2, 3), dtype=torch.float32)
    true_positive = (predicted_mask * focus_target).sum(dim=(1, 2, 3), dtype=torch.float32)
    recall = true_positive / target_count.clamp(min=1.0)
    neighborhood = _separable_max_pool2d(focus_target, radius)
    local_negative = neighborhood * (1.0 - ground_truth)
    evaluation_region = (focus_target + local_negative).clamp(max=1.0)
    prediction_count = (predicted_mask * evaluation_region).sum(
        dim=(1, 2, 3), dtype=torch.float32
    )
    precision = true_positive / prediction_count.clamp(min=1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp(min=1e-6)
    valid = (target_count > 0).float()
    return precision, recall, f1, valid


def _focused_scores_from_neighborhood(
    predicted_mask: torch.Tensor,
    focus_target: torch.Tensor,
    neighborhood: torch.Tensor,
    inv_ground_truth: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample focused scores reusing a pre-dilated neighborhood tensor."""
    target_count = focus_target.sum(dim=(1, 2, 3), dtype=torch.float32)
    true_positive = (predicted_mask * focus_target).sum(dim=(1, 2, 3), dtype=torch.float32)
    recall = true_positive / target_count.clamp(min=1.0)
    evaluation_region = (focus_target + neighborhood * inv_ground_truth).clamp(max=1.0)
    prediction_count = (predicted_mask * evaluation_region).sum(
        dim=(1, 2, 3), dtype=torch.float32
    )
    precision = true_positive / prediction_count.clamp(min=1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp(min=1e-6)
    valid = (target_count > 0).float()
    return precision, recall, f1, valid


def fuse_conv_bn_eval(module: torch.nn.Module) -> torch.nn.Module:
    """Fold BatchNorm2d layers into Conv2d weights in-place for eval-mode models."""
    module.eval()
    for child in module.children():
        if (
            hasattr(child, "conv")
            and hasattr(child, "bn")
            and isinstance(child.conv, torch.nn.Conv2d)
            and isinstance(child.bn, torch.nn.BatchNorm2d)
        ):
            child.conv = torch.nn.utils.fuse_conv_bn_eval(child.conv, child.bn)
            child.bn = torch.nn.Identity()
        elif (
            isinstance(child, torch.nn.Sequential)
            and len(child) >= 2
            and isinstance(child[0], torch.nn.Conv2d)
            and isinstance(child[1], torch.nn.BatchNorm2d)
        ):
            child[0] = torch.nn.utils.fuse_conv_bn_eval(child[0], child[1])
            child[1] = torch.nn.Identity()
        else:
            fuse_conv_bn_eval(child)
    return module


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
        model = fuse_conv_bn_eval(model)
        model = model.to(device)
        if device.type == "cuda":
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


class _ThreadedBatchLoader:
    """Bounded threaded batch prefetch loader for GIL-releasing OpenCV preprocessing."""

    def __init__(
        self,
        dataset: BubbleEvalDataset,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = False,
    ) -> None:
        self.dataset = dataset
        self.batch_size = max(int(batch_size), 1)
        self.num_workers = max(int(num_workers), 0)
        self.pin_memory = bool(pin_memory)

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)

    def _collate_batch(self, indices: range) -> dict[str, torch.Tensor]:
        items = [self.dataset[i] for i in indices]
        batch = {k: torch.stack([item[k] for item in items]) for k in items[0]}
        if self.pin_memory:
            batch = {k: v.pin_memory() for k, v in batch.items()}
        return batch

    def __iter__(self):
        total = len(self.dataset)
        batch_ranges = [
            range(start, min(start + self.batch_size, total))
            for start in range(0, total, self.batch_size)
        ]
        if self.num_workers <= 0 or len(batch_ranges) <= 1:
            for indices in batch_ranges:
                yield self._collate_batch(indices)
            return

        prev_cv2_threads = cv2.getNumThreads()
        cv2.setNumThreads(1)
        try:
            max_in_flight = min(self.num_workers + 2, len(batch_ranges))
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                it = iter(batch_ranges)
                pending = deque(
                    executor.submit(self._collate_batch, next(it))
                    for _ in range(max_in_flight)
                )
                while pending:
                    yield pending.popleft().result()
                    try:
                        pending.append(executor.submit(self._collate_batch, next(it)))
                    except StopIteration:
                        pass
        finally:
            cv2.setNumThreads(prev_cv2_threads)


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader | Iterable[dict[str, torch.Tensor]],
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
    cuda_amp = use_amp and device.type == "cuda"
    mps_fp16 = use_amp and device.type == "mps" and isinstance(model, BubbleUNet)
    use_channels_last = device.type == "cuda" and isinstance(model, BubbleUNet)
    bin_dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32
    use_mps_acc = device.type == "mps"

    use_logit_thresh = 0.0 < threshold < 1.0
    logit_threshold = math.log(threshold / (1.0 - threshold)) if use_logit_thresh else 0.0
    inv_norm_scale = 1.0 / (255.0 * NORM_STD)
    norm_bias = NORM_MEAN / NORM_STD

    first_param = next(model.parameters(), None)
    orig_model_dtype = first_param.dtype if first_param is not None else None
    if mps_fp16 and orig_model_dtype != torch.float16:
        model.to(dtype=torch.float16)

    # Device-side float64 accumulator on CUDA/CPU; batched per-sample float32 -> CPU float64
    # flush on MPS (which lacks float64) so every pixel count remains exact with zero sync stalls.
    num_slots = 37
    if use_mps_acc:
        acc_cpu = torch.zeros(num_slots, dtype=torch.float64)
        mps_chunks: list[torch.Tensor] = []
    else:
        acc = torch.zeros(num_slots, dtype=torch.float64, device=device)

    total_pixels = 0
    total_samples = 0
    dataset_len = len(loader.dataset) if hasattr(loader, "dataset") else 0
    progress = ProgressBar(dataset_len, enabled=show_progress and dataset_len > 0)

    def _flush_mps_chunks() -> None:
        if mps_chunks:
            acc_cpu.add_(torch.cat(mps_chunks, dim=0).cpu().double().sum(dim=0))
            mps_chunks.clear()

    try:
        for batch in loader:
            raw_image = batch["image"]
            if raw_image.dtype == torch.uint8:
                image = raw_image.to(device, dtype=torch.float32, non_blocking=True)
                image.mul_(inv_norm_scale).sub_(norm_bias)
            else:
                image = raw_image.to(device, non_blocking=True)
            if mps_fp16:
                image = image.to(dtype=torch.float16)
            elif use_channels_last:
                image = image.contiguous(memory_format=torch.channels_last)

            gt_mask = batch["mask"].to(device, dtype=bin_dtype, non_blocking=True)
            gt_boundary = batch["boundary"].to(device, dtype=bin_dtype, non_blocking=True)
            gt_thin = batch["thin"].to(device, dtype=bin_dtype, non_blocking=True)
            gt_narrow = batch["narrow"].to(device, dtype=bin_dtype, non_blocking=True)
            gt_conf = batch["conf"].to(device, non_blocking=True)
            gt_distance = (
                batch["distance"].to(device, non_blocking=True) if eval_aux else None
            )

            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=cuda_amp
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

            logits_f = logits.float()
            pred_prob = torch.sigmoid(logits_f[:, 0:1])
            conf_prob = torch.sigmoid(logits_f[:, 2:3])

            predicted_mask = (pred_prob > threshold).to(bin_dtype)
            if use_logit_thresh:
                pred_boundary = (logits_f[:, 1:2] > logit_threshold).to(bin_dtype)
            else:
                pred_boundary = (torch.sigmoid(logits_f[:, 1:2]) > threshold).to(bin_dtype)

            # Global mask metrics
            intersection = (predicted_mask * gt_mask).sum(dim=(1, 2, 3), dtype=torch.float32)
            predicted_sum = predicted_mask.sum(dim=(1, 2, 3), dtype=torch.float32)
            target_sum = gt_mask.sum(dim=(1, 2, 3), dtype=torch.float32)

            # Auxiliary boundary head (channel 1)
            b_inter = (pred_boundary * gt_boundary).sum(dim=(1, 2, 3), dtype=torch.float32)
            b_pred_sum = pred_boundary.sum(dim=(1, 2, 3), dtype=torch.float32)
            b_gt_sum = gt_boundary.sum(dim=(1, 2, 3), dtype=torch.float32)
            b_union = b_pred_sum + b_gt_sum - b_inter

            # Final-mask contour metrics (single batched 3x3 max_pool2d for both contours)
            inv_gt_mask = 1.0 - gt_mask
            contour_pool = F.max_pool2d(
                torch.cat((predicted_mask, gt_mask, 1.0 - predicted_mask, inv_gt_mask), dim=1),
                kernel_size=3,
                stride=1,
                padding=1,
            )
            pred_contour = ((contour_pool[:, 0:1] + contour_pool[:, 2:3]) > 1.0).to(bin_dtype)
            gt_contour = ((contour_pool[:, 1:2] + contour_pool[:, 3:4]) > 1.0).to(bin_dtype)

            # Single batched dilation for auxiliary boundaries & mask contours
            dilated_all = _separable_max_pool2d(
                torch.cat((gt_boundary, pred_boundary, gt_contour, pred_contour), dim=1),
                boundary_tolerance_px,
            )
            aux_matched_pred = (pred_boundary * dilated_all[:, 0:1]).sum(
                dim=(1, 2, 3), dtype=torch.float32
            )
            aux_matched_gt = (gt_boundary * dilated_all[:, 1:2]).sum(
                dim=(1, 2, 3), dtype=torch.float32
            )
            mask_matched_pred = (pred_contour * dilated_all[:, 2:3]).sum(
                dim=(1, 2, 3), dtype=torch.float32
            )
            mask_matched_gt = (gt_contour * dilated_all[:, 3:4]).sum(
                dim=(1, 2, 3), dtype=torch.float32
            )
            pred_contour_sum = pred_contour.sum(dim=(1, 2, 3), dtype=torch.float32)
            gt_contour_sum = gt_contour.sum(dim=(1, 2, 3), dtype=torch.float32)

            # Thin & narrow focused scores (single batched separable pooling)
            tn_neighborhood = _separable_max_pool2d(
                torch.cat((gt_thin, gt_narrow), dim=1),
                THIN_NEGATIVE_RADIUS,
            )
            thin_p, thin_r, thin_f1, thin_valid = _focused_scores_from_neighborhood(
                predicted_mask, gt_thin, tn_neighborhood[:, 0:1], inv_gt_mask
            )
            narrow_p, narrow_r, narrow_f1, narrow_valid = _focused_scores_from_neighborhood(
                predicted_mask, gt_narrow, tn_neighborhood[:, 1:2], inv_gt_mask
            )

            # Confidence head calibration (avoids full-res conf_bg allocation)
            hw = float(gt_mask.shape[-2] * gt_mask.shape[-1])
            conf_error = (conf_prob - gt_conf).abs()
            conf_fg = ((gt_conf > 0) | (gt_mask > 0)).float()
            conf_fg_cnt = conf_fg.sum(dim=(1, 2, 3))
            conf_bg_cnt = hw - conf_fg_cnt
            conf_err_total = conf_error.sum(dim=(1, 2, 3))
            conf_fg_err = (conf_error * conf_fg).sum(dim=(1, 2, 3))
            conf_fg_mae = conf_fg_err / conf_fg_cnt.clamp(min=1.0)
            conf_bg_mae = (conf_err_total - conf_fg_err) / conf_bg_cnt.clamp(min=1.0)
            has_fg = (conf_fg_cnt > 0).float()
            has_bg = (conf_bg_cnt > 0).float()
            conf_mae = (
                has_fg * conf_fg_mae + CONFIDENCE_BACKGROUND_WEIGHT * has_bg * conf_bg_mae
            ) / (has_fg + CONFIDENCE_BACKGROUND_WEIGHT * has_bg).clamp(min=1e-6)

            gt_mask_f = gt_mask.float() if bin_dtype != torch.float32 else gt_mask
            fg_prob_sum = (pred_prob * gt_mask_f).sum(dim=(1, 2, 3))
            bg_prob_sum = pred_prob.sum(dim=(1, 2, 3)) - fg_prob_sum
            bg_pix_sum = hw - target_sum
            empty_pages = target_sum == 0

            has_aux_outputs = (
                eval_aux
                and tail_half is not None
                and distance_half is not None
                and gt_distance is not None
            )
            if has_aux_outputs:
                if use_logit_thresh:
                    pred_tail = (tail_half.float() > logit_threshold).to(bin_dtype)
                else:
                    pred_tail = (torch.sigmoid(tail_half.float()) > threshold).to(bin_dtype)
                tail_target = F.adaptive_max_pool2d(gt_narrow, pred_tail.shape[-2:])
                dist_pred = distance_half.float()
                dist_target = F.interpolate(
                    gt_distance.float(),
                    size=dist_pred.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                dist_band = (dist_target.abs() < 0.999).float()
                aux_tail_inter = (pred_tail * tail_target).sum(dim=(1, 2, 3), dtype=torch.float32)
                aux_tail_pred = pred_tail.sum(dim=(1, 2, 3), dtype=torch.float32)
                aux_tail_gt = tail_target.sum(dim=(1, 2, 3), dtype=torch.float32)
                aux_dist_err = ((dist_pred - dist_target).abs() * dist_band).sum(dim=(1, 2, 3))
                aux_dist_band = dist_band.sum(dim=(1, 2, 3))

            batch_size_cur = int(image.shape[0])
            total_pixels += int(predicted_mask.numel())
            total_samples += batch_size_cur

            if use_mps_acc:
                zeros_b = torch.zeros(batch_size_cur, dtype=torch.float32, device=device)
                per_sample = torch.stack(
                    (
                        intersection,
                        predicted_sum,
                        target_sum,
                        b_inter,
                        b_union,
                        aux_matched_pred,
                        b_pred_sum,
                        aux_matched_gt,
                        b_gt_sum,
                        mask_matched_pred,
                        pred_contour_sum,
                        mask_matched_gt,
                        gt_contour_sum,
                        thin_r * thin_valid,
                        thin_p * thin_valid,
                        thin_f1 * thin_valid,
                        thin_valid,
                        narrow_r * narrow_valid,
                        narrow_p * narrow_valid,
                        narrow_f1 * narrow_valid,
                        narrow_valid,
                        conf_mae,
                        conf_bg_mae,
                        fg_prob_sum,
                        target_sum,
                        bg_prob_sum,
                        bg_pix_sum,
                        predicted_sum,
                        zeros_b,
                        empty_pages.float(),
                        (empty_pages & (predicted_sum > 0)).float(),
                        zeros_b,
                        aux_tail_inter if has_aux_outputs else zeros_b,
                        aux_tail_pred if has_aux_outputs else zeros_b,
                        aux_tail_gt if has_aux_outputs else zeros_b,
                        aux_dist_err if has_aux_outputs else zeros_b,
                        aux_dist_band if has_aux_outputs else zeros_b,
                    ),
                    dim=1,
                )
                mps_chunks.append(per_sample)
                if len(mps_chunks) >= 64:
                    _flush_mps_chunks()
            else:
                acc[0] += intersection.sum()
                acc[1] += predicted_sum.sum()
                acc[2] += target_sum.sum()
                acc[3] += b_inter.sum()
                acc[4] += b_union.sum()
                acc[5] += aux_matched_pred.sum()
                acc[6] += b_pred_sum.sum()
                acc[7] += aux_matched_gt.sum()
                acc[8] += b_gt_sum.sum()
                acc[9] += mask_matched_pred.sum()
                acc[10] += pred_contour_sum.sum()
                acc[11] += mask_matched_gt.sum()
                acc[12] += gt_contour_sum.sum()
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
                acc[23] += fg_prob_sum.sum()
                acc[24] += target_sum.sum()
                acc[25] += bg_prob_sum.sum()
                acc[26] += bg_pix_sum.sum()
                acc[27] += predicted_sum.sum()
                acc[29] += empty_pages.sum()
                acc[30] += (empty_pages & (predicted_sum > 0)).sum()
                if has_aux_outputs:
                    acc[32] += aux_tail_inter.sum()
                    acc[33] += aux_tail_pred.sum()
                    acc[34] += aux_tail_gt.sum()
                    acc[35] += aux_dist_err.sum()
                    acc[36] += aux_dist_band.sum()

            progress.update(batch_size_cur)
    finally:
        progress.close()
        if mps_fp16 and orig_model_dtype is not None and orig_model_dtype != torch.float16:
            model.to(dtype=orig_model_dtype)

    if use_mps_acc:
        _flush_mps_chunks()
        vals = acc_cpu.tolist()
    else:
        vals = acc.tolist()
    vals[28] = float(total_pixels)
    vals[31] = float(total_samples)
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
        help="Evaluation batch size (default: 8 on CUDA/MPS, 1 on CPU)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Background prefetch threads (default: up to 4 on CUDA/MPS, up to 2 on CPU)",
    )
    parser.add_argument(
        "--device",
        default=_default_device(),
        help="Evaluation device: 'cuda', 'mps', or 'cpu' (default: cuda > mps > cpu)",
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
        help="Disable FP16 acceleration on CUDA/MPS",
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
    is_gpu = device.type in ("cuda", "mps")
    cpu_count = os.cpu_count() or 1
    batch_size = (
        args.batch_size if args.batch_size is not None else (8 if is_gpu else 1)
    )
    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else min(4 if is_gpu else 2, cpu_count)
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
        compact=True,
    )
    loader = _ThreadedBatchLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
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
