"""Evaluation dataset for BubbleUNet segmentation models.

Expected directory structure:

    <dataset_root>/
    ├── images/                 # Required: input pages (.jpg, .jpeg, .png, .webp)
    │   └── [subfolders/]
    │       └── <stem>.jpg
    ├── masks/                  # Required: binary foreground masks (.png)
    │   └── [subfolders/]
    │       └── <stem>.png      # 8-bit grayscale; > 0 is foreground, same HxW as image
    └── labels_yolo/            # Optional: per-instance YOLO segmentation labels (.txt)
        └── [subfolders/]
            └── <stem>.txt

File formats:
  - images/:      Any grayscale or color image readable by OpenCV (.jpg, .jpeg,
                  .png, .webp), nested arbitrarily under images/. Each image is
                  converted to grayscale, centered-letterboxed onto a white (255)
                  canvas of size (input_height, input_width), and normalized with
                  (pixel / 255.0 - 0.449) / 0.226.
  - masks/:       Required 8-bit single-channel PNG matching each image's relative
                  path under images/ with a .png suffix and identical (H, W).
                  Non-zero pixels are treated as bubble foreground (1.0); resized
                  with nearest-neighbor interpolation onto the letterbox canvas.
  - labels_yolo/: Optional UTF-8 text file matching each image's relative path
                  with a .txt suffix. One bubble instance per non-empty line:

                      0 x1 y1 x2 y2 ... xn yn score

                  where class id is 0, (xi, yi) are >= 3 normalized polygon
                  vertices in [0, 1], and trailing `score` is in [0, 1]. Used to
                  preserve touching-bubble boundary walls, compute per-instance
                  narrow protrusions (tails) for instances with score >= 0.50,
                  and build the confidence target map. If absent, evaluation
                  falls back cleanly to the hard mask alone.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 768
NORM_MEAN = 0.449
NORM_STD = 0.226

CONFIDENCE_LOW = 0.50
BOUNDARY_KERNEL_SIZE = 3
THIN_FOREGROUND_DISTANCE_PX = 4.0
NARROW_OPENING_BODY_RATIO = 0.20
NARROW_OPENING_MIN_RADIUS_PX = 3
NARROW_OPENING_MAX_RADIUS_PX = 24
NARROW_MIN_BODY_RADIUS_PX = 7.0
SDF_TRUNCATION_PX = 12.0

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

_NARROW_ELLIPSE_KERNELS: dict[int, np.ndarray] = {
    radius: cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    for radius in range(NARROW_OPENING_MIN_RADIUS_PX, NARROW_OPENING_MAX_RADIUS_PX + 1)
}


def imread_unicode(path: Path, flags: int) -> np.ndarray | None:
    """Unicode-safe cv2.imread that strips PNG iCCP chunks to silence libpng warnings."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if not raw:
        return None
    if raw.startswith(b"\x89PNG\r\n\x1a\n") and b"iCCP" in raw:
        chunks = [raw[:8]]
        pos = 8
        while pos < len(raw):
            if pos + 12 > len(raw):
                break
            length = int.from_bytes(raw[pos : pos + 4], "big")
            end = pos + 12 + length
            if end > len(raw):
                break
            if raw[pos + 4 : pos + 8] != b"iCCP":
                chunks.append(raw[pos:end])
            pos = end
        else:
            raw = b"".join(chunks)
    return cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), flags)


def read_scored_polygons(
    label_path: Path, width: int, height: int
) -> list[tuple[np.ndarray, float]]:
    """Read YOLO segmentation polygons with trailing confidence score."""
    if not label_path.is_file():
        return []
    scored_polygons: list[tuple[np.ndarray, float]] = []
    with label_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            parts = list(map(float, line.split()))
            if len(parts) < 8 or parts[0] != 0.0:
                raise ValueError(f"Invalid polygon label at {label_path}:{line_number}")
            score = float(parts[-1])
            coords = np.asarray(parts[1:-1], dtype=np.float32)
            if coords.size < 6 or coords.size % 2:
                raise ValueError(f"Invalid polygon coordinates at {label_path}:{line_number}")
            polygon = coords.reshape(-1, 2)
            polygon[:, 0] *= width
            polygon[:, 1] *= height
            scored_polygons.append((polygon, score))
    return scored_polygons


def accumulate_instance_narrow(
    instance_mask: np.ndarray,
    narrow_residue: np.ndarray,
    origin_x: int = 0,
    origin_y: int = 0,
) -> None:
    """Extract scale-aware narrow protrusions (e.g. speech tails) from one instance."""
    component_count, component_labels, component_stats, _ = (
        cv2.connectedComponentsWithStats(instance_mask, connectivity=8)
    )
    for component_id in range(1, component_count):
        x, y, comp_w, comp_h = component_stats[component_id, :4]
        if component_count == 2:
            component_uint8 = instance_mask[y : y + comp_h, x : x + comp_w]
        else:
            label_crop = component_labels[y : y + comp_h, x : x + comp_w]
            component_uint8 = (label_crop == component_id).astype(np.uint8)
        component_dist = cv2.distanceTransform(component_uint8, cv2.DIST_L2, 5)
        body_radius = float(component_dist.max())
        if body_radius < NARROW_MIN_BODY_RADIUS_PX:
            continue
        opening_radius = int(round(body_radius * NARROW_OPENING_BODY_RATIO))
        opening_radius = min(
            max(opening_radius, NARROW_OPENING_MIN_RADIUS_PX),
            NARROW_OPENING_MAX_RADIUS_PX,
        )
        opening_kernel = _NARROW_ELLIPSE_KERNELS[opening_radius]
        opened = cv2.morphologyEx(component_uint8, cv2.MORPH_OPEN, opening_kernel)
        residue_crop = component_uint8 & (opened == 0)
        target_crop = narrow_residue[
            origin_y + y : origin_y + y + comp_h,
            origin_x + x : origin_x + x + comp_w,
        ]
        np.maximum(target_crop, residue_crop, out=target_crop)


def resolve_dataset_path(dataset_arg: Path) -> Path:
    """Resolve dataset path relative to CWD or repository root."""
    if dataset_arg.exists():
        return dataset_arg
    repo_fallback = Path(__file__).resolve().parent.parent / dataset_arg
    if repo_fallback.exists():
        return repo_fallback
    raise FileNotFoundError(f"Dataset path not found: {dataset_arg}")


class BubbleEvalDataset(Dataset):
    """Deterministic validation dataset matching training preprocessing without unused targets."""

    def __init__(
        self,
        root_dir: Path,
        input_height: int = DEFAULT_HEIGHT,
        input_width: int = DEFAULT_WIDTH,
        confidence_low: float = CONFIDENCE_LOW,
        compute_distance: bool = False,
        compact: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.img_root = self.root_dir / "images"
        self.mask_root = self.root_dir / "masks"
        self.label_root = self.root_dir / "labels_yolo"
        self.input_height = int(input_height)
        self.input_width = int(input_width)
        self.confidence_low = float(confidence_low)
        self.compute_distance = bool(compute_distance)
        self.compact = bool(compact)

        if not self.img_root.is_dir():
            raise FileNotFoundError(f"Images directory not found: {self.img_root}")
        if not self.mask_root.is_dir():
            raise FileNotFoundError(f"Masks directory not found: {self.mask_root}")

        self.samples: list[tuple[Path, Path, Path]] = []
        missing_masks: list[str] = []
        for path in sorted(self.img_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            rel_stem = path.relative_to(self.img_root).with_suffix("")
            mask_path = self.mask_root / rel_stem.with_suffix(".png")
            label_path = self.label_root / rel_stem.with_suffix(".txt")
            if not mask_path.is_file():
                missing_masks.append(str(mask_path))
                continue
            self.samples.append((path, mask_path, label_path))

        if missing_masks:
            raise FileNotFoundError(
                f"Missing {len(missing_masks)} mask files (first 5):\n"
                + "\n".join(missing_masks[:5])
            )
        if not self.samples:
            raise RuntimeError(f"No evaluation images found in {self.img_root}")

        self._ellipse3 = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (BOUNDARY_KERNEL_SIZE, BOUNDARY_KERNEL_SIZE)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        img_path, mask_path, label_path = self.samples[index]
        image = imread_unicode(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Cannot read image: {img_path}")
        raw_mask = imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
        if raw_mask is None:
            raise RuntimeError(f"Cannot read mask: {mask_path}")

        height, width = image.shape[:2]
        if raw_mask.shape[:2] != (height, width):
            raise ValueError(
                f"Mask shape mismatch for {img_path}: {raw_mask.shape[:2]} != {(height, width)}"
            )
        hard_mask = (raw_mask > 0).astype(np.uint8)
        scored_polygons = read_scored_polygons(label_path, width, height)

        # Deterministic centered letterbox
        scale = min(self.input_width / max(width, 1), self.input_height / max(height, 1))
        resized_w = max(1, int(round(width * scale)))
        resized_h = max(1, int(round(height * scale)))
        pad_x = (self.input_width - resized_w) // 2
        pad_y = (self.input_height - resized_h) // 2
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR

        canvas_img = np.full((self.input_height, self.input_width), 255, dtype=np.uint8)
        canvas_img[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = cv2.resize(
            image, (resized_w, resized_h), interpolation=interp
        )

        resized_hard_mask = np.zeros((self.input_height, self.input_width), dtype=np.uint8)
        resized_hard_mask[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = (
            cv2.resize(hard_mask, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST) > 0
        )

        has_fg = bool(resized_hard_mask.any())

        # Thin foreground band + mask boundary
        if has_fg:
            fg_dist = cv2.distanceTransform(resized_hard_mask, cv2.DIST_L2, 5)
            canvas_thin_u8 = (
                (resized_hard_mask > 0) & (fg_dist <= THIN_FOREGROUND_DISTANCE_PX)
            ).astype(np.uint8)
            canvas_boundary = cv2.morphologyEx(
                resized_hard_mask, cv2.MORPH_GRADIENT, self._ellipse3
            )
        else:
            fg_dist = (
                np.zeros((self.input_height, self.input_width), dtype=np.float32)
                if self.compute_distance
                else None
            )
            canvas_thin_u8 = np.zeros_like(resized_hard_mask, dtype=np.uint8)
            canvas_boundary = np.zeros_like(resized_hard_mask, dtype=np.uint8)

        # Per-instance narrow protrusions + confidence map
        narrow_residue = np.zeros_like(resized_hard_mask, dtype=np.uint8)
        canvas_conf = np.zeros((self.input_height, self.input_width), dtype=np.float32)

        scale_x = resized_w / max(width, 1)
        scale_y = resized_h / max(height, 1)
        narrow_instance_count = 0

        for poly, score in scored_polygons:
            transformed = poly.copy()
            transformed[:, 0] = transformed[:, 0] * scale_x + pad_x
            transformed[:, 1] = transformed[:, 1] * scale_y + pad_y
            poly_i32 = np.round(transformed).astype(np.int32)

            cv2.fillPoly(canvas_conf, [poly_i32], float(score))
            if score < self.confidence_low:
                continue

            x, y, bw, bh = cv2.boundingRect(poly_i32)
            bx0 = max(x, 0)
            by0 = max(y, 0)
            bx1 = min(x + bw, self.input_width)
            by1 = min(y + bh, self.input_height)
            if bx1 <= bx0 or by1 <= by0:
                continue

            # 1-px padded crop preserves exact full-canvas 3x3 morphology boundary semantics
            px0 = max(bx0 - 1, 0)
            py0 = max(by0 - 1, 0)
            px1 = min(bx1 + 1, self.input_width)
            py1 = min(by1 + 1, self.input_height)
            local_poly = poly_i32 - np.array((px0, py0), dtype=np.int32)

            instance_mask = np.zeros((py1 - py0, px1 - px0), dtype=np.uint8)
            cv2.fillPoly(instance_mask, [local_poly], 1)
            inst_boundary = cv2.morphologyEx(
                instance_mask, cv2.MORPH_GRADIENT, self._ellipse3
            )
            boundary_slice = canvas_boundary[py0:py1, px0:px1]
            np.maximum(boundary_slice, inst_boundary, out=boundary_slice)

            instance_mask &= resized_hard_mask[py0:py1, px0:px1]
            if instance_mask.any():
                accumulate_instance_narrow(instance_mask, narrow_residue, px0, py0)
                narrow_instance_count += 1

        if not scored_polygons and has_fg:
            canvas_conf = resized_hard_mask.astype(np.float32)
        if narrow_instance_count == 0 and has_fg:
            accumulate_instance_narrow(resized_hard_mask, narrow_residue)

        if has_fg and narrow_residue.any():
            canvas_narrow_u8 = (
                (cv2.dilate(narrow_residue, self._ellipse3) > 0) & (resized_hard_mask > 0)
            ).astype(np.uint8)
        else:
            canvas_narrow_u8 = np.zeros_like(resized_hard_mask, dtype=np.uint8)

        if self.compact:
            item = {
                "image": torch.from_numpy(canvas_img).unsqueeze(0),
                "mask": torch.from_numpy(resized_hard_mask).unsqueeze(0),
                "boundary": torch.from_numpy(canvas_boundary).unsqueeze(0),
                "thin": torch.from_numpy(canvas_thin_u8).unsqueeze(0),
                "narrow": torch.from_numpy(canvas_narrow_u8).unsqueeze(0),
                "conf": torch.from_numpy(canvas_conf).unsqueeze(0),
            }
        else:
            image_tensor = (
                torch.from_numpy(canvas_img)
                .unsqueeze(0)
                .float()
                .div_(255.0)
                .sub_(NORM_MEAN)
                .div_(NORM_STD)
            )
            item = {
                "image": image_tensor,
                "mask": torch.from_numpy(resized_hard_mask.astype(np.float32)).unsqueeze(0),
                "boundary": torch.from_numpy(canvas_boundary.astype(np.float32)).unsqueeze(0),
                "thin": torch.from_numpy(canvas_thin_u8.astype(np.float32)).unsqueeze(0),
                "narrow": torch.from_numpy(canvas_narrow_u8.astype(np.float32)).unsqueeze(0),
                "conf": torch.from_numpy(canvas_conf).unsqueeze(0),
            }
        if self.compute_distance:
            bg_dist = cv2.distanceTransform(1 - resized_hard_mask, cv2.DIST_L2, 5)
            sdf = np.clip((fg_dist - bg_dist) / SDF_TRUNCATION_PX, -1.0, 1.0).astype(
                np.float32
            )
            item["distance"] = torch.from_numpy(sdf).unsqueeze(0)
        return item
