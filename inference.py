"""Generate a speech bubble overlay from a page image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from model import load_model


# Model input dimensions (fixed static resolution)
HEIGHT = 1024
WIDTH = 768

# ImageNet grayscale normalization
NORM_MEAN = 0.449
NORM_STD = 0.226

# Postprocessing thresholds
MASK_THRESHOLD = 0.50
MIN_AREA_PIXELS = 60
MIN_INSTANCE_SCORE = 0.60


@torch.inference_mode()
def inference(
    image_path: Path | str,
    model_path: Path | str,
    output: Path | str,
    device: str = "cpu",
) -> Path:
    """Run bubble segmentation on an image and save an annotated overlay."""
    image_path = Path(image_path)
    model_path = Path(model_path)
    output = Path(output)

    # Load model weights (support checkpoint dict and TorchScript fallback)
    try:
        model, _ = load_model(model_path)
        model = model.to(device)
    except Exception:
        model = torch.jit.load(str(model_path), map_location=device).eval()

    # Read image (Unicode-safe for non-ASCII paths)
    image_data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]

    # Letterbox resize maintaining aspect ratio with canvas padding
    scale = min(WIDTH / width, HEIGHT / height)
    rw = max(1, round(width * scale))
    rh = max(1, round(height * scale))
    pad_x = (WIDTH - rw) // 2
    pad_y = (HEIGHT - rh) // 2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized_gray = cv2.resize(gray, (rw, rh), interpolation=interp)

    canvas = np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)
    canvas[pad_y : pad_y + rh, pad_x : pad_x + rw] = resized_gray

    # Normalize and convert to PyTorch tensor (NCHW format)
    tensor = torch.from_numpy(canvas).float()[None, None] / 255.0
    normalized = ((tensor - NORM_MEAN) / NORM_STD).to(device)

    # Run inference and apply sigmoid to logits
    probabilities = model(normalized).sigmoid()[0].cpu().numpy()

    # Unpad and resize mask and confidence maps to original dimensions
    mask_cropped = probabilities[0, pad_y : pad_y + rh, pad_x : pad_x + rw]
    conf_cropped = probabilities[2, pad_y : pad_y + rh, pad_x : pad_x + rw]
    mask = cv2.resize(mask_cropped, (width, height), interpolation=cv2.INTER_LINEAR)
    confidence = cv2.resize(conf_cropped, (width, height), interpolation=cv2.INTER_LINEAR)

    # Find connected bubble instances
    binary_mask = (mask >= MASK_THRESHOLD).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    overlay = image.copy()
    colors = image.copy()

    # Process and draw each detected bubble instance
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < MIN_AREA_PIXELS:
            continue

        region = labels == label
        score = float((mask[region].mean() + confidence[region].mean()) / 2.0)
        if score < MIN_INSTANCE_SCORE:
            continue

        # Deterministic random color per instance
        color = tuple(int(c) for c in np.random.default_rng(label).integers(64, 240, 3))
        colors[region] = color

        # Draw anti-aliased contour boundary
        contours, _ = cv2.findContours(
            region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color, 2, lineType=cv2.LINE_AA)

        # Draw instance confidence score
        box_x = stats[label, cv2.CC_STAT_LEFT]
        box_y = stats[label, cv2.CC_STAT_TOP]
        cv2.putText(
            overlay,
            f"{score:.2f}",
            (int(box_x), max(16, int(box_y) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    # Blend colored masks with original image (alpha = 0.7 original, 0.3 mask)
    overlay = cv2.addWeighted(overlay, 0.7, colors, 0.3, 0)

    # Save output image (Unicode-safe)
    output.parent.mkdir(parents=True, exist_ok=True)
    ext = output.suffix or ".jpg"
    ok, encoded = cv2.imencode(ext, overlay)
    if not ok:
        raise OSError(f"Could not encode overlay: {output}")
    encoded.tofile(str(output))

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "image",
        type=Path,
        help="Path to input manga/comic page image",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).with_name("model.pt"),
        help="Path to PyTorch checkpoint or TorchScript model (default: model.pt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("overlay.jpg"),
        help="Path to save output overlay image (default: overlay.jpg)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device: 'cuda' or 'cpu' (default: cuda if available)",
    )
    args = parser.parse_args()

    saved_path = inference(args.image, args.model, args.output, args.device)
    print(f"Saved {saved_path}")


if __name__ == "__main__":
    main()
