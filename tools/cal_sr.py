import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_image(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float64)


def resize_to(image: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    pil = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
    pil = pil.resize((w, h), Image.BICUBIC)
    return np.asarray(pil, dtype=np.float64)


def compute_psnr(sr: np.ndarray, gt: np.ndarray, data_range: float = 255.0) -> float:
    mse = np.mean((sr - gt) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(data_range / math.sqrt(mse))


def gaussian_kernel(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    axis = np.arange(size) - size // 2
    kernel_1d = np.exp(-(axis ** 2) / (2 * sigma ** 2))
    kernel_1d /= kernel_1d.sum()
    return np.outer(kernel_1d, kernel_1d)


def filter2d_channel(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    pad = kernel.shape[0] // 2
    padded = np.pad(image, ((pad, pad), (pad, pad)), mode="reflect")
    out = np.zeros_like(image, dtype=np.float64)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            patch = padded[y : y + kernel.shape[0], x : x + kernel.shape[1]]
            out[y, x] = np.sum(patch * kernel)
    return out


def compute_ssim(sr: np.ndarray, gt: np.ndarray, data_range: float = 255.0) -> float:
    kernel = gaussian_kernel()
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    scores = []

    for c in range(sr.shape[2]):
        x = sr[:, :, c]
        y = gt[:, :, c]
        mu_x = filter2d_channel(x, kernel)
        mu_y = filter2d_channel(y, kernel)
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x_sq = filter2d_channel(x * x, kernel) - mu_x_sq
        sigma_y_sq = filter2d_channel(y * y, kernel) - mu_y_sq
        sigma_xy = filter2d_channel(x * y, kernel) - mu_xy

        numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
        scores.append(np.mean(numerator / denominator))

    return float(np.mean(scores))


def collect_images(root: Path) -> dict[str, Path]:
    files = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            name = path.stem
            if name in files:
                raise ValueError(
                    f"Duplicate image name '{name}' found under {root}: "
                    f"{files[name]} and {path}. Please keep image names unique."
                )
            files[name] = path
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute PSNR and SSIM between super-resolved images and ground-truth images. "
            "Images are matched by file name without extension."
        )
    )
    parser.add_argument("--sr-dir", required=True, type=Path, help="Directory containing SR output images.")
    parser.add_argument("--gt-dir", required=True, type=Path, help="Directory containing ground-truth images.")
    parser.add_argument("--csv", type=Path, default=None, help="Optional path to save per-image results as CSV.")
    parser.add_argument(
        "--resize",
        choices=["none", "sr_to_gt", "gt_to_sr"],
        default="none",
        help="How to handle size mismatch. Default: none.",
    )
    args = parser.parse_args()

    sr_files = collect_images(args.sr_dir)
    gt_files = collect_images(args.gt_dir)
    names = sorted(set(sr_files) & set(gt_files))
    if not names:
        raise SystemExit("No matched image pairs found. Files are matched by file name without extension.")

    rows = []
    for name in names:
        sr = load_image(sr_files[name])
        gt = load_image(gt_files[name])

        if sr.shape[:2] != gt.shape[:2]:
            if args.resize == "sr_to_gt":
                sr = resize_to(sr, gt.shape[:2])
            elif args.resize == "gt_to_sr":
                gt = resize_to(gt, sr.shape[:2])
            else:
                raise ValueError(
                    f"Size mismatch for {name}: SR {sr.shape[:2]} vs GT {gt.shape[:2]}. "
                    "Use --resize sr_to_gt or --resize gt_to_sr."
                )

        psnr = compute_psnr(sr, gt)
        ssim = compute_ssim(sr, gt)
        rows.append({"name": name, "psnr": psnr, "ssim": ssim})
        print(f"{name}: PSNR={psnr:.4f}, SSIM={ssim:.6f}")

    finite_psnr = [row["psnr"] for row in rows if math.isfinite(row["psnr"])]
    mean_psnr = float(np.mean(finite_psnr)) if finite_psnr else float("inf")
    mean_ssim = float(np.mean([row["ssim"] for row in rows]))
    print(f"\nMatched images: {len(rows)}")
    print(f"Average PSNR: {mean_psnr:.4f}")
    print(f"Average SSIM: {mean_ssim:.6f}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "psnr", "ssim"])
            writer.writeheader()
            writer.writerows(rows)
            writer.writerow({"name": "AVERAGE", "psnr": mean_psnr, "ssim": mean_ssim})
        print(f"Saved CSV to: {args.csv}")


if __name__ == "__main__":
    main()
