from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps


SUPPORTED_EXTENSIONS = {
    ".bmp",
    ".dib",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompose images into RGB, HSV, and HVI color spaces."
    )
    parser.add_argument(
        "--input",
        default="input",
        help="Input image file or directory. Defaults to ./input",
    )
    parser.add_argument(
        "--output",
        default="output",
        help="Output directory. Defaults to ./output",
    )
    parser.add_argument(
        "--spaces",
        nargs="+",
        choices=["rgb", "hsv", "hvi", "all"],
        default=["all"],
        help="Color spaces to export. Defaults to all.",
    )
    parser.add_argument(
        "--fixed-k",
        type=float,
        default=0.425,
        help=(
            "Deterministic k value for HVI decomposition. "
            "Used to avoid random HVI output when no trained k predictor is provided."
        ),
    )
    return parser.parse_args()


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(
            path
            for path in input_path.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def load_image(path: Path) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    if image.mode in {"RGBA", "LA"}:
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image.convert("RGBA")).convert("RGB")
    else:
        image = image.convert("RGB")
    return image


def save_grayscale(channel: np.ndarray, path: Path) -> None:
    Image.fromarray(channel.astype(np.uint8), mode="L").save(path)


def save_rgb(channel: np.ndarray, path: Path) -> None:
    Image.fromarray(channel.astype(np.uint8), mode="RGB").save(path)


def channel_triplet(single_channel: np.ndarray, channel_index: int) -> np.ndarray:
    output = np.zeros((single_channel.shape[0], single_channel.shape[1], 3), dtype=np.uint8)
    output[..., channel_index] = single_channel
    return output


def decompose_rgb(rgb_array: np.ndarray, output_dir: Path) -> None:
    names = ["r", "g", "b"]
    for index, name in enumerate(names):
        channel = rgb_array[..., index]
        save_grayscale(channel, output_dir / f"rgb_{name}_gray.png")
        save_rgb(channel_triplet(channel, index), output_dir / f"rgb_{name}_color.png")


def hsv_to_rgb_visualization(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    h = h.astype(np.float32) / 255.0
    s = s.astype(np.float32) / 255.0
    v = v.astype(np.float32) / 255.0

    i = np.floor(h * 6.0).astype(np.int32)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6

    rgb = np.zeros((*h.shape, 3), dtype=np.float32)
    selectors = [
        i == 0,
        i == 1,
        i == 2,
        i == 3,
        i == 4,
        i == 5,
    ]

    rgb[selectors[0]] = np.stack([v, t, p], axis=-1)[selectors[0]]
    rgb[selectors[1]] = np.stack([q, v, p], axis=-1)[selectors[1]]
    rgb[selectors[2]] = np.stack([p, v, t], axis=-1)[selectors[2]]
    rgb[selectors[3]] = np.stack([p, q, v], axis=-1)[selectors[3]]
    rgb[selectors[4]] = np.stack([t, p, v], axis=-1)[selectors[4]]
    rgb[selectors[5]] = np.stack([v, p, q], axis=-1)[selectors[5]]
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def rgb_to_hsv_components(rgb_array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = rgb_array.astype(np.float32) / 255.0
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    value = np.max(rgb, axis=-1)
    min_value = np.min(rgb, axis=-1)
    delta = value - min_value
    eps = 1e-8

    hue = np.zeros_like(value)
    mask = delta > eps

    mask_r = mask & (r == value)
    mask_g = mask & (g == value)
    mask_b = mask & (b == value)

    hue[mask_r] = np.mod((g[mask_r] - b[mask_r]) / (delta[mask_r] + eps), 6.0)
    hue[mask_g] = ((b[mask_g] - r[mask_g]) / (delta[mask_g] + eps)) + 2.0
    hue[mask_b] = ((r[mask_b] - g[mask_b]) / (delta[mask_b] + eps)) + 4.0
    hue = (hue / 6.0) % 1.0

    saturation = np.zeros_like(value)
    value_mask = value > eps
    saturation[value_mask] = delta[value_mask] / (value[value_mask] + eps)

    return hue, saturation, value


def decompose_hsv(image: Image.Image, output_dir: Path) -> None:
    hsv = image.convert("HSV")
    hsv_array = np.array(hsv, dtype=np.uint8)
    h, s, v = hsv_array[..., 0], hsv_array[..., 1], hsv_array[..., 2]

    save_grayscale(h, output_dir / "hsv_h_gray.png")
    save_grayscale(s, output_dir / "hsv_s_gray.png")
    save_grayscale(v, output_dir / "hsv_v_gray.png")

    hue_vis = hsv_to_rgb_visualization(h, np.full_like(h, 255), np.full_like(h, 255))
    sat_vis = hsv_to_rgb_visualization(h, s, np.full_like(v, 255))
    val_vis = hsv_to_rgb_visualization(h, np.full_like(s, 255), v)
    save_rgb(hue_vis, output_dir / "hsv_h_color.png")
    save_rgb(sat_vis, output_dir / "hsv_s_color.png")
    save_rgb(val_vis, output_dir / "hsv_v_color.png")


def normalize_signed_channel(channel: np.ndarray) -> np.ndarray:
    return np.clip((channel + 1.0) * 127.5, 0, 255).astype(np.uint8)


def decompose_hvi(rgb_array: np.ndarray, output_dir: Path, fixed_k: float) -> None:
    hue, saturation, intensity = rgb_to_hsv_components(rgb_array)
    eps = 1e-8
    k_value = float(np.clip(fixed_k, 0.05, 0.80))

    base = np.sin(intensity * 0.5 * math.pi) + eps
    color_sensitive = np.power(base, k_value)
    h_component = color_sensitive * saturation * np.cos(2.0 * math.pi * hue)
    v_component = color_sensitive * saturation * np.sin(2.0 * math.pi * hue)

    h_channel = normalize_signed_channel(h_component)
    v_channel = normalize_signed_channel(v_component)
    i_channel = np.clip(intensity * 255.0, 0, 255).astype(np.uint8)

    save_grayscale(h_channel, output_dir / "hvi_h_gray.png")
    save_grayscale(v_channel, output_dir / "hvi_v_gray.png")
    save_grayscale(i_channel, output_dir / "hvi_i_gray.png")

    h_color = np.stack([h_channel, np.zeros_like(h_channel), 255 - h_channel], axis=-1)
    v_color = np.stack([v_channel, 255 - v_channel, np.zeros_like(v_channel)], axis=-1)
    i_color = np.stack([i_channel, i_channel, i_channel], axis=-1)
    composite = np.stack([h_channel, v_channel, i_channel], axis=-1)

    save_rgb(h_color, output_dir / "hvi_h_color.png")
    save_rgb(v_color, output_dir / "hvi_v_color.png")
    save_rgb(i_color, output_dir / "hvi_i_color.png")
    save_rgb(composite, output_dir / "hvi_composite.png")


def ensure_spaces(spaces: Iterable[str]) -> set[str]:
    values = set(spaces)
    if "all" in values:
        return {"rgb", "hsv", "hvi"}
    return values


def process_image(path: Path, output_root: Path, spaces: set[str], fixed_k: float, base_input: Path) -> None:
    image = load_image(path)
    rgb_array = np.array(image, dtype=np.uint8)

    relative_parent = path.parent.relative_to(base_input) if base_input.is_dir() else Path()
    image_output_dir = output_root / relative_parent / path.stem
    image_output_dir.mkdir(parents=True, exist_ok=True)

    if "rgb" in spaces:
        decompose_rgb(rgb_array, image_output_dir)
    if "hsv" in spaces:
        decompose_hsv(image, image_output_dir)
    if "hvi" in spaces:
        decompose_hvi(rgb_array, image_output_dir, fixed_k)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    spaces = ensure_spaces(args.spaces)
    images = collect_images(input_path)
    if not images:
        raise SystemExit(f"No supported images found under: {input_path}")

    for path in images:
        process_image(path, output_root, spaces, args.fixed_k, input_path)
        print(f"Processed: {path}")

    print(f"Finished. Results saved to: {output_root}")


if __name__ == "__main__":
    main()
