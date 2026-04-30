from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "yolo_training" / "runs" / "segment" / "droplets_yolo11n_seg" / "weights" / "best.pt"
DEFAULT_IMAGE_DIR = Path("")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
EXCLUDED_IMAGE_DIR_NAMES = {
    "crops",
    "overlays",
    "droplets_analysis",
    "channel_app_batch",
    "channel_app_batch_transverse_40um",
}


st.set_page_config(page_title="Droplets channel analysis", layout="wide")


@st.cache_resource
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    arr = array.astype(np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(finite, [0.5, 99.5])
    if hi <= lo:
        lo, hi = float(finite.min()), float(finite.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0) * 255.0
    return arr.astype(np.uint8)


def load_image_from_path(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image)
    return to_rgb8(array)


def load_image_from_upload(uploaded_file) -> np.ndarray:
    uploaded_file.seek(0)
    with Image.open(uploaded_file) as image:
        array = np.asarray(image)
    return to_rgb8(array)


def to_rgb8(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        if array.dtype != np.uint8:
            array = normalize_to_uint8(array)
        return np.repeat(array[..., None], 3, axis=2)
    if array.ndim == 3 and array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        array = normalize_to_uint8(array)
    if array.ndim == 3 and array.shape[-1] == 3:
        return array
    raise ValueError(f"Unsupported image shape: {array.shape}")


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTS
        and "contact_sheet" not in path.stem.lower()
        and not any(part in EXCLUDED_IMAGE_DIR_NAMES for part in path.relative_to(folder).parts[:-1])
    )


def list_uploaded_images(uploaded_files) -> list:
    return sorted(
        (
            file
            for file in uploaded_files
            if Path(file.name).suffix.lower() in IMAGE_EXTS
            and "contact_sheet" not in Path(file.name).stem.lower()
        ),
        key=lambda file: file.name,
    )


def uploaded_folder_name(uploaded_files: list) -> str:
    if not uploaded_files:
        return "uploaded_folder"
    first = Path(uploaded_files[0].name)
    return first.parts[0] if len(first.parts) > 1 else first.stem


def safe_path_name(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_." else "_" for char in value.strip())
    return cleaned.strip("._") or "uploaded_folder"


def source_label(source, input_folder: Path | None = None) -> str:
    if isinstance(source, Path):
        if input_folder is not None:
            try:
                return str(source.relative_to(input_folder))
            except ValueError:
                pass
        return source.name
    return source.name


def source_stem(source) -> str:
    return Path(source.name if not isinstance(source, Path) else source.name).stem


def load_image_source(source) -> np.ndarray:
    if isinstance(source, Path):
        return load_image_from_path(source)
    return load_image_from_upload(source)


def has_explicit_folder(folder: Path) -> bool:
    text = str(folder).strip()
    return bool(text) and text != "."


def image_luminance(image: np.ndarray) -> np.ndarray:
    return (0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]).astype(np.float32)


def compute_contrast_limits(
    sources: list,
    low_percentile: float,
    high_percentile: float,
    max_images: int = 20,
    max_pixels_per_image: int = 50_000,
) -> tuple[float, float]:
    if not sources:
        return 0.0, 255.0
    sample_indexes = np.linspace(0, len(sources) - 1, min(len(sources), max_images), dtype=int)
    samples: list[np.ndarray] = []
    for index in sample_indexes:
        image = load_image_source(sources[int(index)])
        lum = image_luminance(image).ravel()
        if lum.size > max_pixels_per_image:
            step = max(1, lum.size // max_pixels_per_image)
            lum = lum[::step]
        samples.append(lum)
    values = np.concatenate(samples) if samples else np.array([0.0, 255.0], dtype=np.float32)
    lo, hi = np.percentile(values, [low_percentile, high_percentile])
    if hi <= lo:
        return 0.0, 255.0
    return float(lo), float(hi)


def apply_contrast_calibration(image: np.ndarray, limits: tuple[float, float] | None) -> np.ndarray:
    if limits is None:
        return image
    lo, hi = limits
    if hi <= lo:
        return image
    calibrated = (image.astype(np.float32) - lo) / (hi - lo)
    calibrated = np.clip(calibrated, 0.0, 1.0) * 255.0
    return calibrated.astype(np.uint8)


def rotate_image_keep_size(image: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 1e-6:
        return image
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    border = tuple(float(v) for v in np.median(image.reshape(-1, image.shape[-1]), axis=0))
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


def load_analysis_image(source, contrast_limits: tuple[float, float] | None, deskew_angle_deg: float = 0.0) -> np.ndarray:
    image = apply_contrast_calibration(load_image_source(source), contrast_limits)
    return rotate_image_keep_size(image, deskew_angle_deg)


def auto_crop_channel(image: np.ndarray) -> tuple[int, int, int, int]:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    col_score = blur.std(axis=0) + np.abs(np.gradient(blur.mean(axis=0)))
    threshold = np.percentile(col_score, 55)
    active_cols = np.where(col_score >= threshold)[0]
    if active_cols.size:
        x1 = max(0, int(active_cols.min()) - max(4, width // 20))
        x2 = min(width, int(active_cols.max()) + max(4, width // 20))
    else:
        x1, x2 = 0, width

    row_score = blur.std(axis=1)
    smooth = cv2.GaussianBlur(row_score.astype(np.float32), (1, 31), 0).ravel()
    if height > width * 2:
        search_start = int(height * 0.55)
        tail = smooth[search_start:]
        if tail.size and tail.max() > smooth.mean() + smooth.std():
            y2 = search_start + int(np.argmax(tail))
            y2 = max(int(height * 0.55), min(y2, int(height * 0.92)))
        else:
            y2 = int(height * 0.86)
        y1 = 0
    else:
        y1, y2 = 0, height

    if y2 - y1 < max(40, height // 4):
        y1, y2 = 0, int(height * 0.85)
    return x1, y1, x2, y2


def draw_crop_preview(image: np.ndarray, crop: tuple[int, int, int, int]) -> Image.Image:
    preview = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(preview)
    x1, y1, x2, y2 = crop
    draw.rectangle([x1, y1, x2 - 1, y2 - 1], outline=(255, 0, 0), width=3)
    return preview


def fit_display_size(width: int, height: int, max_width: int = 900, max_height: int = 650) -> tuple[int, int, float]:
    scale = min(max_width / width, max_height / height, 1.0)
    return max(1, int(width * scale)), max(1, int(height * scale)), scale


def rect_object(left: float, top: float, width: float, height: float, stroke: str, fill: str) -> dict:
    return {
        "type": "rect",
        "left": left,
        "top": top,
        "width": width,
        "height": height,
        "fill": fill,
        "stroke": stroke,
        "strokeWidth": 3,
        "selectable": True,
    }


def canvas_initial(objects: list[dict]) -> dict:
    return {"version": "4.4.0", "objects": objects}


def parse_canvas_rect(json_data: dict | None, fallback: tuple[int, int, int, int], scale: float) -> tuple[int, int, int, int]:
    if not json_data:
        return fallback
    rects = [obj for obj in json_data.get("objects", []) if obj.get("type") == "rect"]
    if not rects:
        return fallback
    obj = rects[0]
    left = float(obj.get("left", 0.0))
    top = float(obj.get("top", 0.0))
    width = float(obj.get("width", 0.0)) * float(obj.get("scaleX", 1.0))
    height = float(obj.get("height", 0.0)) * float(obj.get("scaleY", 1.0))
    x1 = int(round(left / scale))
    y1 = int(round(top / scale))
    x2 = int(round((left + width) / scale))
    y2 = int(round((top + height) / scale))
    return x1, y1, x2, y2


def parse_bar_rects(json_data: dict | None, fallback: tuple[int, int], scale: float) -> tuple[int, int]:
    if not json_data:
        return fallback
    rects = [obj for obj in json_data.get("objects", []) if obj.get("type") == "rect"]
    if len(rects) < 2:
        return fallback
    centers = []
    for obj in rects[:2]:
        left = float(obj.get("left", 0.0))
        width = float(obj.get("width", 0.0)) * float(obj.get("scaleX", 1.0))
        centers.append(int(round((left + width / 2.0) / scale)))
    centers = sorted(centers)
    return centers[0], centers[1]


def auto_channel_bars(crop: np.ndarray) -> tuple[int, int]:
    height, width = crop.shape[:2]
    if width <= 4:
        return 0, max(0, width - 1)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    col_mean = blur.mean(axis=0).astype(np.float32)
    col_std = blur.std(axis=0).astype(np.float32)
    edge_score = np.abs(np.gradient(col_mean)) + 0.35 * np.abs(np.gradient(col_std))
    edge_score = cv2.GaussianBlur(edge_score.reshape(1, -1), (31, 1), 0).ravel()

    min_gap = max(12, int(width * 0.18))
    candidates = np.argsort(edge_score)[::-1]
    best_left, best_right = int(width * 0.08), int(width * 0.92)
    for left in candidates[: max(20, width // 3)]:
        for right in candidates[: max(20, width // 3)]:
            if right <= left + min_gap:
                continue
            if left > width * 0.45 or right < width * 0.55:
                continue
            best_left, best_right = int(left), int(right)
            return best_left, best_right
    return best_left, best_right


def detect_channel_geometry(image: np.ndarray) -> dict[str, int | float]:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    y_profile_start = int(height * 0.15)
    y_profile_end = int(height * 0.82)
    profile_roi = blur[y_profile_start:y_profile_end]

    sobel_x = np.abs(cv2.Sobel(profile_roi, cv2.CV_32F, 1, 0, ksize=3)).mean(axis=0)
    dark_score = (255.0 - profile_roi.astype(np.float32)).mean(axis=0)
    score = sobel_x + 0.15 * dark_score
    score = cv2.GaussianBlur(score.reshape(1, -1), (51, 1), 0).ravel()

    center = width // 2
    search_left = int(width * 0.30)
    search_right = int(width * 0.70)
    candidates = np.arange(search_left, search_right)
    top_candidates = candidates[np.argsort(score[candidates])[::-1][: max(40, width // 12)]]

    best: tuple[float, int, int] | None = None
    min_gap = max(18, int(width * 0.010))
    max_gap = max(90, int(width * 0.075))
    for left in top_candidates:
        for right in top_candidates:
            if right <= left:
                continue
            gap = int(right - left)
            if gap < min_gap or gap > max_gap:
                continue
            pair_center = (left + right) / 2.0
            center_penalty = 0.05 * abs(pair_center - center)
            pair_score = float(score[left] + score[right] - center_penalty)
            if best is None or pair_score > best[0]:
                best = (pair_score, int(left), int(right))

    if best is None:
        bar_left = int(width * 0.48)
        bar_right = int(width * 0.52)
        confidence = 0.0
    else:
        confidence = best[0]
        bar_left, bar_right = best[1], best[2]

    channel_width = max(1, bar_right - bar_left)
    wide_margin = max(60, int(channel_width * 2.0))
    x1 = max(0, bar_left - wide_margin)
    x2 = min(width, bar_right + wide_margin)

    wide = gray[:, max(0, bar_left - wide_margin): min(width, bar_right + wide_margin)]
    threshold_dark = np.percentile(gray, 30)
    dark_mask = wide < threshold_dark
    row_dark_count = dark_mask.sum(axis=1).astype(np.float32)
    row_dark_count = cv2.GaussianBlur(row_dark_count.reshape(-1, 1), (1, 61), 0).ravel()

    reference_slice = row_dark_count[int(height * 0.25): int(height * 0.65)]
    reference = float(np.median(reference_slice)) if reference_slice.size else float(np.median(row_dark_count))
    high_row = max(reference * 2.2, float(np.percentile(row_dark_count, 90)))
    y2 = int(height * 0.90)
    for row in range(int(height * 0.55), int(height * 0.96)):
        if row_dark_count[row] > high_row:
            y2 = max(int(height * 0.50), row - max(4, int(channel_width * 0.25)))
            break

    y1 = 0
    crop_height = y2 - y1
    if crop_height < int(height * 0.35):
        y2 = int(height * 0.82)

    return {
        "crop_x1": int(x1),
        "crop_y1": int(y1),
        "crop_x2": int(x2),
        "crop_y2": int(y2),
        "bar_left": int(bar_left - x1),
        "bar_right": int(bar_right - x1),
        "global_bar_left": int(bar_left),
        "global_bar_right": int(bar_right),
        "channel_width_px": int(channel_width),
        "detection_score": float(confidence),
    }


def build_manual_geometry(
    crop_x1: int,
    crop_y1: int,
    crop_x2: int,
    crop_y2: int,
    global_bar_left: int,
    global_bar_right: int,
    detection_score: float = 0.0,
) -> dict[str, int | float]:
    bar_left, bar_right = sorted((int(global_bar_left), int(global_bar_right)))
    crop_x1 = int(crop_x1)
    crop_y1 = int(crop_y1)
    crop_x2 = int(crop_x2)
    crop_y2 = int(crop_y2)
    return {
        "crop_x1": crop_x1,
        "crop_y1": crop_y1,
        "crop_x2": crop_x2,
        "crop_y2": crop_y2,
        "bar_left": int(bar_left - crop_x1),
        "bar_right": int(bar_right - crop_x1),
        "global_bar_left": int(bar_left),
        "global_bar_right": int(bar_right),
        "channel_width_px": int(max(1, bar_right - bar_left)),
        "detection_score": float(detection_score),
    }


def clamp_session_value(key: str, lower: int, upper: int) -> None:
    if key in st.session_state:
        st.session_state[key] = int(np.clip(st.session_state[key], lower, upper))


def integer_slider(container, label: str, lower: int, upper: int, default: int, key: str) -> int:
    if key in st.session_state:
        clamp_session_value(key, lower, upper)
        return int(container.slider(label, lower, upper, step=1, key=key))
    return int(container.slider(label, lower, upper, int(np.clip(default, lower, upper)), 1, key=key))


def geometry_from_folder_state(
    auto_geometry: dict[str, int | float],
    image_shape: tuple[int, int, int],
    key_prefix: str,
) -> dict[str, int | float]:
    height, width = image_shape[:2]
    crop_x1 = int(st.session_state.get(f"{key_prefix}_crop_x1", int(auto_geometry["crop_x1"])))
    crop_x1 = int(np.clip(crop_x1, 0, max(0, width - 2)))
    crop_x2 = int(st.session_state.get(f"{key_prefix}_crop_x2", int(auto_geometry["crop_x2"])))
    crop_x2 = int(np.clip(crop_x2, crop_x1 + 1, width))
    crop_y1 = int(st.session_state.get(f"{key_prefix}_crop_y1", int(auto_geometry["crop_y1"])))
    crop_y1 = int(np.clip(crop_y1, 0, max(0, height - 2)))
    crop_y2 = int(st.session_state.get(f"{key_prefix}_crop_y2", int(auto_geometry["crop_y2"])))
    crop_y2 = int(np.clip(crop_y2, crop_y1 + 1, height))
    bar_left = int(st.session_state.get(f"{key_prefix}_bar_left", int(auto_geometry["global_bar_left"])))
    bar_left = int(np.clip(bar_left, crop_x1, max(crop_x1, crop_x2 - 1)))
    bar_right = int(st.session_state.get(f"{key_prefix}_bar_right", int(auto_geometry["global_bar_right"])))
    bar_right = int(np.clip(bar_right, bar_left + 1, crop_x2))
    return build_manual_geometry(
        crop_x1=crop_x1,
        crop_y1=crop_y1,
        crop_x2=crop_x2,
        crop_y2=crop_y2,
        global_bar_left=bar_left,
        global_bar_right=bar_right,
        detection_score=float(auto_geometry["detection_score"]),
    )


def geometry_adjustment_controls(
    auto_geometry: dict[str, int | float],
    image_shape: tuple[int, int, int],
    key_prefix: str,
) -> dict[str, int | float]:
    height, width = image_shape[:2]
    auto_crop_x1 = int(auto_geometry["crop_x1"])
    auto_crop_y1 = int(auto_geometry["crop_y1"])
    auto_crop_x2 = int(auto_geometry["crop_x2"])
    auto_crop_y2 = int(auto_geometry["crop_y2"])
    auto_bar_left = int(auto_geometry["global_bar_left"])
    auto_bar_right = int(auto_geometry["global_bar_right"])

    st.subheader("Manual geometry adjustment")
    st.caption("Automatic detection initializes these controls once per folder. Edits are kept when you switch images and can be reused for batch export.")
    reset = st.button("Reset to automatic detection", key=f"{key_prefix}_reset_geometry")
    if reset:
        for suffix in ("crop_x1", "crop_x2", "crop_y1", "crop_y2", "bar_left", "bar_right"):
            st.session_state.pop(f"{key_prefix}_{suffix}", None)

    crop_cols = st.columns(4)
    key = f"{key_prefix}_crop_x1"
    crop_x1 = integer_slider(crop_cols[0], "Crop left x", 0, max(0, width - 2), auto_crop_x1, key)

    key = f"{key_prefix}_crop_x2"
    crop_x2 = integer_slider(crop_cols[1], "Crop right x", crop_x1 + 1, width, auto_crop_x2, key)

    key = f"{key_prefix}_crop_y1"
    crop_y1 = integer_slider(crop_cols[2], "Crop top y", 0, max(0, height - 2), auto_crop_y1, key)

    key = f"{key_prefix}_crop_y2"
    crop_y2 = integer_slider(crop_cols[3], "Crop bottom y", crop_y1 + 1, height, auto_crop_y2, key)

    wall_cols = st.columns(2)
    key = f"{key_prefix}_bar_left"
    bar_left = integer_slider(wall_cols[0], "Left channel wall x", crop_x1, crop_x2 - 1, auto_bar_left, key)

    key = f"{key_prefix}_bar_right"
    bar_right = integer_slider(wall_cols[1], "Right channel wall x", bar_left + 1, crop_x2, auto_bar_right, key)

    geometry = build_manual_geometry(
        crop_x1=crop_x1,
        crop_y1=crop_y1,
        crop_x2=crop_x2,
        crop_y2=crop_y2,
        global_bar_left=bar_left,
        global_bar_right=bar_right,
        detection_score=float(auto_geometry["detection_score"]),
    )
    st.session_state[f"{key_prefix}_last_geometry"] = geometry
    return geometry


def draw_geometry_preview(image: np.ndarray, geometry: dict[str, int | float]) -> np.ndarray:
    out = image.copy()
    x1 = int(geometry["crop_x1"])
    y1 = int(geometry["crop_y1"])
    x2 = int(geometry["crop_x2"])
    y2 = int(geometry["crop_y2"])
    gl = int(geometry["global_bar_left"])
    gr = int(geometry["global_bar_right"])
    cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), (255, 0, 0), 3)
    cv2.line(out, (gl, y1), (gl, y2 - 1), (255, 255, 255), 3)
    cv2.line(out, (gr, y1), (gr, y2 - 1), (255, 255, 255), 3)
    cv2.line(out, (gl, y1), (gl, y2 - 1), (0, 0, 0), 1)
    cv2.line(out, (gr, y1), (gr, y2 - 1), (0, 0, 0), 1)
    return out


def estimate_wall_line(crop: np.ndarray, expected_x: int, half_window: int = 24) -> dict[str, float | int | bool]:
    height, width = crop.shape[:2]
    if height < 20 or width < 4:
        return {"detected": False, "angle_deg": 0.0, "drift_px": 0.0, "points": 0}
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    grad = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    rows = np.arange(0, height, max(1, height // 400))
    xs: list[int] = []
    ys: list[int] = []
    scores: list[float] = []
    for row in rows:
        left = max(0, expected_x - half_window)
        right = min(width, expected_x + half_window + 1)
        if right <= left:
            continue
        window = grad[row, left:right]
        local = int(np.argmax(window))
        xs.append(left + local)
        ys.append(int(row))
        scores.append(float(window[local]))
    if len(xs) < 20:
        return {"detected": False, "angle_deg": 0.0, "drift_px": 0.0, "points": len(xs)}
    score_array = np.asarray(scores, dtype=np.float32)
    keep = score_array >= np.percentile(score_array, 55)
    if int(keep.sum()) < 20:
        keep = score_array > 0
    x_array = np.asarray(xs, dtype=np.float32)[keep]
    y_array = np.asarray(ys, dtype=np.float32)[keep]
    if x_array.size < 20:
        return {"detected": False, "angle_deg": 0.0, "drift_px": 0.0, "points": int(x_array.size)}
    slope, intercept = np.polyfit(y_array, x_array, 1)
    drift = float(slope * max(1, height - 1))
    angle = float(math.degrees(math.atan(slope)))
    x_top = float(intercept)
    x_bottom = float(slope * (height - 1) + intercept)
    residual = float(np.median(np.abs(x_array - (slope * y_array + intercept))))
    return {
        "detected": True,
        "angle_deg": angle,
        "drift_px": drift,
        "points": int(x_array.size),
        "x_top": x_top,
        "x_bottom": x_bottom,
        "residual_px": residual,
    }


def check_wall_alignment(image: np.ndarray, geometry: dict[str, int | float]) -> dict[str, float | int | bool | str]:
    x1 = int(geometry["crop_x1"])
    y1 = int(geometry["crop_y1"])
    x2 = int(geometry["crop_x2"])
    y2 = int(geometry["crop_y2"])
    crop = image[y1:y2, x1:x2]
    left_x = int(geometry["bar_left"])
    right_x = int(geometry["bar_right"])
    left = estimate_wall_line(crop, left_x)
    right = estimate_wall_line(crop, right_x)
    detected = bool(left["detected"] and right["detected"])
    left_angle = float(left.get("angle_deg", 0.0))
    right_angle = float(right.get("angle_deg", 0.0))
    mean_angle = float((left_angle + right_angle) / 2.0)
    angle_delta = float(abs(left_angle - right_angle))
    left_drift = float(left.get("drift_px", 0.0))
    right_drift = float(right.get("drift_px", 0.0))
    max_drift = float(max(abs(left_drift), abs(right_drift)))
    status = "not_detected"
    if detected:
        status = "tilted" if abs(mean_angle) > 0.4 or max_drift > 8 or angle_delta > 0.6 else "aligned"
    return {
        "alignment_status": status,
        "walls_detected": detected,
        "left_wall_angle_deg": left_angle,
        "right_wall_angle_deg": right_angle,
        "mean_wall_angle_deg": mean_angle,
        "wall_angle_delta_deg": angle_delta,
        "left_wall_drift_px": left_drift,
        "right_wall_drift_px": right_drift,
        "max_wall_drift_px": max_drift,
        "left_wall_residual_px": float(left.get("residual_px", 0.0)),
        "right_wall_residual_px": float(right.get("residual_px", 0.0)),
        "left_wall_points": int(left.get("points", 0)),
        "right_wall_points": int(right.get("points", 0)),
    }


def draw_alignment_preview(image: np.ndarray, geometry: dict[str, int | float], alignment: dict[str, float | int | bool | str]) -> np.ndarray:
    out = draw_geometry_preview(image, geometry)
    x1 = int(geometry["crop_x1"])
    y1 = int(geometry["crop_y1"])
    y2 = int(geometry["crop_y2"])
    height = max(1, y2 - y1)
    for side in ("left", "right"):
        drift = float(alignment.get(f"{side}_wall_drift_px", 0.0))
        base = int(geometry["global_bar_left"] if side == "left" else geometry["global_bar_right"])
        top_x = int(round(base - drift / 2.0))
        bottom_x = int(round(base + drift / 2.0))
        cv2.line(out, (top_x, y1), (bottom_x, y1 + height - 1), (255, 215, 0), 2)
    return out


def overlay_masks(
    image: np.ndarray,
    masks: list[np.ndarray],
    bars: tuple[int, int],
    alpha: float = 0.45,
) -> np.ndarray:
    out = image.copy()
    colors = [
        (255, 99, 71),
        (0, 180, 216),
        (46, 204, 113),
        (241, 196, 15),
        (155, 89, 182),
        (255, 127, 80),
    ]
    for index, mask in enumerate(masks):
        color = np.array(colors[index % len(colors)], dtype=np.uint8)
        colored = np.zeros_like(out)
        colored[:] = color
        out = np.where(mask[..., None], (out * (1 - alpha) + colored * alpha).astype(np.uint8), out)
    left, right = bars
    cv2.line(out, (left, 0), (left, out.shape[0] - 1), (255, 255, 255), 2)
    cv2.line(out, (right, 0), (right, out.shape[0] - 1), (255, 255, 255), 2)
    cv2.line(out, (left, 0), (left, out.shape[0] - 1), (0, 0, 0), 1)
    cv2.line(out, (right, 0), (right, out.shape[0] - 1), (0, 0, 0), 1)
    return out


def segment_and_measure(
    model: YOLO,
    crop_rgb: np.ndarray,
    conf: float,
    bar_left: int,
    bar_right: int,
    channel_distance: float,
    min_area_px: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    result = model.predict(crop_rgb, imgsz=1024, conf=conf, retina_masks=True, verbose=False)[0]
    if result.masks is None:
        return overlay_masks(crop_rgb, [], (bar_left, bar_right)), pd.DataFrame()

    channel_width_px = max(1, abs(bar_right - bar_left))
    pixel_size = channel_distance / channel_width_px
    masks_tensor = result.masks.data.detach().cpu().numpy()
    confidences = result.boxes.conf.detach().cpu().numpy() if result.boxes is not None else np.ones(len(masks_tensor))
    kept_masks: list[np.ndarray] = []
    records: list[dict[str, float]] = []

    for idx, raw_mask in enumerate(masks_tensor, start=1):
        mask = raw_mask > 0.5
        if mask.shape != crop_rgb.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (crop_rgb.shape[1], crop_rgb.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        channel_left = min(bar_left, bar_right)
        channel_right = max(bar_left, bar_right)
        channel_mask = np.zeros_like(mask, dtype=bool)
        channel_mask[:, channel_left : channel_right + 1] = True
        clipped_mask = mask & channel_mask

        area_px = int(mask.sum())
        area_channel_px = int(clipped_mask.sum())
        if area_channel_px < min_area_px:
            continue
        ys, xs = np.where(clipped_mask)
        if xs.size == 0:
            continue
        cx = float(xs.mean())
        cy = float(ys.mean())
        row_widths = []
        for row in np.unique(ys):
            row_xs = xs[ys == row]
            row_widths.append(int(row_xs.max() - row_xs.min() + 1))
        row_widths_array = np.asarray(row_widths, dtype=np.float32)
        transverse_width_px = float(np.percentile(row_widths_array, 90))
        transverse_width_px = min(transverse_width_px, float(channel_width_px))
        longitudinal_height_px = float(ys.max() - ys.min() + 1)
        area_equivalent_px = 2.0 * math.sqrt(area_channel_px / math.pi)
        area_equivalent_px_capped = min(area_equivalent_px, float(channel_width_px))
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        kept_masks.append(clipped_mask)
        records.append(
            {
                "id": len(records) + 1,
                "confidence": float(confidences[idx - 1]),
                "center_x_px": cx,
                "center_y_px": cy,
                "area_px": area_px,
                "area_channel_px": area_channel_px,
                "diameter_px": transverse_width_px,
                "diameter_unit": transverse_width_px * pixel_size,
                "diameter_transverse_px": transverse_width_px,
                "diameter_transverse_unit": transverse_width_px * pixel_size,
                "diameter_area_eq_px": area_equivalent_px,
                "diameter_area_eq_unit": area_equivalent_px * pixel_size,
                "diameter_area_eq_capped_px": area_equivalent_px_capped,
                "diameter_area_eq_capped_unit": area_equivalent_px_capped * pixel_size,
                "longitudinal_length_px": longitudinal_height_px,
                "aspect_ratio_longitudinal_over_transverse": longitudinal_height_px / max(1e-6, transverse_width_px),
                "bbox_width_px": x_max - x_min + 1,
                "bbox_height_px": y_max - y_min + 1,
            }
        )

    table = pd.DataFrame.from_records(records)
    return overlay_masks(crop_rgb, kept_masks, (bar_left, bar_right)), table


def analyze_image(
    model: YOLO,
    image: np.ndarray,
    conf: float,
    channel_distance: float,
    min_area_px: int,
    geometry: dict[str, int | float] | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, int | float]]:
    if geometry is None:
        geometry = detect_channel_geometry(image)
    x1 = int(geometry["crop_x1"])
    y1 = int(geometry["crop_y1"])
    x2 = int(geometry["crop_x2"])
    y2 = int(geometry["crop_y2"])
    crop = image[y1:y2, x1:x2]
    overlay, table = segment_and_measure(
        model=model,
        crop_rgb=crop,
        conf=conf,
        bar_left=int(geometry["bar_left"]),
        bar_right=int(geometry["bar_right"]),
        channel_distance=channel_distance,
        min_area_px=min_area_px,
    )
    if not table.empty:
        table.insert(1, "crop_x1", x1)
        table.insert(2, "crop_y1", y1)
        table.insert(3, "bar_left_px", int(geometry["bar_left"]))
        table.insert(4, "bar_right_px", int(geometry["bar_right"]))
    return draw_geometry_preview(image, geometry), overlay, table, geometry


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def export_folder_data(
    *,
    sources: list,
    model: YOLO,
    input_folder: Path,
    using_uploaded_folder: bool,
    output_dir: Path,
    use_verified_geometry: bool,
    save_overlays: bool,
    geometry_key_prefix: str,
    contrast_limits: tuple[float, float] | None,
    contrast_low: float,
    contrast_high: float,
    deskew_angle_deg: float,
    conf: float,
    channel_distance: float,
    unit: str,
    min_area_px: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path, Path, Path]:
    measurements: list[pd.DataFrame] = []
    summary: list[dict[str, object]] = []
    progress = st.progress(0)
    status = st.empty()
    overlay_root = output_dir / "overlays"
    crop_root = output_dir / "crops"

    for index, source in enumerate(sources, start=1):
        rel = Path(source_label(source, None if using_uploaded_folder else input_folder))
        status.write(f"Processing {index}/{len(sources)}: {rel.name}")
        try:
            image = load_analysis_image(source, contrast_limits, deskew_angle_deg)
            geometry = None
            if use_verified_geometry:
                auto_geometry = detect_channel_geometry(image)
                geometry = geometry_from_folder_state(
                    auto_geometry=auto_geometry,
                    image_shape=image.shape,
                    key_prefix=geometry_key_prefix,
                )
            geometry_preview, segmentation_overlay, table, geometry = analyze_image(
                model=model,
                image=image,
                conf=conf,
                channel_distance=channel_distance,
                min_area_px=int(min_area_px),
                geometry=geometry,
            )
            alignment = check_wall_alignment(image, geometry)
            if not table.empty:
                table = table.copy()
                table.insert(0, "image", str(rel))
                for key, value in alignment.items():
                    table[key] = value
                measurements.append(table)
            summary.append(
                {
                    "image": str(rel),
                    "droplet_count": len(table),
                    "channel_width_px": int(geometry["channel_width_px"]),
                    "crop_x1": int(geometry["crop_x1"]),
                    "crop_y1": int(geometry["crop_y1"]),
                    "crop_x2": int(geometry["crop_x2"]),
                    "crop_y2": int(geometry["crop_y2"]),
                    "global_bar_left": int(geometry["global_bar_left"]),
                    "global_bar_right": int(geometry["global_bar_right"]),
                    "geometry_mode": "verified_single_image" if use_verified_geometry else "automatic_per_image",
                    "confidence_threshold": float(conf),
                    "min_area_px": int(min_area_px),
                    "channel_distance": float(channel_distance),
                    "unit": unit,
                    "contrast_calibration": "folder_auto" if contrast_limits is not None else "off",
                    "contrast_low_value": None if contrast_limits is None else float(contrast_limits[0]),
                    "contrast_high_value": None if contrast_limits is None else float(contrast_limits[1]),
                    "deskew_angle_deg": float(deskew_angle_deg),
                    "mean_diameter": None if table.empty else float(table["diameter_unit"].mean()),
                    "mean_area_equivalent_diameter": None if table.empty else float(table["diameter_area_eq_unit"].mean()),
                    **alignment,
                }
            )
            if save_overlays:
                save_rgb(overlay_root / rel.with_suffix(".png"), segmentation_overlay)
                save_rgb(crop_root / rel.with_suffix(".png"), draw_alignment_preview(image, geometry, alignment))
        except Exception as exc:
            summary.append({"image": str(rel), "error": str(exc), "droplet_count": 0})
        progress.progress(index / len(sources))

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary)
    measurements_df = pd.concat(measurements, ignore_index=True) if measurements else pd.DataFrame()
    settings_df = pd.DataFrame(
        [
            {
                "image_folder": "uploaded_folder" if using_uploaded_folder else str(input_folder),
                "working_folder_name": uploaded_folder_name(sources) if using_uploaded_folder else input_folder.name,
                "output_dir": str(output_dir),
                "image_source": "browser_folder_upload" if using_uploaded_folder else "local_path",
                "geometry_mode": "verified_single_image" if use_verified_geometry else "automatic_per_image",
                "crop_x1": int(st.session_state.get(f"{geometry_key_prefix}_crop_x1", -1)),
                "crop_y1": int(st.session_state.get(f"{geometry_key_prefix}_crop_y1", -1)),
                "crop_x2": int(st.session_state.get(f"{geometry_key_prefix}_crop_x2", -1)),
                "crop_y2": int(st.session_state.get(f"{geometry_key_prefix}_crop_y2", -1)),
                "global_bar_left": int(st.session_state.get(f"{geometry_key_prefix}_bar_left", -1)),
                "global_bar_right": int(st.session_state.get(f"{geometry_key_prefix}_bar_right", -1)),
                "channel_distance": float(channel_distance),
                "unit": unit,
                "confidence_threshold": float(conf),
                "min_area_px": int(min_area_px),
                "contrast_calibration": "folder_auto" if contrast_limits is not None else "off",
                "contrast_low_percentile": float(contrast_low),
                "contrast_high_percentile": float(contrast_high),
                "contrast_low_value": None if contrast_limits is None else float(contrast_limits[0]),
                "contrast_high_value": None if contrast_limits is None else float(contrast_limits[1]),
                "deskew_angle_deg": float(deskew_angle_deg),
            }
        ]
    )
    summary_path = output_dir / "batch_summary.csv"
    measurements_path = output_dir / "droplet_measurements.csv"
    settings_path = output_dir / "analysis_settings.csv"
    summary_df.to_csv(summary_path, index=False)
    measurements_df.to_csv(measurements_path, index=False)
    settings_df.to_csv(settings_path, index=False)
    status.empty()
    return summary_df, measurements_df, settings_df, summary_path, measurements_path, settings_path


def show_export_results(
    summary_df: pd.DataFrame,
    measurements_df: pd.DataFrame,
    settings_df: pd.DataFrame,
    summary_path: Path,
    measurements_path: Path,
    settings_path: Path,
) -> None:
    st.success(f"Export complete: {len(summary_df)} images processed.")
    st.write(f"Summary: `{summary_path}`")
    st.write(f"Measurements: `{measurements_path}`")
    st.write(f"Settings: `{settings_path}`")
    st.metric("Total droplets", 0 if measurements_df.empty else len(measurements_df))
    st.dataframe(summary_df, width="stretch", hide_index=True)
    st.download_button("Download batch summary CSV", summary_df.to_csv(index=False).encode("utf-8"), "batch_summary.csv", "text/csv")
    st.download_button(
        "Download droplet measurements CSV",
        measurements_df.to_csv(index=False).encode("utf-8"),
        "droplet_measurements.csv",
        "text/csv",
    )
    st.download_button("Download analysis settings CSV", settings_df.to_csv(index=False).encode("utf-8"), "analysis_settings.csv", "text/csv")


st.title("Droplets channel analysis")

if "image_folder" not in st.session_state:
    st.session_state["image_folder"] = str(DEFAULT_IMAGE_DIR)
if "folder_uploader_key" not in st.session_state:
    st.session_state["folder_uploader_key"] = 0
if "ignore_uploaded_files" not in st.session_state:
    st.session_state["ignore_uploaded_files"] = False

with st.sidebar:
    st.header("Analysis settings")
    model_path = st.text_input("Model path", value=str(DEFAULT_MODEL))
    channel_distance = st.number_input("Channel thickness between detected walls", min_value=0.000001, value=100.0, step=10.0)
    unit = st.text_input("Distance unit", value="um")
    conf = st.slider("YOLO confidence", 0.01, 0.95, 0.25, 0.01)
    min_area_px = st.number_input("Minimum droplet mask area (px)", min_value=1, value=20, step=5)
    st.divider()
    uploaded_files = st.file_uploader(
        "Choisir un dossier d'images avec Finder",
        type=sorted(ext.lstrip(".") for ext in IMAGE_EXTS),
        accept_multiple_files="directory",
        key=f"folder_uploader_{st.session_state['folder_uploader_key']}",
        help="Utilise le selecteur natif du navigateur. Toutes les images choisies seront traitees comme une meme video.",
    )
    if uploaded_files and st.session_state["ignore_uploaded_files"]:
        uploaded_files = []
    elif uploaded_files:
        st.session_state["ignore_uploaded_files"] = False
    if uploaded_files:
        st.caption(f"{len(uploaded_files)} fichier(s) charges via le navigateur. Le chemin macOS exact du dossier n'est pas transmis par Streamlit.")
        if st.button("Vider ce dossier et choisir un autre", width="stretch"):
            st.session_state["folder_uploader_key"] += 1
            st.session_state["ignore_uploaded_files"] = True
            for key in list(st.session_state.keys()):
                if key.startswith("folder_") and not key.startswith("folder_uploader_"):
                    st.session_state.pop(key, None)
            st.session_state["image_folder"] = ""
            st.rerun()
    input_folder_text = st.text_input(
        "Image folder",
        key="image_folder",
        help="Optionnel. Pour enregistrer les resultats directement dans le dossier source, colle ici le chemin local du dossier au lieu d'utiliser l'upload navigateur.",
    )
    input_folder = Path(input_folder_text).expanduser() if input_folder_text.strip() else Path("")
    st.divider()
    auto_contrast = st.checkbox("Auto-calibrer le contraste sur tout le dossier", value=True)
    contrast_low = st.number_input("Contrast low percentile", min_value=0.0, max_value=20.0, value=0.5, step=0.1)
    contrast_high = st.number_input("Contrast high percentile", min_value=80.0, max_value=100.0, value=99.5, step=0.1)
    st.divider()
    correct_tilt = st.checkbox("Redresser l'inclinaison camera", value=False)
    deskew_angle = st.number_input(
        "Angle de redressement (deg)",
        min_value=-5.0,
        max_value=5.0,
        step=0.05,
        format="%.3f",
        key="deskew_angle_deg",
    )
    st.caption("Angle commun applique a toutes les images du dossier avant detection.")
    st.divider()

uploaded_images = list_uploaded_images(uploaded_files or [])
using_uploaded_folder = len(uploaded_images) > 0
using_local_folder = (not using_uploaded_folder) and has_explicit_folder(input_folder)
all_images = uploaded_images if using_uploaded_folder else (list_images(input_folder) if using_local_folder else [])
current_source_name = uploaded_folder_name(uploaded_images) if using_uploaded_folder else (input_folder.name if using_local_folder else "no_folder_selected")
default_output_dir = (input_folder / "droplets_analysis") if using_local_folder else (ROOT / "runs" / "channel_app_batch" / safe_path_name(current_source_name))
with st.sidebar:
    st.divider()
    output_dir = Path(st.text_input("Output folder", value=str(default_output_dir)))
    if using_uploaded_folder:
        st.warning("Mode upload navigateur: macOS ne transmet pas le chemin reel du dossier. Pour sauvegarder dans le dossier source, colle son chemin dans `Image folder`.")
folder_identity = f"{current_source_name}|{len(uploaded_images)}|" + "|".join(file.name for file in uploaded_images[:20]) if using_uploaded_folder else (str(input_folder.resolve() if input_folder.exists() else input_folder) if using_local_folder else "no_folder_selected")
folder_key = folder_identity
folder_key = folder_key.replace("/", "_").replace("\\", "_").replace(" ", "_").replace(".", "_").replace(":", "_")
geometry_key_prefix = f"folder_{folder_key}"
active_deskew_angle = float(deskew_angle) if correct_tilt else 0.0
contrast_limits = None
if auto_contrast and all_images:
    with st.spinner("Calibrating folder contrast..."):
        contrast_limits = compute_contrast_limits(all_images, contrast_low, contrast_high)
    st.sidebar.caption(f"Contrast calibration: {contrast_limits[0]:.1f} -> {contrast_limits[1]:.1f}")

model_file = Path(model_path)
if not model_file.exists():
    st.error(f"Model not found: {model_file}")
    st.stop()
model = load_model(str(model_file))

tab_single, tab_batch = st.tabs(["Single image", "Batch"])

with tab_single:
    images = all_images
    if not images:
        st.warning("No image found. Choose an image folder with Finder or enter a valid image folder path.")
    else:
        selected = st.selectbox("Image", images, format_func=lambda source: source_label(source, None if using_uploaded_folder else input_folder))
        raw_image = load_analysis_image(selected, contrast_limits, 0.0)
        raw_auto_geometry = detect_channel_geometry(raw_image)
        raw_alignment = check_wall_alignment(raw_image, raw_auto_geometry)
        image = rotate_image_keep_size(raw_image, active_deskew_angle)
        auto_geometry = detect_channel_geometry(image)
        geometry = geometry_adjustment_controls(
            auto_geometry=auto_geometry,
            image_shape=image.shape,
            key_prefix=geometry_key_prefix,
        )
        geometry_preview, segmentation_overlay, table, geometry = analyze_image(
            model=model,
            image=image,
            conf=conf,
            channel_distance=channel_distance,
            min_area_px=int(min_area_px),
            geometry=geometry,
        )
        alignment = check_wall_alignment(image, geometry)
        alignment_preview = draw_alignment_preview(image, geometry, alignment)

        crop_x1 = int(geometry["crop_x1"])
        crop_y1 = int(geometry["crop_y1"])
        crop_x2 = int(geometry["crop_x2"])
        crop_y2 = int(geometry["crop_y2"])
        pixel_size = channel_distance / max(1, int(geometry["channel_width_px"]))

        metric_cols = st.columns(5)
        metric_cols[0].metric("Droplets", len(table))
        metric_cols[1].metric(f"Mean diameter ({unit})", "NA" if table.empty else f"{table['diameter_unit'].mean():.3f}")
        metric_cols[2].metric("Channel width (px)", int(geometry["channel_width_px"]))
        metric_cols[3].metric(f"Pixel size ({unit}/px)", f"{pixel_size:.5f}")
        metric_cols[4].metric("Crop height (px)", crop_y2 - crop_y1)

        align_cols = st.columns(4)
        align_cols[0].metric("Wall alignment", str(alignment["alignment_status"]))
        align_cols[1].metric("Mean tilt (deg)", f"{float(alignment['mean_wall_angle_deg']):.3f}")
        align_cols[2].metric("Max drift (px)", f"{float(alignment['max_wall_drift_px']):.1f}")
        align_cols[3].metric("Wall angle delta (deg)", f"{float(alignment['wall_angle_delta_deg']):.3f}")
        st.caption(
            f"Detected raw camera tilt on this image: {float(raw_alignment['mean_wall_angle_deg']):.3f} deg. "
            f"Applied deskew: {active_deskew_angle:.3f} deg."
        )
        if st.button("Utiliser l'angle detecte pour redresser tout le dossier"):
            st.session_state["deskew_angle_deg"] = float(raw_alignment["mean_wall_angle_deg"])
            st.rerun()
        if alignment["alignment_status"] == "tilted":
            st.warning("L'image semble inclinee globalement. Active `Redresser l'inclinaison camera`, puis utilise l'angle detecte si la preview devient droite.")
        elif alignment["alignment_status"] == "not_detected":
            st.info("Alignement des murs non confirme automatiquement sur cette image.")

        view_left, view_right = st.columns([1, 1])
        with view_left:
            st.image(alignment_preview, caption="Adjusted channel geometry: red crop, white walls, yellow fitted wall alignment")
        with view_right:
            st.image(segmentation_overlay, caption="YOLO segmentation inside the adjusted channel")

        st.caption(
            f"Adjusted crop x={crop_x1}:{crop_x2}, y={crop_y1}:{crop_y2}; "
            f"global channel walls x={int(geometry['global_bar_left'])}, {int(geometry['global_bar_right'])}. "
            f"Automatic values were crop x={int(auto_geometry['crop_x1'])}:{int(auto_geometry['crop_x2'])}, "
            f"y={int(auto_geometry['crop_y1'])}:{int(auto_geometry['crop_y2'])}; "
            f"walls x={int(auto_geometry['global_bar_left'])}, {int(auto_geometry['global_bar_right'])}. "
            f"Contrast calibration: {'off' if contrast_limits is None else f'{contrast_limits[0]:.1f}->{contrast_limits[1]:.1f}'}."
        )
        if table.empty:
            st.warning("No droplet detected with the current settings.")
        else:
            display = table.copy().rename(
                columns={
                    "diameter_unit": f"diameter_{unit}",
                    "diameter_transverse_unit": f"diameter_transverse_{unit}",
                    "diameter_area_eq_unit": f"diameter_area_eq_{unit}",
                    "diameter_area_eq_capped_unit": f"diameter_area_eq_capped_{unit}",
                }
            )
            st.dataframe(display, width="stretch", hide_index=True)
            st.download_button(
                "Download this image measurements CSV",
                data=dataframe_to_csv_bytes(display),
                file_name=f"{source_stem(selected)}_droplet_measurements.csv",
                mime="text/csv",
            )

        st.divider()
        save_single_overlays = st.checkbox("Save overlay images during export", value=True, key=f"{geometry_key_prefix}_single_save_overlays")
        if st.button("Exporter toutes les donnees du dossier avec ces parametres", type="primary"):
            summary_df, measurements_df, settings_df, summary_path, measurements_path, settings_path = export_folder_data(
                sources=all_images,
                model=model,
                input_folder=input_folder,
                using_uploaded_folder=using_uploaded_folder,
                output_dir=output_dir,
                use_verified_geometry=True,
                save_overlays=save_single_overlays,
                geometry_key_prefix=geometry_key_prefix,
                contrast_limits=contrast_limits,
                contrast_low=float(contrast_low),
                contrast_high=float(contrast_high),
                deskew_angle_deg=active_deskew_angle,
                conf=float(conf),
                channel_distance=float(channel_distance),
                unit=unit,
                min_area_px=int(min_area_px),
            )
            show_export_results(summary_df, measurements_df, settings_df, summary_path, measurements_path, settings_path)

with tab_batch:
    recursive = st.checkbox("Recursive folder scan", value=True)
    use_verified_geometry = st.checkbox("Use verified Single image geometry for every image", value=True)
    save_overlays = st.checkbox("Save overlay images", value=True)
    run_batch = st.button("Exporter toutes les donnees du dossier", type="primary")

    if run_batch:
        if using_uploaded_folder:
            batch_images = uploaded_images
        else:
            batch_images = all_images
            if using_local_folder and not recursive:
                batch_images = sorted(
                    path
                    for path in input_folder.iterdir()
                    if path.is_file()
                    and path.suffix.lower() in IMAGE_EXTS
                    and "contact_sheet" not in path.stem.lower()
                )
        if not batch_images:
            st.warning("No image found for batch analysis.")
        else:
            summary_df, measurements_df, settings_df, summary_path, measurements_path, settings_path = export_folder_data(
                sources=batch_images,
                model=model,
                input_folder=input_folder,
                using_uploaded_folder=using_uploaded_folder,
                output_dir=output_dir,
                use_verified_geometry=use_verified_geometry,
                save_overlays=save_overlays,
                geometry_key_prefix=geometry_key_prefix,
                contrast_limits=contrast_limits,
                contrast_low=float(contrast_low),
                contrast_high=float(contrast_high),
                deskew_angle_deg=active_deskew_angle,
                conf=float(conf),
                channel_distance=float(channel_distance),
                unit=unit,
                min_area_px=int(min_area_px),
            )
            show_export_results(summary_df, measurements_df, settings_df, summary_path, measurements_path, settings_path)
