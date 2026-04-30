#!/usr/bin/env python3
"""Compile YOLO droplet analyses into publication-style tables and figures.

Run the light UI with:

    streamlit run apps/publication_compiler/compile_droplet_publication_app.py

The app expects folders shaped like:

    extracted_frames_highres/<video_or_condition>/droplets_analysis/
        droplet_measurements.csv
        batch_summary.csv
        analysis_settings.csv
        overlays/*.png

It exports a publication folder compatible with the style/naming of:
    micro/droplet_analysis_manual/figures/figures_publication/en
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from string import ascii_uppercase
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageOps

try:
    from scipy import stats
except Exception:  # pragma: no cover - scipy is optional for the app to open.
    stats = None


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_OUTPUT = ROOT / "runs" / "publication_reference" / "en"
DEFAULT_ROOT = ROOT / "analysis_inputs"
ANALYSIS_PRESETS = {
    "repeatability": "Même expérience répétée: comparer les runs et quantifier l'écart.",
    "parameter_sweep": "Plusieurs conditions: effet des débits/fractions et cartes de régime.",
    "qc_only": "Contrôle qualité YOLO: calibration, frames, distributions et overlays.",
}
OKABE_ITO = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
]
STATUS_ORDER = ["usable", "to_confirm", "excluded"]
STATUS_LABELS = {
    "usable": "Usable",
    "to_confirm": "To confirm",
    "excluded": "Excluded",
}
STATUS_COLORS = {
    "usable": "#0072B2",
    "to_confirm": "#E69F00",
    "excluded": "#7A7A7A",
}
STATUS_FACE = {
    "usable": "#DCEBF7",
    "to_confirm": "#FCE7B5",
    "excluded": "#E4E4E4",
}
RUN_KEY_COLUMNS = ["experiment", "flow", "fraction", "video"]

CONDITION_STATS_COLUMNS = [
    "experiment",
    "Qc_uLh",
    "Qaq_uLh",
    "flow_ratio_Qc_over_Qaq",
    "EtOH_pct",
    "W_pct",
    "n_frames",
    "n_valid_frames",
    "valid_frame_fraction",
    "manual_bad_frame_fraction",
    "accumulation_frame_fraction",
    "n_droplets",
    "n_large_droplets",
    "large_droplet_fraction",
    "diameter_um_median",
    "median_ci_low",
    "median_ci_high",
    "diameter_um_p25",
    "diameter_um_p75",
    "diameter_um_mean",
    "diameter_um_std",
    "diameter_cv",
    "publication_status",
    "publication_note",
    "quality_status",
    "quality_flags",
]

DROPLET_COLUMNS = [
    "date",
    "experiment",
    "flow",
    "fraction",
    "Qc_uLh",
    "Qaq_uLh",
    "flow_ratio_Qc_over_Qaq",
    "EtOH_pct",
    "W_pct",
    "video",
    "frame_name",
    "source_frame",
    "frame_key",
    "droplet_id_in_frame",
    "channel_left_px",
    "channel_right_px",
    "channel_width_px",
    "um_per_px_assuming_40um_channel_width",
    "segment_id",
    "segment_y_start_px",
    "segment_y_end_px",
    "x_centroid_px",
    "y_centroid_px",
    "manual_radius_px",
    "manual_diameter_px",
    "mask_area_px2_contour",
    "mask_bbox_width_px",
    "mask_bbox_height_px",
    "vertical_length_px",
    "vertical_length_um",
    "equivalent_diameter_um",
    "contour_circularity_approx",
    "is_large_droplet",
]

FRAME_COLUMNS = [
    "date",
    "experiment",
    "flow",
    "fraction",
    "Qc_uLh",
    "Qaq_uLh",
    "flow_ratio_Qc_over_Qaq",
    "EtOH_pct",
    "W_pct",
    "video",
    "frame_name",
    "source_frame",
    "mask_frame",
    "overlay_frame",
    "channel_left_px",
    "channel_right_px",
    "channel_width_px",
    "um_per_px_assuming_40um_channel_width",
    "valid_segment_count",
    "valid_segment_total_px",
    "accumulation_score_off_channel",
    "accumulation_frame",
    "n_droplets",
    "n_large_droplets",
    "frame_key",
    "manual_annotated",
    "manual_bad",
    "manual_notes",
    "manual_overlay_frame",
    "calibration_distance_px",
]


@dataclass(frozen=True)
class AnalysisFolder:
    analysis_dir: Path
    condition_dir: Path
    video: str


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def discover_analysis_folders(root: Path) -> list[AnalysisFolder]:
    folders: list[AnalysisFolder] = []
    for analysis_dir in sorted(root.rglob("droplets_analysis")):
        if (analysis_dir / "droplet_measurements.csv").exists() and (
            analysis_dir / "batch_summary.csv"
        ).exists():
            condition_dir = analysis_dir.parent
            folders.append(
                AnalysisFolder(
                    analysis_dir=analysis_dir,
                    condition_dir=condition_dir,
                    video=condition_dir.name,
                )
            )
    return folders


def parse_date_from_path(path: Path) -> str:
    for part in path.parts:
        if re.fullmatch(r"\d{8}", part):
            return f"{part[4:8]}-{part[2:4]}-{part[0:2]}"
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", part):
            return part
    return ""


def infer_metadata(folder: AnalysisFolder) -> dict[str, object]:
    text = str(folder.condition_dir)
    qc = _extract_number(text, r"Qc[_ -]?(\d+)\s*u?L?h?", default=np.nan)
    qaq = _extract_number(text, r"Qaq[_ -]?(\d+)\s*u?L?h?", default=np.nan)
    etoh = _extract_number(text, r"EtOH[_ -]?(\d+)", default=np.nan)
    water = _extract_number(text, r"(?:W|Water|H2O)[_ -]?(\d+)", default=np.nan)
    if math.isnan(water) and not math.isnan(etoh):
        water = 100 - etoh
    experiment = ""
    if not math.isnan(qaq):
        experiment = f"Qaq_{int(qaq)}uLh"
    elif not math.isnan(qc) and not math.isnan(qaq):
        experiment = f"Qtot_{int(qc + qaq)}uLh"
    return {
        "include": True,
        "analysis_dir": str(folder.analysis_dir),
        "video": folder.video,
        "date": parse_date_from_path(folder.condition_dir),
        "experiment": experiment,
        "Qc_uLh": qc,
        "Qaq_uLh": qaq,
        "EtOH_pct": etoh,
        "W_pct": water,
        "manual_bad": False,
        "manual_notes": "",
    }


def _extract_number(text: str, pattern: str, default: float) -> float:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else default


def read_metadata_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "include" not in df:
        df["include"] = True
    return df


def bootstrap_ci_median(values: np.ndarray, n_resamples: int = 2000) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(12345)
    samples = rng.choice(values, size=(n_resamples, len(values)), replace=True)
    medians = np.median(samples, axis=1)
    return tuple(np.percentile(medians, [2.5, 97.5]).astype(float))


def bootstrap_median_diff_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_resamples: int = 3000,
) -> tuple[float, float]:
    values_a = values_a[np.isfinite(values_a)]
    values_b = values_b[np.isfinite(values_b)]
    if len(values_a) == 0 or len(values_b) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(54321)
    samples_a = rng.choice(values_a, size=(n_resamples, len(values_a)), replace=True)
    samples_b = rng.choice(values_b, size=(n_resamples, len(values_b)), replace=True)
    diffs = np.median(samples_b, axis=1) - np.median(samples_a, axis=1)
    return tuple(np.percentile(diffs, [2.5, 97.5]).astype(float))


def iqr(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    return float(np.percentile(values, 75) - np.percentile(values, 25))


def cv(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan
    mean = float(np.mean(values))
    return float(np.std(values, ddof=1) / mean) if mean else np.nan


def robust_cv(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return 1.4826 * mad / median if median else np.nan


def mann_whitney_and_cliff(values_b: np.ndarray, values_a: np.ndarray) -> tuple[float, float]:
    values_a = values_a[np.isfinite(values_a)]
    values_b = values_b[np.isfinite(values_b)]
    if stats is None or len(values_a) == 0 or len(values_b) == 0:
        return np.nan, np.nan
    result = stats.mannwhitneyu(values_b, values_a, alternative="two-sided")
    cliff = (2 * result.statistic / (len(values_a) * len(values_b))) - 1
    return float(result.pvalue), float(cliff)


def ks_test(values_b: np.ndarray, values_a: np.ndarray) -> tuple[float, float]:
    values_a = values_a[np.isfinite(values_a)]
    values_b = values_b[np.isfinite(values_b)]
    if stats is None or len(values_a) == 0 or len(values_b) == 0:
        return np.nan, np.nan
    result = stats.ks_2samp(values_b, values_a)
    return float(result.pvalue), float(result.statistic)


def levene_median_p(values_b: np.ndarray, values_a: np.ndarray) -> float:
    values_a = values_a[np.isfinite(values_a)]
    values_b = values_b[np.isfinite(values_b)]
    if stats is None or len(values_a) < 2 or len(values_b) < 2:
        return np.nan
    return float(stats.levene(values_b, values_a, center="median").pvalue)


def repeatability_interpretation(relative_delta_pct: float) -> str:
    if not np.isfinite(relative_delta_pct):
        return ""
    magnitude = abs(relative_delta_pct)
    if magnitude < 5:
        return "close"
    if magnitude < 15:
        return "moderate shift"
    return "large shift"


def format_p_value(value: float) -> str:
    if pd.isna(value):
        return ""
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def shorten_text(value: object, max_chars: int = 32) -> str:
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def apply_robust_ylim(
    ax: plt.Axes,
    values: np.ndarray,
    lower_q: float = 0.02,
    upper_q: float = 0.98,
) -> None:
    values = values[np.isfinite(values)]
    if len(values) < 5:
        return
    lower = float(np.quantile(values, lower_q))
    upper = float(np.quantile(values, upper_q))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
        return
    padding = 0.12 * (upper - lower)
    y0 = lower - padding
    y1 = upper + padding
    hidden = int(np.sum((values < y0) | (values > y1)))
    ax.set_ylim(y0, y1)
    if hidden:
        ax.text(
            0.99,
            0.02,
            f"{hidden} point(s) outside robust display range",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="#555555",
        )


def stable_frame_key(*parts: object) -> str:
    raw = "|".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def choose_diameter_column(df: pd.DataFrame) -> str:
    for column in [
        "diameter_area_eq_unit",
        "diameter_unit",
        "diameter_transverse_unit",
        "diameter_area_eq_capped_unit",
    ]:
        if column in df:
            return column
    raise ValueError(
        "No diameter column found. Expected one of diameter_area_eq_unit, "
        "diameter_unit, diameter_transverse_unit."
    )


def build_tables(metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    droplet_rows: list[dict[str, object]] = []
    frame_rows: list[dict[str, object]] = []
    condition_rows: list[dict[str, object]] = []

    for meta in metadata.to_dict("records"):
        if not bool(meta.get("include", True)):
            continue
        analysis_dir = Path(str(meta["analysis_dir"]))
        meas_path = analysis_dir / "droplet_measurements.csv"
        summary_path = analysis_dir / "batch_summary.csv"
        settings_path = analysis_dir / "analysis_settings.csv"
        if not meas_path.exists() or not summary_path.exists():
            continue

        measurements = pd.read_csv(meas_path)
        summary = pd.read_csv(summary_path)
        settings = pd.read_csv(settings_path) if settings_path.exists() else pd.DataFrame()

        diameter_col = choose_diameter_column(measurements)
        diameters = pd.to_numeric(measurements[diameter_col], errors="coerce")
        diameters = diameters[np.isfinite(diameters)]

        qc = _clean_number(meta.get("Qc_uLh"))
        qaq = _clean_number(meta.get("Qaq_uLh"))
        etoh = _clean_number(meta.get("EtOH_pct"))
        water = _clean_number(meta.get("W_pct"))
        ratio = qc / qaq if qaq and np.isfinite(qaq) else np.nan
        experiment = str(meta.get("experiment") or "")
        flow = f"Qc_{_fmt_int(qc)}uLh_Qaq_{_fmt_int(qaq)}uLh"
        fraction = f"EtOH_{_fmt_int(etoh)}_W_{_fmt_int(water)}"
        video = str(meta.get("video") or analysis_dir.parent.name)
        date = str(meta.get("date") or parse_date_from_path(analysis_dir))
        manual_bad_condition = bool(meta.get("manual_bad", False))
        manual_notes = str(meta.get("manual_notes") or "")

        condition_key = (experiment, flow, fraction, video)
        threshold = float(np.nanmedian(diameters) * 1.4) if len(diameters) else np.nan

        for _, row in measurements.iterrows():
            frame_name = str(row.get("image", ""))
            frame_key = stable_frame_key(*condition_key, frame_name)
            diameter_um = _clean_number(row.get(diameter_col))
            channel_left = _first_number(row, ["bar_left_px", "global_bar_left"])
            channel_right = _first_number(row, ["bar_right_px", "global_bar_right"])
            channel_width = channel_right - channel_left if np.isfinite(channel_right - channel_left) else _first_number(row, ["channel_width_px"])
            um_per_px = 40.0 / channel_width if channel_width and np.isfinite(channel_width) else np.nan
            vertical_px = _first_number(row, ["longitudinal_length_px", "bbox_height_px", "diameter_px"])
            is_large = int(np.isfinite(threshold) and diameter_um > threshold)
            droplet_rows.append(
                {
                    "date": date,
                    "experiment": experiment,
                    "flow": flow,
                    "fraction": fraction,
                    "Qc_uLh": qc,
                    "Qaq_uLh": qaq,
                    "flow_ratio_Qc_over_Qaq": ratio,
                    "EtOH_pct": etoh,
                    "W_pct": water,
                    "video": video,
                    "frame_name": frame_name,
                    "source_frame": str(analysis_dir.parent / frame_name),
                    "frame_key": frame_key,
                    "droplet_id_in_frame": row.get("id", np.nan),
                    "channel_left_px": channel_left,
                    "channel_right_px": channel_right,
                    "channel_width_px": channel_width,
                    "um_per_px_assuming_40um_channel_width": um_per_px,
                    "segment_id": 1,
                    "segment_y_start_px": np.nan,
                    "segment_y_end_px": np.nan,
                    "x_centroid_px": row.get("center_x_px", np.nan),
                    "y_centroid_px": row.get("center_y_px", np.nan),
                    "manual_radius_px": _first_number(row, ["diameter_px", "bbox_width_px"]) / 2,
                    "manual_diameter_px": _first_number(row, ["diameter_px", "bbox_width_px"]),
                    "mask_area_px2_contour": row.get("area_px", np.nan),
                    "mask_bbox_width_px": row.get("bbox_width_px", np.nan),
                    "mask_bbox_height_px": row.get("bbox_height_px", np.nan),
                    "vertical_length_px": vertical_px,
                    "vertical_length_um": vertical_px * um_per_px if np.isfinite(vertical_px * um_per_px) else np.nan,
                    "equivalent_diameter_um": diameter_um,
                    "contour_circularity_approx": np.nan,
                    "is_large_droplet": is_large,
                }
            )

        droplet_df_tmp = pd.DataFrame(droplet_rows)
        this_condition_droplets = droplet_df_tmp[
            (droplet_df_tmp["experiment"] == experiment)
            & (droplet_df_tmp["flow"] == flow)
            & (droplet_df_tmp["fraction"] == fraction)
            & (droplet_df_tmp["video"] == video)
        ]
        large_by_frame = (
            this_condition_droplets.groupby("frame_name")["is_large_droplet"].sum().to_dict()
            if not this_condition_droplets.empty
            else {}
        )

        for _, row in summary.iterrows():
            frame_name = str(row.get("image", ""))
            frame_key = stable_frame_key(*condition_key, frame_name)
            channel_left = _first_number(row, ["global_bar_left", "bar_left_px"])
            channel_right = _first_number(row, ["global_bar_right", "bar_right_px"])
            channel_width = _first_number(row, ["channel_width_px"])
            if not np.isfinite(channel_width) and np.isfinite(channel_left) and np.isfinite(channel_right):
                channel_width = channel_right - channel_left
            um_per_px = 40.0 / channel_width if channel_width and np.isfinite(channel_width) else np.nan
            frame_bad = manual_bad_condition
            overlay = analysis_dir / "overlays" / Path(frame_name).with_suffix(".png").name
            frame_rows.append(
                {
                    "date": date,
                    "experiment": experiment,
                    "flow": flow,
                    "fraction": fraction,
                    "Qc_uLh": qc,
                    "Qaq_uLh": qaq,
                    "flow_ratio_Qc_over_Qaq": ratio,
                    "EtOH_pct": etoh,
                    "W_pct": water,
                    "video": video,
                    "frame_name": frame_name,
                    "source_frame": str(analysis_dir.parent / frame_name),
                    "mask_frame": "",
                    "overlay_frame": str(overlay) if overlay.exists() else "",
                    "channel_left_px": channel_left,
                    "channel_right_px": channel_right,
                    "channel_width_px": channel_width,
                    "um_per_px_assuming_40um_channel_width": um_per_px,
                    "valid_segment_count": row.get("droplet_count", 0),
                    "valid_segment_total_px": 0,
                    "accumulation_score_off_channel": 0.0,
                    "accumulation_frame": 0,
                    "n_droplets": row.get("droplet_count", 0),
                    "n_large_droplets": large_by_frame.get(frame_name, 0),
                    "frame_key": frame_key,
                    "manual_annotated": 1,
                    "manual_bad": int(frame_bad),
                    "manual_notes": manual_notes,
                    "manual_overlay_frame": str(overlay) if overlay.exists() else "",
                    "calibration_distance_px": channel_width,
                }
            )

        n_frames = int(len(summary))
        n_valid_frames = int(0 if manual_bad_condition else (summary["droplet_count"] > 0).sum())
        n_droplets = int(len(diameters)) if not manual_bad_condition else 0
        condition_diameters = np.array([]) if manual_bad_condition else diameters.to_numpy(float)
        ci_low, ci_high = bootstrap_ci_median(condition_diameters)
        n_large = int(np.sum(condition_diameters > threshold)) if len(condition_diameters) else 0
        status, note = publication_status(n_droplets, 0.0)
        if manual_bad_condition:
            status, note = "excluded", "manual bad condition"
        quality_status = {"usable": "ok", "to_confirm": "a_verifier", "excluded": "a_exclure"}[status]
        quality_flags = "yolo_compiled"
        if manual_bad_condition:
            quality_flags = "manual_bad_frames;yolo_compiled"
        condition_rows.append(
            {
                "date": date,
                "video": video,
                "flow": flow,
                "fraction": fraction,
                "run_label": video,
                "experiment": experiment,
                "Qc_uLh": qc,
                "Qaq_uLh": qaq,
                "flow_ratio_Qc_over_Qaq": ratio,
                "EtOH_pct": etoh,
                "W_pct": water,
                "n_frames": n_frames,
                "n_valid_frames": n_valid_frames,
                "valid_frame_fraction": n_valid_frames / n_frames if n_frames else np.nan,
                "manual_bad_frame_fraction": 1.0 if manual_bad_condition else 0.0,
                "accumulation_frame_fraction": 0.0,
                "n_droplets": n_droplets,
                "n_large_droplets": n_large,
                "large_droplet_fraction": n_large / n_droplets if n_droplets else 0.0,
                "diameter_um_median": np.nanmedian(condition_diameters) if len(condition_diameters) else np.nan,
                "median_ci_low": ci_low,
                "median_ci_high": ci_high,
                "diameter_um_p25": np.nanpercentile(condition_diameters, 25) if len(condition_diameters) else np.nan,
                "diameter_um_p75": np.nanpercentile(condition_diameters, 75) if len(condition_diameters) else np.nan,
                "diameter_um_mean": np.nanmean(condition_diameters) if len(condition_diameters) else np.nan,
                "diameter_um_std": np.nanstd(condition_diameters, ddof=1) if len(condition_diameters) > 1 else np.nan,
                "diameter_cv": (
                    np.nanstd(condition_diameters, ddof=1) / np.nanmean(condition_diameters)
                    if len(condition_diameters) > 1 and np.nanmean(condition_diameters)
                    else np.nan
                ),
                "publication_status": status,
                "publication_note": note,
                "quality_status": quality_status,
                "quality_flags": quality_flags,
            }
        )

    droplet_df = pd.DataFrame(droplet_rows).reindex(columns=DROPLET_COLUMNS)
    frame_df = pd.DataFrame(frame_rows).reindex(columns=FRAME_COLUMNS)
    condition_df = pd.DataFrame(condition_rows)
    return droplet_df, frame_df, condition_df


def _clean_number(value: object) -> float:
    try:
        if pd.isna(value) or value == "":
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def _fmt_int(value: float) -> str:
    return "" if not np.isfinite(value) else str(int(round(value)))


def _first_number(row: pd.Series, columns: Iterable[str]) -> float:
    for column in columns:
        if column in row and pd.notna(row[column]):
            return _clean_number(row[column])
    return np.nan


def publication_status(n_droplets: int, pileup_fraction: float) -> tuple[str, str]:
    if n_droplets < 5:
        return "excluded", "no droplet retained"
    if n_droplets < 10:
        return "to_confirm", "limited n"
    if pileup_fraction >= 0.6:
        return "to_confirm", "majority pile-up"
    return "usable", "ok"


def export_all(
    droplet_df: pd.DataFrame,
    frame_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    output_dir: Path,
    preset: str = "parameter_sweep",
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "compiled_data"
    data_dir.mkdir(exist_ok=True)

    condition_df.reindex(columns=CONDITION_STATS_COLUMNS).to_csv(
        output_dir / "condition_stats_publication.csv", index=False
    )
    droplet_df.to_csv(data_dir / "droplet_measurements_manual.csv", index=False)
    frame_df.to_csv(data_dir / "frame_summary_manual.csv", index=False)
    condition_df.to_csv(data_dir / "condition_summary_manual.csv", index=False)

    apply_publication_style()
    figures = make_figures(droplet_df, frame_df, condition_df, output_dir, preset=preset)
    write_readme(output_dir, preset=preset)
    bundle_pdf(output_dir, figures)
    return figures


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, figures: list[Path]) -> None:
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            fig.tight_layout()
        except RuntimeError:
            # Figures with colorbars and constrained_layout already have a layout
            # engine; saving them directly avoids Matplotlib engine conflicts.
            pass
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    figures.append(pdf)


def add_panel_label(ax: plt.Axes, idx: int) -> None:
    ax.text(
        -0.11,
        1.05,
        ascii_uppercase[idx],
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )


def make_figures(
    droplet_df: pd.DataFrame,
    frame_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    output_dir: Path,
    preset: str = "parameter_sweep",
) -> list[Path]:
    figures: list[Path] = []
    valid = condition_df[np.isfinite(condition_df["diameter_um_median"])].copy()
    colors = {"usable": "#1f77b4", "to_confirm": "#ff7f0e", "excluded": "#7f7f7f"}

    if preset == "repeatability":
        return make_repeatability_figures(droplet_df, frame_df, condition_df, output_dir)
    if preset == "qc_only":
        return make_qc_figures(droplet_df, frame_df, condition_df, output_dir)
    if preset == "parameter_sweep":
        return make_parameter_sweep_figures(droplet_df, frame_df, condition_df, output_dir)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for experiment, group in valid.groupby("experiment", dropna=False):
        group = group.sort_values("flow_ratio_Qc_over_Qaq")
        yerr = np.vstack(
            [
                group["diameter_um_median"] - group["median_ci_low"],
                group["median_ci_high"] - group["diameter_um_median"],
            ]
        )
        ax.errorbar(
            group["flow_ratio_Qc_over_Qaq"],
            group["diameter_um_median"],
            yerr=yerr,
            marker="o",
            capsize=3,
            linewidth=1.4,
            label=str(experiment),
        )
    ax.set_xlabel("Qc / Qaq")
    ax.set_ylabel("Median droplet diameter (um)")
    ax.set_title("Droplet size vs imposed flow ratio")
    ax.legend()
    save_figure(fig, output_dir, "fig01_size_vs_flow", figures)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), constrained_layout=True)
    heatmap(valid, "diameter_um_median", "Median diameter (um)", axes[0])
    heatmap(valid, "diameter_cv", "CV", axes[1])
    save_figure(fig, output_dir, "fig02_regime_heatmaps", figures)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if not droplet_df.empty:
        labels = []
        values = []
        for key, group in droplet_df.groupby(["experiment", "flow", "fraction"], dropna=False):
            clean = group["equivalent_diameter_um"].dropna().to_numpy(float)
            if len(clean):
                labels.append(f"{key[0]}\n{key[1]}\n{key[2]}")
                values.append(clean)
        if values:
            ax.boxplot(values, tick_labels=labels, showfliers=False)
            ax.tick_params(axis="x", rotation=80, labelsize=7)
    ax.set_ylabel("Equivalent diameter (um)")
    ax.set_title("Diameter distributions by condition")
    save_figure(fig, output_dir, "fig03_diameter_distributions", figures)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
    status_counts = condition_df["publication_status"].value_counts()
    axes[0].bar(status_counts.index, status_counts.values, color=[colors.get(x, "#555555") for x in status_counts.index])
    axes[0].set_ylabel("Conditions")
    axes[0].set_title("Publication status")
    axes[1].scatter(condition_df["n_droplets"], condition_df["diameter_cv"], c=condition_df["publication_status"].map(colors))
    axes[1].set_xlabel("Detected droplets")
    axes[1].set_ylabel("CV")
    axes[1].set_title("Monodispersity QC")
    save_figure(fig, output_dir, "fig04_monodispersity_qc", figures)

    make_contact_sheet(frame_df, output_dir, "fig05_representative_masks", max_images=12, figures=figures)

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    for experiment, group in valid.groupby("experiment", dropna=False):
        group = group[(group["flow_ratio_Qc_over_Qaq"] > 0) & (group["diameter_um_median"] > 0)]
        if len(group) < 2:
            continue
        x = np.log(group["flow_ratio_Qc_over_Qaq"].to_numpy(float))
        y = np.log(group["diameter_um_median"].to_numpy(float))
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.scatter(np.exp(x), np.exp(y), label=f"{experiment} data")
        ax.plot(np.exp(xs), np.exp(intercept + slope * xs), label=f"{experiment}: alpha={slope:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Qc / Qaq")
    ax.set_ylabel("Median diameter (um)")
    ax.set_title("Scaling law")
    ax.legend(fontsize=7)
    save_figure(fig, output_dir, "fig06_scaling_law", figures)

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.axis("off")
    stat_text = stat_tests_text(droplet_df, condition_df, preset=preset)
    ax.text(0.01, 0.98, stat_text, va="top", family="monospace", fontsize=7)
    save_figure(fig, output_dir, "fig07_stat_tests", figures)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
    axes[0].hist(droplet_df["equivalent_diameter_um"].dropna(), bins=25, color="#4c78a8", edgecolor="white")
    axes[0].set_xlabel("Equivalent diameter (um)")
    axes[0].set_ylabel("Droplets")
    axes[0].set_title("Global diameter histogram")
    axes[1].hist(droplet_df["manual_diameter_px"].dropna(), bins=np.arange(0, droplet_df["manual_diameter_px"].max() + 2, 1), color="#f58518", edgecolor="white")
    axes[1].set_xlabel("Diameter (px)")
    axes[1].set_title("Pixel discretization")
    save_figure(fig, output_dir, "fig08_pixel_discretization", figures)

    fig, ax = plt.subplots(figsize=(11, max(2.8, 0.35 * len(condition_df) + 1)))
    ax.axis("off")
    table_cols = ["experiment", "Qc_uLh", "Qaq_uLh", "EtOH_pct", "W_pct", "n_droplets", "diameter_um_median", "diameter_cv", "publication_status"]
    table_df = condition_df[table_cols].copy()
    table_df["diameter_um_median"] = table_df["diameter_um_median"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    table_df["diameter_cv"] = table_df["diameter_cv"].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    table = ax.table(cellText=table_df.values, colLabels=table_df.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.25)
    ax.set_title("Summary table", pad=12)
    save_figure(fig, output_dir, "fig09_summary_table", figures)

    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.2))
    axes[0].bar(["frames", "valid"], [frame_df.shape[0], condition_df["n_valid_frames"].sum()], color=["#999999", "#1f77b4"])
    axes[0].set_title("Frame coverage")
    axes[1].bar(["droplets"], [droplet_df.shape[0]], color="#1f77b4")
    axes[1].set_title("Detected droplets")
    axes[2].bar(condition_df["publication_status"].value_counts().index, condition_df["publication_status"].value_counts().values, color="#4c78a8")
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].set_title("QC status")
    save_figure(fig, output_dir, "fig00_method_qc_summary", figures)

    for stem, experiment_filter in [
        ("fig10a_annotation_coverage_qtot", "Qtot"),
        ("fig10b_annotation_coverage_qaq", "Qaq"),
    ]:
        subset = condition_df[condition_df["experiment"].astype(str).str.contains(experiment_filter, case=False, na=False)]
        if subset.empty:
            subset = condition_df
        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        heatmap(subset, "valid_frame_fraction", "Valid frame fraction", ax)
        save_figure(fig, output_dir, stem, figures)

    make_contact_sheet(frame_df, output_dir, "fig11_manual_overlay_contact_sheet_01", max_images=16, figures=figures)
    return figures


def make_parameter_sweep_figures(
    droplet_df: pd.DataFrame,
    frame_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    figures: list[Path] = []
    condition_df, droplet_df, frame_df = prepare_parameter_sweep_frames(droplet_df, frame_df, condition_df)
    frame_stats = compute_parameter_frame_stats(droplet_df, frame_df)
    condition_stats = compute_parameter_condition_stats(condition_df, frame_stats)
    model_stats = compute_parameter_model_stats(condition_stats)
    pairwise_stats = compute_parameter_pairwise_stats(droplet_df, frame_stats, condition_stats)
    group_palette = make_group_palette(condition_stats["group_label"].dropna().unique())

    make_parameter_method_qc_summary(condition_stats, frame_stats, droplet_df, output_dir, figures)
    make_parameter_trend_figure(condition_stats, model_stats, output_dir, figures, group_palette)
    make_parameter_heatmap_figure(condition_stats, output_dir, figures)
    make_parameter_distribution_figure(droplet_df, condition_stats, output_dir, figures, group_palette)
    make_parameter_qc_figure(condition_stats, frame_stats, output_dir, figures, group_palette)
    make_contact_sheet(frame_df, output_dir, "fig05_representative_masks", max_images=12, figures=figures)
    make_parameter_scaling_figure(condition_stats, model_stats, output_dir, figures, group_palette)
    make_parameter_stats_table(model_stats, pairwise_stats, output_dir, figures)
    make_parameter_precision_figure(droplet_df, frame_df, condition_stats, output_dir, figures, group_palette)
    make_summary_table(condition_stats, output_dir, figures)
    make_parameter_coverage_figure(condition_stats, output_dir, figures, "fig10a_annotation_coverage_qtot", "Qtot")
    make_parameter_coverage_figure(condition_stats, output_dir, figures, "fig10b_annotation_coverage_qaq", "Qaq")
    make_contact_sheet_pages(frame_df, output_dir, "fig11_manual_overlay_contact_sheet", max_images=16, figures=figures)
    write_parameter_sweep_stats_csv(condition_stats, frame_stats, model_stats, pairwise_stats, output_dir)
    return figures


def prepare_parameter_sweep_frames(
    droplet_df: pd.DataFrame,
    frame_df: pd.DataFrame,
    condition_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    condition_df = condition_df.copy().reset_index(drop=True)
    for column in RUN_KEY_COLUMNS:
        if column not in condition_df:
            condition_df[column] = ""
    condition_df = condition_df.sort_values(
        by=["experiment", "EtOH_pct", "W_pct", "flow_ratio_Qc_over_Qaq", "Qc_uLh", "Qaq_uLh"],
        na_position="last",
    ).reset_index(drop=True)
    condition_df["condition_id"] = [f"C{i + 1:02d}" for i in range(len(condition_df))]
    condition_df["condition_label"] = condition_df.apply(parameter_condition_label, axis=1)
    condition_df["short_label"] = condition_df.apply(parameter_short_label, axis=1)
    condition_df["group_label"] = condition_df.apply(parameter_group_label, axis=1)

    lookup = condition_df[
        RUN_KEY_COLUMNS + ["condition_id", "condition_label", "short_label", "group_label"]
    ].drop_duplicates()
    droplet_df = droplet_df.copy()
    frame_df = frame_df.copy()
    if not droplet_df.empty:
        droplet_df = droplet_df.merge(lookup, on=RUN_KEY_COLUMNS, how="left")
    if not frame_df.empty:
        frame_df = frame_df.merge(lookup, on=RUN_KEY_COLUMNS, how="left")
        frame_df["frame_index"] = frame_df.groupby("condition_id", dropna=False).cumcount() + 1
    return condition_df, droplet_df, frame_df


def parameter_condition_label(row: pd.Series) -> str:
    condition_id = str(row.get("condition_id", "") or "C")
    experiment = str(row.get("experiment", "") or "").strip()
    qc = _clean_number(row.get("Qc_uLh"))
    qaq = _clean_number(row.get("Qaq_uLh"))
    etoh = _clean_number(row.get("EtOH_pct"))
    water = _clean_number(row.get("W_pct"))
    ratio = _clean_number(row.get("flow_ratio_Qc_over_Qaq"))
    parts = [condition_id]
    if experiment and experiment != "nan":
        parts.append(experiment)
    if np.isfinite(qc) and np.isfinite(qaq):
        parts.append(f"Qc {_fmt_int(qc)} / Qaq {_fmt_int(qaq)}")
    if np.isfinite(ratio):
        parts.append(f"R={ratio:g}")
    if np.isfinite(etoh) and np.isfinite(water):
        parts.append(f"EtOH/W {_fmt_int(etoh)}/{_fmt_int(water)}")
    return " | ".join(parts)


def parameter_short_label(row: pd.Series) -> str:
    condition_id = str(row.get("condition_id", "") or "C")
    ratio = _clean_number(row.get("flow_ratio_Qc_over_Qaq"))
    etoh = _clean_number(row.get("EtOH_pct"))
    water = _clean_number(row.get("W_pct"))
    ratio_text = f"R={ratio:g}" if np.isfinite(ratio) else str(row.get("flow", ""))
    frac_text = f"{_fmt_int(etoh)}/{_fmt_int(water)}" if np.isfinite(etoh) and np.isfinite(water) else str(row.get("fraction", ""))
    return f"{condition_id}\n{ratio_text}\nEtOH/W {frac_text}"


def parameter_group_label(row: pd.Series) -> str:
    experiment = str(row.get("experiment", "") or "").strip()
    etoh = _clean_number(row.get("EtOH_pct"))
    water = _clean_number(row.get("W_pct"))
    fraction = f"EtOH/W {_fmt_int(etoh)}/{_fmt_int(water)}" if np.isfinite(etoh) and np.isfinite(water) else str(row.get("fraction", "") or "")
    if experiment and experiment != "nan":
        return f"{experiment} | {fraction}"
    return fraction or "series"


def make_group_palette(groups: Iterable[object]) -> dict[str, str]:
    clean_groups = [str(group) for group in groups if str(group) and str(group) != "nan"]
    return {group: OKABE_ITO[i % len(OKABE_ITO)] for i, group in enumerate(clean_groups)}


def compute_parameter_frame_stats(droplet_df: pd.DataFrame, frame_df: pd.DataFrame) -> pd.DataFrame:
    if frame_df.empty:
        return pd.DataFrame()
    group_cols = ["condition_id", "condition_label", "short_label", "group_label", "frame_key", "frame_name"]
    droplets = droplet_df.dropna(subset=["equivalent_diameter_um"]).copy()
    if droplets.empty:
        frame_medians = pd.DataFrame(columns=group_cols + ["frame_median_um"])
    else:
        frame_medians = (
            droplets.groupby(group_cols, dropna=False, sort=False)["equivalent_diameter_um"]
            .agg(
                frame_median_um="median",
                frame_mean_um="mean",
                frame_iqr_um=lambda s: s.quantile(0.75) - s.quantile(0.25),
                frame_droplets_from_measurements="count",
            )
            .reset_index()
        )
    base_cols = [
        "condition_id",
        "condition_label",
        "short_label",
        "group_label",
        "frame_key",
        "frame_name",
        "frame_index",
        "n_droplets",
        "calibration_distance_px",
        "manual_bad",
    ]
    existing = [column for column in base_cols if column in frame_df]
    return frame_df[existing].merge(frame_medians, on=group_cols, how="left")


def compute_parameter_condition_stats(
    condition_df: pd.DataFrame,
    frame_stats: pd.DataFrame,
) -> pd.DataFrame:
    stats_df = condition_df.copy()
    rows = []
    for _, row in stats_df.iterrows():
        condition_id = row["condition_id"]
        frames = frame_stats[frame_stats["condition_id"] == condition_id] if not frame_stats.empty else pd.DataFrame()
        frame_medians = frames["frame_median_um"].dropna().to_numpy(float) if not frames.empty else np.array([])
        frame_ci_low, frame_ci_high = bootstrap_ci_median(frame_medians)
        rows.append(
            {
                "condition_id": condition_id,
                "frame_median_um": float(np.nanmedian(frame_medians)) if len(frame_medians) else row.get("diameter_um_median", np.nan),
                "frame_ci_low": frame_ci_low if len(frame_medians) else row.get("median_ci_low", np.nan),
                "frame_ci_high": frame_ci_high if len(frame_medians) else row.get("median_ci_high", np.nan),
                "frame_median_iqr_um": iqr(frame_medians),
                "droplets_per_frame_median": float(np.nanmedian(frames["n_droplets"])) if not frames.empty else np.nan,
                "droplets_per_frame_cv": cv(frames["n_droplets"].to_numpy(float)) if not frames.empty else np.nan,
                "calibration_px_median": float(np.nanmedian(frames["calibration_distance_px"])) if not frames.empty else np.nan,
            }
        )
    stats_df = stats_df.merge(pd.DataFrame(rows), on="condition_id", how="left")
    stats_df["log_flow_ratio"] = np.where(
        stats_df["flow_ratio_Qc_over_Qaq"] > 0,
        np.log(stats_df["flow_ratio_Qc_over_Qaq"]),
        np.nan,
    )
    stats_df["log_diameter"] = np.where(stats_df["frame_median_um"] > 0, np.log(stats_df["frame_median_um"]), np.nan)
    return stats_df


def compute_parameter_model_stats(condition_stats: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    valid = condition_stats.dropna(subset=["frame_median_um"]).copy()
    predictors = [
        ("flow_ratio_Qc_over_Qaq", "Qc/Qaq"),
        ("Qc_uLh", "Qc"),
        ("Qaq_uLh", "Qaq"),
        ("EtOH_pct", "EtOH"),
        ("W_pct", "Water"),
    ]
    for column, label in predictors:
        subset = valid[[column, "frame_median_um"]].dropna()
        if len(subset) >= 3 and subset[column].nunique() >= 2:
            rho, p_value = stats.spearmanr(subset[column], subset["frame_median_um"]) if stats else (np.nan, np.nan)
            rows.append(
                {
                    "analysis": "spearman",
                    "scope": "all conditions",
                    "predictor": label,
                    "n_conditions": len(subset),
                    "estimate": rho,
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "p_value": p_value,
                    "r2": np.nan,
                    "note": "Spearman rho vs frame-level median diameter",
                }
            )

    for group_label, group in valid.groupby("group_label", dropna=False, sort=False):
        subset = group.dropna(subset=["log_flow_ratio", "log_diameter"])
        if len(subset) >= 2 and subset["flow_ratio_Qc_over_Qaq"].nunique() >= 2:
            x = subset["log_flow_ratio"].to_numpy(float)
            y = subset["log_diameter"].to_numpy(float)
            if stats and len(subset) >= 3:
                result = stats.linregress(x, y)
                tcrit = stats.t.ppf(0.975, len(subset) - 2)
                ci_low = result.slope - tcrit * result.stderr
                ci_high = result.slope + tcrit * result.stderr
                p_value = result.pvalue
                r2 = result.rvalue**2
                intercept = result.intercept
            else:
                slope, intercept = np.polyfit(x, y, 1)
                ci_low, ci_high, p_value, r2 = np.nan, np.nan, np.nan, np.nan
            rows.append(
                {
                    "analysis": "loglog_slope",
                    "scope": str(group_label),
                    "predictor": "log(Qc/Qaq)",
                    "n_conditions": len(subset),
                    "estimate": slope if len(subset) < 3 else result.slope,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": p_value,
                    "r2": r2,
                    "intercept": intercept,
                    "note": "log(diameter) = intercept + slope*log(Qc/Qaq)",
                }
            )
    return pd.DataFrame(rows)


def compute_parameter_pairwise_stats(
    droplet_df: pd.DataFrame,
    frame_stats: pd.DataFrame,
    condition_stats: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if len(condition_stats) < 2:
        return pd.DataFrame(rows)
    adjacent_pairs = set()
    for _, group in condition_stats.groupby("group_label", dropna=False, sort=False):
        ordered = group.sort_values("flow_ratio_Qc_over_Qaq")["condition_id"].tolist()
        adjacent_pairs.update(tuple(sorted(pair)) for pair in zip(ordered[:-1], ordered[1:]))

    for i, left in condition_stats.iterrows():
        for j, right in condition_stats.iterrows():
            if j <= i:
                continue
            left_id = left["condition_id"]
            right_id = right["condition_id"]
            left_d = droplet_df.loc[droplet_df["condition_id"] == left_id, "equivalent_diameter_um"].dropna().to_numpy(float)
            right_d = droplet_df.loc[droplet_df["condition_id"] == right_id, "equivalent_diameter_um"].dropna().to_numpy(float)
            left_f = frame_stats.loc[frame_stats["condition_id"] == left_id, "frame_median_um"].dropna().to_numpy(float) if not frame_stats.empty else np.array([])
            right_f = frame_stats.loc[frame_stats["condition_id"] == right_id, "frame_median_um"].dropna().to_numpy(float) if not frame_stats.empty else np.array([])
            delta = np.nanmedian(right_d) - np.nanmedian(left_d) if len(left_d) and len(right_d) else np.nan
            diff_ci_low, diff_ci_high = bootstrap_median_diff_ci(left_d, right_d)
            frame_delta = np.nanmedian(right_f) - np.nanmedian(left_f) if len(left_f) and len(right_f) else np.nan
            frame_ci_low, frame_ci_high = bootstrap_median_diff_ci(left_f, right_f)
            pooled = np.nanmean([np.nanmedian(left_d), np.nanmedian(right_d)])
            mw_p, cliff = mann_whitney_and_cliff(right_d, left_d)
            pair_key = tuple(sorted([left_id, right_id]))
            rows.append(
                {
                    "condition_a_id": left_id,
                    "condition_b_id": right_id,
                    "condition_a": left["condition_label"],
                    "condition_b": right["condition_label"],
                    "same_series": left["group_label"] == right["group_label"],
                    "adjacent_in_series": pair_key in adjacent_pairs,
                    "median_a_um": np.nanmedian(left_d) if len(left_d) else np.nan,
                    "median_b_um": np.nanmedian(right_d) if len(right_d) else np.nan,
                    "delta_b_minus_a_um": delta,
                    "delta_ci_low_um": diff_ci_low,
                    "delta_ci_high_um": diff_ci_high,
                    "relative_delta_pct": 100 * delta / pooled if pooled else np.nan,
                    "frame_median_delta_um": frame_delta,
                    "frame_delta_ci_low_um": frame_ci_low,
                    "frame_delta_ci_high_um": frame_ci_high,
                    "cliffs_delta_b_vs_a": cliff,
                    "mann_whitney_p": mw_p,
                    "n_a": len(left_d),
                    "n_b": len(right_d),
                    "n_frames_a": len(left_f),
                    "n_frames_b": len(right_f),
                    "interpretation": repeatability_interpretation(100 * delta / pooled if pooled else np.nan),
                }
            )
    return pd.DataFrame(rows)


def make_parameter_method_qc_summary(
    condition_stats: pd.DataFrame,
    frame_stats: pd.DataFrame,
    droplet_df: pd.DataFrame,
    output_dir: Path,
    figures: list[Path],
) -> None:
    n_conditions = len(condition_stats)
    n_frames = int(len(frame_stats)) if not frame_stats.empty else int(condition_stats.get("n_frames", pd.Series(dtype=float)).sum())
    if not frame_stats.empty and "manual_bad" in frame_stats:
        manual_bad = pd.to_numeric(frame_stats["manual_bad"], errors="coerce").fillna(0)
        n_bad_frames = int(manual_bad.sum())
        n_valid_frames = int((manual_bad == 0).sum())
    else:
        n_valid_frames = int(condition_stats.get("n_valid_frames", pd.Series(dtype=float)).sum())
        n_bad_frames = max(0, n_frames - n_valid_frames)

    n_droplets = int(pd.to_numeric(condition_stats.get("n_droplets", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
    median_n = condition_stats["n_droplets"].median() if "n_droplets" in condition_stats else np.nan
    median_cv = condition_stats["diameter_cv"].dropna().median() * 100 if "diameter_cv" in condition_stats and condition_stats["diameter_cv"].notna().any() else np.nan
    median_d = condition_stats["diameter_um_median"].dropna().median() if "diameter_um_median" in condition_stats and condition_stats["diameter_um_median"].notna().any() else np.nan

    fig = plt.figure(figsize=(9.2, 6.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25])
    ax_text = fig.add_subplot(gs[0, 0])
    ax_status = fig.add_subplot(gs[0, 1])
    ax_n = fig.add_subplot(gs[1, 0])
    ax_bad = fig.add_subplot(gs[1, 1])

    ax_text.axis("off")
    rows = [
        ["Conditions", f"{n_conditions}"],
        ["Annotated frames", f"{n_frames}"],
        ["Valid / bad frames", f"{n_valid_frames} / {n_bad_frames}"],
        ["Retained droplets", f"{n_droplets}"],
        ["Median n per condition", f"{median_n:.0f}" if np.isfinite(median_n) else "-"],
        ["Median diameter", f"{median_d:.1f} um" if np.isfinite(median_d) else "-"],
        ["Median CV", f"{median_cv:.1f} %" if np.isfinite(median_cv) else "-"],
        ["Calibration", "40 um channel width"],
    ]
    table = ax_text.table(cellText=rows, colLabels=["Item", "Value"], cellLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    table.scale(1.0, 1.25)
    for (row_idx, _col_idx), cell in table.get_celld().items():
        cell.set_linewidth(0.35)
        if row_idx == 0:
            cell.set_facecolor("#ECECEC")
            cell.set_text_props(weight="bold")
    ax_text.set_title("Compiled analysis summary")
    add_panel_label(ax_text, 0)

    counts = condition_stats.get("publication_status", pd.Series(dtype=str)).value_counts()
    status_counts = [int(counts.get(status, 0)) for status in STATUS_ORDER]
    ax_status.bar(
        STATUS_ORDER,
        status_counts,
        color=[STATUS_FACE[status] for status in STATUS_ORDER],
        edgecolor=[STATUS_COLORS[status] for status in STATUS_ORDER],
        linewidth=1.1,
    )
    ax_status.set_xticks(range(len(STATUS_ORDER)), [STATUS_LABELS[status] for status in STATUS_ORDER])
    ax_status.set_ylabel("Number of conditions")
    ax_status.set_title("Publication status")
    ax_status.grid(axis="y", color="#E8E8E8", linewidth=0.7)
    add_panel_label(ax_status, 1)

    droplet_counts = pd.to_numeric(condition_stats.get("n_droplets", pd.Series(dtype=float)), errors="coerce").fillna(0)
    max_count = int(droplet_counts.max()) if len(droplet_counts) else 0
    hist_bins = np.arange(0, max(35, max_count + 5), 5)
    ax_n.hist(droplet_counts, bins=hist_bins, color="#88BDE6", edgecolor="#222222", linewidth=0.5)
    ax_n.axvline(10, color="#B2182B", linestyle="--", linewidth=1.0, label="usable threshold n=10")
    ax_n.set_xlabel("Droplets per condition")
    ax_n.set_ylabel("Number of conditions")
    ax_n.set_title("Coverage by droplet count")
    ax_n.legend(frameon=False, loc="upper right")
    ax_n.grid(axis="y", color="#E8E8E8", linewidth=0.7)
    add_panel_label(ax_n, 2)

    bad_col = "manual_bad_frame_fraction" if "manual_bad_frame_fraction" in condition_stats else "accumulation_frame_fraction"
    plot_rows = condition_stats.sort_values(["experiment", "EtOH_pct", "flow_ratio_Qc_over_Qaq"], na_position="last").reset_index(drop=True)
    y = np.arange(len(plot_rows))
    bad_values = pd.to_numeric(plot_rows.get(bad_col, pd.Series(0, index=plot_rows.index)), errors="coerce").fillna(0)
    row_status = plot_rows.get("publication_status", pd.Series("usable", index=plot_rows.index)).fillna("usable")
    ax_bad.barh(
        y,
        bad_values,
        color=[STATUS_FACE.get(status, "#E4E4E4") for status in row_status],
        edgecolor=[STATUS_COLORS.get(status, "#7A7A7A") for status in row_status],
        linewidth=0.7,
    )
    labels = []
    for _, row in plot_rows.iterrows():
        condition_id = str(row.get("condition_id", "") or "")
        etoh = _clean_number(row.get("EtOH_pct"))
        water = _clean_number(row.get("W_pct"))
        ratio = _clean_number(row.get("flow_ratio_Qc_over_Qaq"))
        frac = f"{_fmt_int(etoh)}/{_fmt_int(water)}" if np.isfinite(etoh) and np.isfinite(water) else str(row.get("fraction", ""))
        ratio_text = f"R={ratio:g}" if np.isfinite(ratio) else str(row.get("flow", ""))
        labels.append(f"{condition_id} {frac} {ratio_text}".strip())
    ax_bad.set_yticks(y, labels, fontsize=7)
    ax_bad.invert_yaxis()
    ax_bad.set_xlim(0, 1.0)
    ax_bad.set_xlabel("Bad frame fraction")
    ax_bad.set_title("Frame exclusion")
    ax_bad.grid(axis="x", color="#E8E8E8", linewidth=0.7)
    add_panel_label(ax_bad, 3)

    fig.suptitle("Annotation and statistical coverage", y=1.02)
    save_figure(fig, output_dir, "fig00_method_qc_summary", figures)


def make_parameter_trend_figure(
    condition_stats: pd.DataFrame,
    model_stats: pd.DataFrame,
    output_dir: Path,
    figures: list[Path],
    group_palette: dict[str, str],
) -> None:
    fig = plt.figure(figsize=(10.6, 6.0))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)
    ax_trend = fig.add_subplot(gs[0, :])
    ax_rel = fig.add_subplot(gs[1, 0])
    ax_table = fig.add_subplot(gs[1, 1])

    valid = condition_stats.dropna(subset=["frame_median_um", "flow_ratio_Qc_over_Qaq"]).copy()
    for group_label, group in valid.groupby("group_label", dropna=False, sort=False):
        group = group.sort_values("flow_ratio_Qc_over_Qaq")
        color = group_palette.get(str(group_label), "#0072B2")
        y = group["frame_median_um"].to_numpy(float)
        yerr = np.vstack([y - group["frame_ci_low"].to_numpy(float), group["frame_ci_high"].to_numpy(float) - y])
        ax_trend.errorbar(
            group["flow_ratio_Qc_over_Qaq"],
            y,
            yerr=yerr,
            marker="o",
            linewidth=1.4,
            capsize=3,
            color=color,
            label=str(group_label),
        )
        for _, row in group.iterrows():
            ax_trend.text(row["flow_ratio_Qc_over_Qaq"], row["frame_median_um"], row["condition_id"], fontsize=7, ha="left", va="bottom")
        baseline = group.iloc[0]["frame_median_um"]
        rel = 100 * (group["frame_median_um"] - baseline) / baseline if baseline else np.nan
        ax_rel.plot(group["flow_ratio_Qc_over_Qaq"], rel, marker="o", color=color, label=str(group_label))
    ax_trend.set_xlabel("Qc / Qaq")
    ax_trend.set_ylabel("Frame-level median diameter (um)")
    ax_trend.set_title("A. Droplet size response to imposed flow ratio")
    ax_trend.legend(fontsize=7)
    ax_rel.axhline(0, color="#555555", linewidth=0.8)
    ax_rel.set_xlabel("Qc / Qaq")
    ax_rel.set_ylabel("Change from series baseline (%)")
    ax_rel.set_title("B. Relative response")

    ax_table.axis("off")
    slopes = model_stats[model_stats.get("analysis", pd.Series(dtype=str)).eq("loglog_slope")] if not model_stats.empty else pd.DataFrame()
    if slopes.empty:
        ax_table.text(0.5, 0.5, "Need >=2 flow-ratio levels per series\nfor log-log slope", ha="center", va="center")
    else:
        table_df = slopes[["scope", "n_conditions", "estimate", "r2", "p_value"]].copy()
        table_df["scope"] = table_df["scope"].map(lambda text: shorten_text(text, 28))
        table_df["estimate"] = table_df["estimate"].map(lambda v: f"{v:.2f}")
        table_df["r2"] = table_df["r2"].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
        table_df["p_value"] = table_df["p_value"].map(format_p_value)
        table_df.columns = ["Series", "n", "alpha", "R2", "p"]
        table = ax_table.table(cellText=table_df.values, colLabels=table_df.columns, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.25)
    ax_table.set_title("C. Log-log scaling summary", pad=10)
    save_figure(fig, output_dir, "fig01_size_vs_flow", figures)


def make_parameter_heatmap_figure(condition_stats: pd.DataFrame, output_dir: Path, figures: list[Path]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0))
    for ax, column, title in [
        (axes[0, 0], "frame_median_um", "Median diameter (um)"),
        (axes[0, 1], "diameter_cv", "Droplet CV"),
        (axes[1, 0], "n_droplets", "Detected droplets"),
        (axes[1, 1], "valid_frame_fraction", "Valid frame fraction"),
    ]:
        heatmap(condition_stats, column, title, ax)
    for ax in axes[0, :]:
        ax.set_xlabel("")
    for ax in axes[:, 1]:
        ax.set_ylabel("")
    fig.subplots_adjust(hspace=0.55, wspace=0.45)
    save_figure(fig, output_dir, "fig02_regime_heatmaps", figures)


def make_parameter_distribution_figure(
    droplet_df: pd.DataFrame,
    condition_stats: pd.DataFrame,
    output_dir: Path,
    figures: list[Path],
    group_palette: dict[str, str],
) -> None:
    fig, ax = plt.subplots(figsize=(max(8.5, 0.55 * max(len(condition_stats), 1)), 4.8))
    rng = np.random.default_rng(42)
    values = []
    labels = []
    positions = []
    colors = []
    for pos, row in enumerate(condition_stats.itertuples(), start=1):
        vals = droplet_df.loc[droplet_df["condition_id"] == row.condition_id, "equivalent_diameter_um"].dropna().to_numpy(float)
        if not len(vals):
            continue
        values.append(vals)
        labels.append(row.short_label)
        positions.append(pos)
        colors.append(group_palette.get(str(row.group_label), "#0072B2"))
    if values:
        violin = ax.violinplot(values, positions=positions, widths=0.75, showmeans=False, showmedians=False, showextrema=False)
        for body, color in zip(violin["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor("none")
            body.set_alpha(0.22)
        ax.boxplot(values, positions=positions, widths=0.22, showfliers=False, patch_artist=True, boxprops={"facecolor": "white", "linewidth": 1.0})
        for pos, vals, color in zip(positions, values, colors):
            sample = vals if len(vals) <= 250 else rng.choice(vals, 250, replace=False)
            ax.scatter(rng.normal(pos, 0.05, len(sample)), sample, s=10, color=color, alpha=0.35, linewidth=0)
            ax.text(pos, np.nanmedian(vals), f"{np.nanmedian(vals):.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
        apply_robust_ylim(ax, np.concatenate(values))
    ax.set_xticks(positions, labels, rotation=65, ha="right")
    ax.set_ylabel("Equivalent diameter (um)")
    ax.set_title("Droplet distributions across parameter conditions")
    save_figure(fig, output_dir, "fig03_diameter_distributions", figures)


def make_parameter_qc_figure(
    condition_stats: pd.DataFrame,
    frame_stats: pd.DataFrame,
    output_dir: Path,
    figures: list[Path],
    group_palette: dict[str, str],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 5.8))
    colors = [group_palette.get(str(group), "#0072B2") for group in condition_stats["group_label"]]
    x = np.arange(len(condition_stats))
    axes[0, 0].bar(x, condition_stats["n_droplets"], color=colors)
    axes[0, 0].set_xticks(x, condition_stats["condition_id"])
    axes[0, 0].set_ylabel("Droplets")
    axes[0, 0].set_title("A. Sample size")

    axes[0, 1].scatter(
        condition_stats["n_droplets"],
        condition_stats["diameter_cv"],
        s=np.maximum(condition_stats["n_frames"], 1) * 18,
        c=colors,
        alpha=0.85,
    )
    axes[0, 1].set_xlabel("Detected droplets")
    axes[0, 1].set_ylabel("CV")
    axes[0, 1].set_title("B. Monodispersity vs sample size")

    axes[1, 0].bar(x, condition_stats["valid_frame_fraction"], color=colors)
    axes[1, 0].set_xticks(x, condition_stats["condition_id"])
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_ylabel("Valid frame fraction")
    axes[1, 0].set_title("C. Frame coverage")

    if frame_stats.empty:
        axes[1, 1].axis("off")
    else:
        axes[1, 1].scatter(
            frame_stats["calibration_distance_px"],
            frame_stats["n_droplets"],
            s=18,
            alpha=0.55,
            color="#555555",
        )
    axes[1, 1].set_xlabel("Calibration distance (px)")
    axes[1, 1].set_ylabel("Droplets/frame")
    axes[1, 1].set_title("D. Calibration vs detections")
    save_figure(fig, output_dir, "fig04_monodispersity_qc", figures)


def make_parameter_scaling_figure(
    condition_stats: pd.DataFrame,
    model_stats: pd.DataFrame,
    output_dir: Path,
    figures: list[Path],
    group_palette: dict[str, str],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    valid = condition_stats.dropna(subset=["flow_ratio_Qc_over_Qaq", "frame_median_um"]).copy()
    valid = valid[(valid["flow_ratio_Qc_over_Qaq"] > 0) & (valid["frame_median_um"] > 0)]
    for group_label, group in valid.groupby("group_label", dropna=False, sort=False):
        color = group_palette.get(str(group_label), "#0072B2")
        axes[0].scatter(group["flow_ratio_Qc_over_Qaq"], group["frame_median_um"], color=color, label=str(group_label))
        group = group.sort_values("flow_ratio_Qc_over_Qaq")
        if len(group) >= 2 and group["flow_ratio_Qc_over_Qaq"].nunique() >= 2:
            x = np.log(group["flow_ratio_Qc_over_Qaq"].to_numpy(float))
            y = np.log(group["frame_median_um"].to_numpy(float))
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 100)
            axes[0].plot(np.exp(xs), np.exp(intercept + slope * xs), color=color, linewidth=1.2)
            residuals = y - (intercept + slope * x)
            axes[1].scatter(group["flow_ratio_Qc_over_Qaq"], residuals, color=color, label=str(group_label))
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Qc / Qaq")
    axes[0].set_ylabel("Frame-level median diameter (um)")
    axes[0].set_title("A. Log-log scaling")
    axes[0].legend(fontsize=7)
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Qc / Qaq")
    axes[1].set_ylabel("Log residual")
    axes[1].set_title("B. Scaling residuals")
    save_figure(fig, output_dir, "fig06_scaling_law", figures)


def make_parameter_stats_table(
    model_stats: pd.DataFrame,
    pairwise_stats: pd.DataFrame,
    output_dir: Path,
    figures: list[Path],
) -> None:
    fig, ax = plt.subplots(figsize=(11.0, 4.2))
    ax.axis("off")
    rows = []
    if not model_stats.empty:
        for _, row in model_stats.head(8).iterrows():
            rows.append(
                [
                    row.get("analysis", ""),
                    row.get("scope", ""),
                    row.get("predictor", ""),
                    "" if pd.isna(row.get("estimate", np.nan)) else f"{row['estimate']:.3f}",
                    "" if pd.isna(row.get("p_value", np.nan)) else format_p_value(row["p_value"]),
                    "" if pd.isna(row.get("r2", np.nan)) else f"{row['r2']:.2f}",
                    row.get("note", ""),
                ]
            )
    if not pairwise_stats.empty:
        candidates = pairwise_stats[pairwise_stats["adjacent_in_series"]].copy()
        if candidates.empty:
            candidates = pairwise_stats.reindex(pairwise_stats["relative_delta_pct"].abs().sort_values(ascending=False).index)
        for _, row in candidates.head(8).iterrows():
            rows.append(
                [
                    "pairwise",
                    f"{row['condition_a_id']} -> {row['condition_b_id']}",
                    "median diameter",
                    f"{row['delta_b_minus_a_um']:+.2f} um",
                    format_p_value(row["mann_whitney_p"]),
                    "",
                    f"{row['relative_delta_pct']:+.1f}%, Cliff={row['cliffs_delta_b_vs_a']:+.2f}",
                ]
            )
    if rows:
        table = ax.table(
            cellText=rows,
            colLabels=["Analysis", "Scope", "Predictor", "Estimate", "p", "R2", "Note"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.35)
    else:
        ax.text(0.5, 0.5, "Not enough conditions for parameter-sweep statistics", ha="center", va="center")
    ax.set_title("Parameter-sweep statistical summary", pad=12)
    save_figure(fig, output_dir, "fig07_stat_tests", figures)


def make_parameter_precision_figure(
    droplet_df: pd.DataFrame,
    frame_df: pd.DataFrame,
    condition_stats: pd.DataFrame,
    output_dir: Path,
    figures: list[Path],
    group_palette: dict[str, str],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.6))
    values_um = droplet_df["equivalent_diameter_um"].dropna().to_numpy(float)
    values_px = droplet_df["manual_diameter_px"].dropna().to_numpy(float)
    if len(values_um):
        axes[0].hist(values_um, bins=25, color="#4c78a8", edgecolor="white")
    axes[0].set_xlabel("Equivalent diameter (um)")
    axes[0].set_ylabel("Droplets")
    axes[0].set_title("A. Global diameter")

    if len(values_px):
        bins = np.arange(np.nanmin(values_px) - 0.5, np.nanmax(values_px) + 1.5, 1)
        axes[1].hist(values_px, bins=bins, color="#999999", edgecolor="white")
    axes[1].set_xlabel("Diameter (px)")
    axes[1].set_title("B. Pixel discretization")

    if not frame_df.empty:
        axes[2].hist(frame_df["calibration_distance_px"].dropna(), bins=12, color="#59a14f", edgecolor="white")
    axes[2].set_xlabel("Calibration distance (px)")
    axes[2].set_title("C. Channel calibration")
    save_figure(fig, output_dir, "fig08_pixel_discretization", figures)


def make_parameter_coverage_figure(
    condition_stats: pd.DataFrame,
    output_dir: Path,
    figures: list[Path],
    stem: str,
    experiment_filter: str,
) -> None:
    subset = condition_stats[condition_stats["experiment"].astype(str).str.contains(experiment_filter, case=False, na=False)]
    if subset.empty:
        subset = condition_stats
    bad_col = "manual_bad_frame_fraction" if "manual_bad_frame_fraction" in subset else "accumulation_frame_fraction"
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 4.1), constrained_layout=True)
    for idx, (ax, column, title) in enumerate(
        [
            (axes[0], "n_droplets", "Detected droplets"),
            (axes[1], "valid_frame_fraction", "Valid frames"),
            (axes[2], bad_col, "Excluded frames"),
        ]
    ):
        heatmap(subset, column, title, ax)
        add_panel_label(ax, idx)
    save_figure(fig, output_dir, stem, figures)


def write_parameter_sweep_stats_csv(
    condition_stats: pd.DataFrame,
    frame_stats: pd.DataFrame,
    model_stats: pd.DataFrame,
    pairwise_stats: pd.DataFrame,
    output_dir: Path,
) -> None:
    condition_stats.to_csv(output_dir / "parameter_sweep_condition_stats.csv", index=False)
    frame_stats.to_csv(output_dir / "parameter_sweep_frame_stats.csv", index=False)
    model_stats.to_csv(output_dir / "parameter_sweep_model_stats.csv", index=False)
    pairwise_stats.to_csv(output_dir / "parameter_sweep_pairwise_stats.csv", index=False)


def make_repeatability_figures(
    droplet_df: pd.DataFrame,
    frame_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    figures: list[Path] = []
    condition_df, droplet_df, frame_df = prepare_repeatability_frames(droplet_df, frame_df, condition_df)
    run_stats = compute_repeatability_run_stats(droplet_df, frame_df, condition_df)
    frame_stats = compute_frame_repeatability_stats(droplet_df, frame_df)
    pairwise_stats = compute_pairwise_repeatability_stats(droplet_df, frame_stats, run_stats)
    palette = {row.run_id: OKABE_ITO[i % len(OKABE_ITO)] for i, row in run_stats.reset_index().iterrows()}

    make_repeatability_dashboard(run_stats, pairwise_stats, output_dir, figures, palette)
    make_repeatability_distributions(droplet_df, run_stats, output_dir, figures, palette)
    make_frame_stability_figure(frame_df, frame_stats, run_stats, output_dir, figures, palette)
    make_repeatability_qc_figure(run_stats, frame_stats, output_dir, figures, palette)
    make_pairwise_repeatability_figure(pairwise_stats, run_stats, output_dir, figures)
    make_distribution_agreement_figure(droplet_df, run_stats, output_dir, figures, palette)
    make_repeatability_stats_table(pairwise_stats, run_stats, output_dir, figures)
    make_measurement_precision_figure(droplet_df, run_stats, output_dir, figures, palette)
    make_summary_table(condition_df, output_dir, figures)
    make_contact_sheet(frame_df, output_dir, "fig05_representative_masks", max_images=12, figures=figures)
    make_contact_sheet(frame_df, output_dir, "fig11_manual_overlay_contact_sheet_01", max_images=16, figures=figures)
    write_replicate_stats_csv(run_stats, frame_stats, pairwise_stats, output_dir)
    return figures


def make_qc_figures(
    droplet_df: pd.DataFrame,
    frame_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    figures: list[Path] = []

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.4))
    axes[0].hist(droplet_df["equivalent_diameter_um"].dropna(), bins=25, color="#4c78a8", edgecolor="white")
    axes[0].set_title("Diameter")
    axes[0].set_xlabel("um")
    axes[1].hist(droplet_df["manual_diameter_px"].dropna(), bins=20, color="#f58518", edgecolor="white")
    axes[1].set_title("Diameter in pixels")
    axes[1].set_xlabel("px")
    axes[2].hist(frame_df["calibration_distance_px"].dropna(), bins=12, color="#59a14f", edgecolor="white")
    axes[2].set_title("Channel calibration")
    axes[2].set_xlabel("px for 40 um")
    save_figure(fig, output_dir, "fig08_pixel_discretization", figures)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
    axes[0].scatter(frame_df["calibration_distance_px"], frame_df["n_droplets"], color="#1f77b4")
    axes[0].set_xlabel("Calibration distance (px)")
    axes[0].set_ylabel("Droplets per frame")
    axes[0].set_title("Calibration vs detections")
    axes[1].bar(condition_df.apply(condition_label, axis=1), condition_df["valid_frame_fraction"], color="#4c78a8")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Valid frame fraction")
    save_figure(fig, output_dir, "fig00_method_qc_summary", figures)

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.axis("off")
    ax.text(0.01, 0.98, stat_tests_text(droplet_df, condition_df, preset="qc_only"), va="top", family="monospace", fontsize=7)
    save_figure(fig, output_dir, "fig07_stat_tests", figures)

    make_summary_table(condition_df, output_dir, figures)
    make_contact_sheet(frame_df, output_dir, "fig05_representative_masks", max_images=12, figures=figures)
    make_contact_sheet(frame_df, output_dir, "fig11_manual_overlay_contact_sheet_01", max_images=16, figures=figures)
    return figures


def prepare_repeatability_frames(
    droplet_df: pd.DataFrame,
    frame_df: pd.DataFrame,
    condition_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    condition_df = condition_df.copy().reset_index(drop=True)
    for column in RUN_KEY_COLUMNS:
        if column not in condition_df:
            condition_df[column] = ""
    condition_df["run_id"] = [f"Run {i + 1}" for i in range(len(condition_df))]
    condition_df["run_label"] = condition_df.apply(repeat_label, axis=1)
    condition_df["short_label"] = condition_df.apply(compact_repeat_label, axis=1)

    lookup = condition_df[RUN_KEY_COLUMNS + ["run_id", "run_label", "short_label"]].drop_duplicates()
    droplet_df = droplet_df.copy()
    frame_df = frame_df.copy()
    if not droplet_df.empty:
        droplet_df = droplet_df.merge(lookup, on=RUN_KEY_COLUMNS, how="left")
    if not frame_df.empty:
        frame_df = frame_df.merge(lookup, on=RUN_KEY_COLUMNS, how="left")
        frame_df["frame_index"] = frame_df.groupby("run_id", dropna=False).cumcount() + 1
    return condition_df, droplet_df, frame_df


def compact_repeat_label(row: pd.Series) -> str:
    run_id = str(row.get("run_id", "") or "Run")
    qc = _clean_number(row.get("Qc_uLh"))
    qaq = _clean_number(row.get("Qaq_uLh"))
    etoh = _clean_number(row.get("EtOH_pct"))
    water = _clean_number(row.get("W_pct"))
    flow = f"Qc {_fmt_int(qc)} / Qaq {_fmt_int(qaq)}" if np.isfinite(qc) and np.isfinite(qaq) else str(row.get("flow", ""))
    fraction = f"EtOH/W {_fmt_int(etoh)}/{_fmt_int(water)}" if np.isfinite(etoh) and np.isfinite(water) else str(row.get("fraction", ""))
    return f"{run_id}\n{flow}\n{fraction}"


def compute_frame_repeatability_stats(droplet_df: pd.DataFrame, frame_df: pd.DataFrame) -> pd.DataFrame:
    frames = frame_df.copy()
    if frames.empty:
        return pd.DataFrame()

    group_cols = ["run_id", "run_label", "short_label", "frame_key", "frame_name"]
    droplets = droplet_df.dropna(subset=["equivalent_diameter_um"]).copy()
    if droplets.empty:
        frame_medians = pd.DataFrame(columns=group_cols + ["frame_median_um", "frame_iqr_um", "frame_droplets_from_measurements"])
    else:
        frame_medians = (
            droplets.groupby(group_cols, dropna=False, sort=False)["equivalent_diameter_um"]
            .agg(
                frame_median_um="median",
                frame_mean_um="mean",
                frame_std_um=lambda s: s.std(ddof=1),
                frame_iqr_um=lambda s: s.quantile(0.75) - s.quantile(0.25),
                frame_droplets_from_measurements="count",
            )
            .reset_index()
        )

    base_cols = [
        "run_id",
        "run_label",
        "short_label",
        "frame_key",
        "frame_name",
        "frame_index",
        "n_droplets",
        "calibration_distance_px",
        "manual_bad",
    ]
    existing = [column for column in base_cols if column in frames]
    result = frames[existing].merge(frame_medians, on=group_cols, how="left")
    return result


def compute_repeatability_run_stats(
    droplet_df: pd.DataFrame,
    frame_df: pd.DataFrame,
    condition_df: pd.DataFrame,
) -> pd.DataFrame:
    frame_stats = compute_frame_repeatability_stats(droplet_df, frame_df)
    rows: list[dict[str, object]] = []
    for _, row in condition_df.iterrows():
        run_id = row["run_id"]
        droplets = droplet_df.loc[droplet_df["run_id"] == run_id, "equivalent_diameter_um"].dropna().to_numpy(float)
        run_frames = frame_stats[frame_stats["run_id"] == run_id] if not frame_stats.empty else pd.DataFrame()
        frame_medians = run_frames["frame_median_um"].dropna().to_numpy(float) if not run_frames.empty else np.array([])
        droplet_ci_low, droplet_ci_high = bootstrap_ci_median(droplets)
        frame_ci_low, frame_ci_high = bootstrap_ci_median(frame_medians)
        median_um = float(np.nanmedian(droplets)) if len(droplets) else np.nan
        n_frames_meta = _clean_number(row.get("n_frames"))
        n_valid_frames_meta = _clean_number(row.get("n_valid_frames"))
        rows.append(
            {
                "run_id": run_id,
                "run_label": row["run_label"],
                "short_label": row["short_label"],
                "experiment": row.get("experiment", ""),
                "flow": row.get("flow", ""),
                "fraction": row.get("fraction", ""),
                "video": row.get("video", ""),
                "n_frames": int(n_frames_meta) if np.isfinite(n_frames_meta) else len(run_frames),
                "n_valid_frames": int(n_valid_frames_meta)
                if np.isfinite(n_valid_frames_meta)
                else int(run_frames["frame_median_um"].notna().sum() if not run_frames.empty else 0),
                "n_droplets": int(len(droplets)),
                "droplet_median_um": median_um,
                "droplet_ci_low": droplet_ci_low,
                "droplet_ci_high": droplet_ci_high,
                "droplet_mean_um": float(np.nanmean(droplets)) if len(droplets) else np.nan,
                "droplet_iqr_um": iqr(droplets),
                "droplet_cv": cv(droplets),
                "droplet_robust_cv": robust_cv(droplets),
                "frame_median_um": float(np.nanmedian(frame_medians)) if len(frame_medians) else median_um,
                "frame_ci_low": frame_ci_low if len(frame_medians) else droplet_ci_low,
                "frame_ci_high": frame_ci_high if len(frame_medians) else droplet_ci_high,
                "droplets_per_frame_median": float(np.nanmedian(run_frames["n_droplets"])) if not run_frames.empty else np.nan,
                "droplets_per_frame_cv": cv(run_frames["n_droplets"].to_numpy(float)) if not run_frames.empty else np.nan,
                "calibration_px_median": float(np.nanmedian(run_frames["calibration_distance_px"])) if not run_frames.empty else np.nan,
            }
        )
    run_stats = pd.DataFrame(rows)
    grand = np.nanmedian(run_stats["frame_median_um"]) if not run_stats.empty else np.nan
    run_stats["relative_shift_from_grand_pct"] = np.where(
        np.isfinite(grand) & (grand != 0),
        100 * (run_stats["frame_median_um"] - grand) / grand,
        np.nan,
    )
    return run_stats


def compute_pairwise_repeatability_stats(
    droplet_df: pd.DataFrame,
    frame_stats: pd.DataFrame,
    run_stats: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for i, left in run_stats.iterrows():
        for j, right in run_stats.iterrows():
            if j <= i:
                continue
            left_d = droplet_df.loc[droplet_df["run_id"] == left["run_id"], "equivalent_diameter_um"].dropna().to_numpy(float)
            right_d = droplet_df.loc[droplet_df["run_id"] == right["run_id"], "equivalent_diameter_um"].dropna().to_numpy(float)
            left_f = frame_stats.loc[frame_stats["run_id"] == left["run_id"], "frame_median_um"].dropna().to_numpy(float) if not frame_stats.empty else np.array([])
            right_f = frame_stats.loc[frame_stats["run_id"] == right["run_id"], "frame_median_um"].dropna().to_numpy(float) if not frame_stats.empty else np.array([])
            delta = np.nanmedian(right_d) - np.nanmedian(left_d) if len(left_d) and len(right_d) else np.nan
            diff_ci_low, diff_ci_high = bootstrap_median_diff_ci(left_d, right_d)
            frame_delta = np.nanmedian(right_f) - np.nanmedian(left_f) if len(left_f) and len(right_f) else np.nan
            frame_ci_low, frame_ci_high = bootstrap_median_diff_ci(left_f, right_f)
            pooled = np.nanmean([np.nanmedian(left_d), np.nanmedian(right_d)])
            mw_p, cliff = mann_whitney_and_cliff(right_d, left_d)
            ks_p, ks_d = ks_test(right_d, left_d)
            levene_p = levene_median_p(right_d, left_d)
            rows.append(
                {
                    "run_a_id": left["run_id"],
                    "run_b_id": right["run_id"],
                    "run_a": left["run_label"],
                    "run_b": right["run_label"],
                    "median_a_um": np.nanmedian(left_d) if len(left_d) else np.nan,
                    "median_b_um": np.nanmedian(right_d) if len(right_d) else np.nan,
                    "delta_b_minus_a_um": delta,
                    "delta_ci_low_um": diff_ci_low,
                    "delta_ci_high_um": diff_ci_high,
                    "relative_delta_pct": 100 * delta / pooled if pooled else np.nan,
                    "frame_median_delta_um": frame_delta,
                    "frame_delta_ci_low_um": frame_ci_low,
                    "frame_delta_ci_high_um": frame_ci_high,
                    "cliffs_delta_b_vs_a": cliff,
                    "mann_whitney_p": mw_p,
                    "ks_d": ks_d,
                    "ks_p": ks_p,
                    "brown_forsythe_p": levene_p,
                    "n_a": len(left_d),
                    "n_b": len(right_d),
                    "n_frames_a": len(left_f),
                    "n_frames_b": len(right_f),
                    "interpretation": repeatability_interpretation(100 * delta / pooled if pooled else np.nan),
                }
            )
    return pd.DataFrame(rows)


def make_repeatability_dashboard(
    run_stats: pd.DataFrame,
    pairwise_stats: pd.DataFrame,
    output_dir: Path,
    figures: list[Path],
    palette: dict[str, str],
) -> None:
    fig = plt.figure(figsize=(10.5, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 0.85], wspace=0.35, hspace=0.5)
    ax_median = fig.add_subplot(gs[0, 0])
    ax_shift = fig.add_subplot(gs[0, 1])
    ax_delta = fig.add_subplot(gs[1, 0])
    ax_table = fig.add_subplot(gs[1, 1])

    x = np.arange(len(run_stats))
    y = run_stats["frame_median_um"].to_numpy(float)
    yerr = np.vstack([y - run_stats["frame_ci_low"].to_numpy(float), run_stats["frame_ci_high"].to_numpy(float) - y])
    for idx, row in run_stats.iterrows():
        ax_median.errorbar(
            idx,
            row["frame_median_um"],
            yerr=np.array([[row["frame_median_um"] - row["frame_ci_low"]], [row["frame_ci_high"] - row["frame_median_um"]]]),
            fmt="o",
            capsize=4,
            color=palette[row["run_id"]],
            markersize=7,
        )
        ax_median.scatter(idx, row["droplet_median_um"], marker="s", s=28, color=palette[row["run_id"]], alpha=0.45)
        ax_median.text(idx, row["frame_ci_high"], f"{row['frame_median_um']:.2f}", ha="center", va="bottom", fontsize=7)
    ax_median.set_xticks(x, run_stats["short_label"])
    ax_median.set_ylabel("Diameter (um)")
    ax_median.set_title("A. Run median diameter\ncircles: frame-median CI; squares: droplet median")

    ax_shift.axhline(0, color="#555555", linewidth=0.8)
    ax_shift.bar(
        x,
        run_stats["relative_shift_from_grand_pct"],
        color=[palette[r] for r in run_stats["run_id"]],
        alpha=0.85,
    )
    ax_shift.set_xticks(x, run_stats["run_id"])
    ax_shift.set_ylabel("Shift from grand median (%)")
    ax_shift.set_title("B. Relative run shift")

    if pairwise_stats.empty:
        ax_delta.axis("off")
        ax_delta.text(0.5, 0.5, "Need at least two runs", ha="center", va="center")
    else:
        yy = np.arange(len(pairwise_stats))
        ax_delta.axvline(0, color="#555555", linewidth=0.8)
        ax_delta.errorbar(
            pairwise_stats["delta_b_minus_a_um"],
            yy,
            xerr=np.vstack(
                [
                    pairwise_stats["delta_b_minus_a_um"] - pairwise_stats["delta_ci_low_um"],
                    pairwise_stats["delta_ci_high_um"] - pairwise_stats["delta_b_minus_a_um"],
                ]
            ),
            fmt="o",
            capsize=3,
            color="#0072B2",
        )
        labels = pairwise_stats["run_b_id"] + " - " + pairwise_stats["run_a_id"]
        ax_delta.set_yticks(yy, labels)
        ax_delta.set_xlabel("Median difference (um)")
    ax_delta.set_title("C. Pairwise median differences")

    ax_table.axis("off")
    table_df = run_stats[["run_id", "n_frames", "n_droplets", "frame_median_um", "droplet_robust_cv"]].copy()
    table_df.columns = ["Run", "Frames", "Droplets", "Frame median", "Robust CV"]
    table_df["Frame median"] = table_df["Frame median"].map(lambda v: f"{v:.2f}")
    table_df["Robust CV"] = table_df["Robust CV"].map(lambda v: f"{v:.3f}")
    table = ax_table.table(cellText=table_df.values, colLabels=table_df.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.3)
    ax_table.set_title("D. Run summary", pad=8)
    save_figure(fig, output_dir, "fig01_size_vs_flow", figures)


def make_repeatability_distributions(
    droplet_df: pd.DataFrame,
    run_stats: pd.DataFrame,
    output_dir: Path,
    figures: list[Path],
    palette: dict[str, str],
) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    rng = np.random.default_rng(42)
    values = []
    positions = []
    labels = []
    run_ids = []
    for idx, row in run_stats.iterrows():
        clean = droplet_df.loc[droplet_df["run_id"] == row["run_id"], "equivalent_diameter_um"].dropna().to_numpy(float)
        if not len(clean):
            continue
        values.append(clean)
        positions.append(idx + 1)
        labels.append(row["short_label"])
        run_ids.append(row["run_id"])
    if values:
        violin = ax.violinplot(values, positions=positions, widths=0.75, showmeans=False, showmedians=False, showextrema=False)
        for body, run_id in zip(violin["bodies"], run_ids):
            body.set_facecolor(palette[run_id])
            body.set_edgecolor("none")
            body.set_alpha(0.25)
        ax.boxplot(values, positions=positions, widths=0.22, showfliers=False, patch_artist=True, boxprops={"facecolor": "white", "linewidth": 1.0})
        for pos, vals, run_id in zip(positions, values, run_ids):
            sample = vals if len(vals) <= 400 else rng.choice(vals, 400, replace=False)
            jitter = rng.normal(pos, 0.055, size=len(sample))
            ax.scatter(jitter, sample, s=12, color=palette[run_id], alpha=0.35, linewidth=0)
            ax.text(pos, np.nanmedian(vals), f"{np.nanmedian(vals):.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
        apply_robust_ylim(ax, np.concatenate(values))
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Equivalent diameter (um)")
    ax.set_title("Droplet distributions by repeated run")
    save_figure(fig, output_dir, "fig03_diameter_distributions", figures)


def make_frame_stability_figure(
    frame_df: pd.DataFrame,
    frame_stats: pd.DataFrame,
    run_stats: pd.DataFrame,
    output_dir: Path,
    figures: list[Path],
    palette: dict[str, str],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), sharex=False)
    for row in run_stats.itertuples():
        frames = frame_df[frame_df["run_id"] == row.run_id]
        medians = frame_stats[frame_stats["run_id"] == row.run_id] if not frame_stats.empty else pd.DataFrame()
        color = palette[row.run_id]
        if not frames.empty:
            axes[0].plot(frames["frame_index"], frames["n_droplets"], marker="o", linewidth=1.2, color=color, label=row.run_id)
            axes[2].plot(frames["frame_index"], frames["calibration_distance_px"], marker="o", linewidth=1.2, color=color)
        if not medians.empty:
            axes[1].plot(medians["frame_index"], medians["frame_median_um"], marker="o", linewidth=1.2, color=color, label=row.run_id)
    axes[0].set_ylabel("Droplets/frame")
    axes[0].set_title("A. Detection count")
    axes[1].set_ylabel("Frame median diameter (um)")
    axes[1].set_title("B. Frame-level diameter")
    axes[2].set_ylabel("Calibration distance (px)")
    axes[2].set_title("C. Channel calibration")
    for ax in axes:
        ax.set_xlabel("Frame index")
    axes[0].legend(fontsize=7)
    save_figure(fig, output_dir, "fig00_method_qc_summary", figures)


def make_repeatability_qc_figure(
    run_stats: pd.DataFrame,
    frame_stats: pd.DataFrame,
    output_dir: Path,
    figures: list[Path],
    palette: dict[str, str],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 5.8))
    x = np.arange(len(run_stats))
    colors = [palette[r] for r in run_stats["run_id"]]
    axes[0, 0].bar(x - 0.18, run_stats["droplet_cv"], width=0.36, color=colors, alpha=0.45, label="standard CV")
    axes[0, 0].bar(x + 0.18, run_stats["droplet_robust_cv"], width=0.36, color=colors, label="robust CV")
    axes[0, 0].set_xticks(x, run_stats["run_id"])
    axes[0, 0].set_ylabel("CV")
    axes[0, 0].set_title("A. Within-run dispersion")
    axes[0, 0].legend(fontsize=7)

    axes[0, 1].bar(x - 0.18, run_stats["n_frames"], width=0.36, color="#999999", label="frames")
    axes[0, 1].bar(x + 0.18, run_stats["n_droplets"], width=0.36, color=colors, label="droplets")
    axes[0, 1].set_xticks(x, run_stats["run_id"])
    axes[0, 1].set_title("B. Sample size")
    axes[0, 1].legend(fontsize=7)

    axes[1, 0].bar(x, run_stats["droplets_per_frame_cv"], color=colors)
    axes[1, 0].set_xticks(x, run_stats["run_id"])
    axes[1, 0].set_ylabel("CV of droplets/frame")
    axes[1, 0].set_title("C. Detection stability")

    if frame_stats.empty:
        axes[1, 1].axis("off")
    else:
        labels = []
        values = []
        for row in run_stats.itertuples():
            vals = frame_stats.loc[frame_stats["run_id"] == row.run_id, "frame_median_um"].dropna().to_numpy(float)
            if len(vals):
                labels.append(row.run_id)
                values.append(vals)
        if values:
            axes[1, 1].boxplot(values, tick_labels=labels, showfliers=False)
    axes[1, 1].set_ylabel("Frame median diameter (um)")
    axes[1, 1].set_title("D. Frame-level spread")
    save_figure(fig, output_dir, "fig04_monodispersity_qc", figures)


def make_pairwise_repeatability_figure(
    pairwise_stats: pd.DataFrame,
    run_stats: pd.DataFrame,
    output_dir: Path,
    figures: list[Path],
) -> None:
    run_ids = list(run_stats["run_id"])
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), constrained_layout=True)
    for ax, column, title, cmap in [
        (axes[0], "relative_delta_pct", "Relative median difference (%)", "PuOr"),
        (axes[1], "cliffs_delta_b_vs_a", "Cliff's delta (distribution shift)", "BrBG"),
    ]:
        matrix = pd.DataFrame(np.nan, index=run_ids, columns=run_ids)
        for _, row in pairwise_stats.iterrows():
            matrix.loc[row["run_b_id"], row["run_a_id"]] = row[column]
            matrix.loc[row["run_a_id"], row["run_b_id"]] = -row[column] if column != "mann_whitney_p" else row[column]
        vmax = np.nanmax(np.abs(matrix.to_numpy(float))) if np.isfinite(matrix.to_numpy(float)).any() else 1
        image = ax.imshow(matrix.to_numpy(float), cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(run_ids)), run_ids)
        ax.set_yticks(range(len(run_ids)), run_ids)
        ax.set_title(title)
        for i in range(len(run_ids)):
            for j in range(len(run_ids)):
                value = matrix.iloc[i, j]
                if pd.notna(value):
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, output_dir, "fig02_regime_heatmaps", figures)


def make_distribution_agreement_figure(
    droplet_df: pd.DataFrame,
    run_stats: pd.DataFrame,
    output_dir: Path,
    figures: list[Path],
    palette: dict[str, str],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0))
    for row in run_stats.itertuples():
        vals = np.sort(droplet_df.loc[droplet_df["run_id"] == row.run_id, "equivalent_diameter_um"].dropna().to_numpy(float))
        if len(vals):
            y = np.arange(1, len(vals) + 1) / len(vals)
            axes[0].step(vals, y, where="post", color=palette[row.run_id], label=row.run_id)
    axes[0].set_xlabel("Equivalent diameter (um)")
    axes[0].set_ylabel("Empirical cumulative probability")
    axes[0].set_title("A. ECDF overlay")
    axes[0].legend(fontsize=7)

    if len(run_stats) >= 2:
        left = run_stats.iloc[0]
        right = run_stats.iloc[1]
        a = np.sort(droplet_df.loc[droplet_df["run_id"] == left["run_id"], "equivalent_diameter_um"].dropna().to_numpy(float))
        b = np.sort(droplet_df.loc[droplet_df["run_id"] == right["run_id"], "equivalent_diameter_um"].dropna().to_numpy(float))
        if len(a) and len(b):
            qs = np.linspace(0.05, 0.95, 19)
            qa = np.quantile(a, qs)
            qb = np.quantile(b, qs)
            low = min(np.nanmin(qa), np.nanmin(qb))
            high = max(np.nanmax(qa), np.nanmax(qb))
            axes[1].plot([low, high], [low, high], color="#555555", linewidth=0.8)
            axes[1].scatter(qa, qb, color="#0072B2")
            axes[1].set_xlabel(f"{left['run_id']} quantiles (um)")
            axes[1].set_ylabel(f"{right['run_id']} quantiles (um)")
    axes[1].set_title("B. Q-Q agreement")
    save_figure(fig, output_dir, "fig06_scaling_law", figures)


def make_measurement_precision_figure(
    droplet_df: pd.DataFrame,
    run_stats: pd.DataFrame,
    output_dir: Path,
    figures: list[Path],
    palette: dict[str, str],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.6))
    all_um = droplet_df["equivalent_diameter_um"].dropna().to_numpy(float)
    all_px = droplet_df["manual_diameter_px"].dropna().to_numpy(float)
    for row in run_stats.itertuples():
        values = droplet_df.loc[droplet_df["run_id"] == row.run_id, "equivalent_diameter_um"].dropna().to_numpy(float)
        if len(values):
            axes[0].hist(values, bins=18, histtype="step", linewidth=1.5, color=palette[row.run_id], label=row.run_id)
    axes[0].set_xlabel("Equivalent diameter (um)")
    axes[0].set_ylabel("Droplets")
    axes[0].set_title("A. Diameter histogram")
    axes[0].legend(fontsize=7)

    if len(all_px):
        bins = np.arange(np.nanmin(all_px) - 0.5, np.nanmax(all_px) + 1.5, 1)
        axes[1].hist(all_px, bins=bins, color="#999999", edgecolor="white")
    axes[1].set_xlabel("Diameter (px)")
    axes[1].set_ylabel("Droplets")
    axes[1].set_title("B. Pixel discretization")

    if len(all_px) and len(all_um):
        for row in run_stats.itertuples():
            subset = droplet_df[droplet_df["run_id"] == row.run_id]
            axes[2].scatter(
                subset["manual_diameter_px"],
                subset["equivalent_diameter_um"],
                s=13,
                alpha=0.45,
                color=palette[row.run_id],
                label=row.run_id,
            )
    axes[2].set_xlabel("Diameter (px)")
    axes[2].set_ylabel("Equivalent diameter (um)")
    axes[2].set_title("C. Pixel-to-um mapping")
    apply_robust_ylim(axes[2], all_um)
    save_figure(fig, output_dir, "fig08_pixel_discretization", figures)


def make_repeatability_stats_table(
    pairwise_stats: pd.DataFrame,
    run_stats: pd.DataFrame,
    output_dir: Path,
    figures: list[Path],
) -> None:
    fig, ax = plt.subplots(figsize=(11.0, max(3.0, 0.42 * max(len(pairwise_stats), 1) + 1.4)))
    ax.axis("off")
    if pairwise_stats.empty:
        ax.text(0.5, 0.5, "Need at least two runs for pairwise statistics", ha="center", va="center")
    else:
        table_df = pairwise_stats[
            [
                "run_a_id",
                "run_b_id",
                "delta_b_minus_a_um",
                "delta_ci_low_um",
                "delta_ci_high_um",
                "relative_delta_pct",
                "cliffs_delta_b_vs_a",
                "mann_whitney_p",
                "ks_p",
                "brown_forsythe_p",
                "interpretation",
            ]
        ].copy()
        table_df["delta_um_95ci"] = table_df.apply(
            lambda r: f"{r['delta_b_minus_a_um']:+.2f} [{r['delta_ci_low_um']:+.2f}, {r['delta_ci_high_um']:+.2f}]",
            axis=1,
        )
        table_df["relative_delta_pct"] = table_df["relative_delta_pct"].map(lambda v: f"{v:+.1f}")
        table_df["cliffs_delta_b_vs_a"] = table_df["cliffs_delta_b_vs_a"].map(lambda v: "" if pd.isna(v) else f"{v:+.2f}")
        for column in ["mann_whitney_p", "ks_p", "brown_forsythe_p"]:
            table_df[column] = table_df[column].map(format_p_value)
        table_df = table_df[
            [
                "run_a_id",
                "run_b_id",
                "delta_um_95ci",
                "relative_delta_pct",
                "cliffs_delta_b_vs_a",
                "mann_whitney_p",
                "ks_p",
                "brown_forsythe_p",
                "interpretation",
            ]
        ]
        table_df.columns = ["A", "B", "Delta um\n[95% CI]", "Delta %", "Cliff", "MW p", "KS p", "BF p", "Interpretation"]
        table = ax.table(cellText=table_df.values, colLabels=table_df.columns, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.45)
    ax.set_title("Pairwise repeatability statistics", pad=12)
    save_figure(fig, output_dir, "fig07_stat_tests", figures)


def condition_label(row: pd.Series) -> str:
    experiment = str(row.get("experiment", "") or "").strip()
    flow = str(row.get("flow", "") or "").strip()
    fraction = str(row.get("fraction", "") or "").strip()
    video = str(row.get("video", "") or "").strip()
    parts = [p for p in [experiment, flow, fraction] if p and p != "nan"]
    if parts:
        return " | ".join(parts)
    return video[:28] if video else "run"


def repeat_label(row: pd.Series) -> str:
    base = condition_label(row)
    video = str(row.get("video", "") or "").strip()
    if video and video != "nan":
        return f"{base} | {video}"
    return base


def grouped_droplet_values(droplet_df: pd.DataFrame, by: list[str]) -> tuple[list[str], list[np.ndarray]]:
    labels: list[str] = []
    values: list[np.ndarray] = []
    if droplet_df.empty:
        return labels, values
    for key, group in droplet_df.groupby(by, dropna=False, sort=False):
        clean = group["equivalent_diameter_um"].dropna().to_numpy(float)
        if len(clean):
            if not isinstance(key, tuple):
                key = (key,)
            labels.append(" | ".join(str(k) for k in key if str(k) and str(k) != "nan")[:60])
            values.append(clean)
    return labels, values


def make_summary_table(condition_df: pd.DataFrame, output_dir: Path, figures: list[Path]) -> None:
    fig, ax = plt.subplots(figsize=(11, max(2.8, 0.35 * len(condition_df) + 1)))
    ax.axis("off")
    table_cols = ["experiment", "Qc_uLh", "Qaq_uLh", "EtOH_pct", "W_pct", "n_droplets", "diameter_um_median", "diameter_cv", "publication_status"]
    table_df = condition_df[table_cols].copy()
    table_df["diameter_um_median"] = table_df["diameter_um_median"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    table_df["diameter_cv"] = table_df["diameter_cv"].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    table = ax.table(cellText=table_df.values, colLabels=table_df.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.25)
    ax.set_title("Summary table", pad=12)
    save_figure(fig, output_dir, "fig09_summary_table", figures)


def heatmap(df: pd.DataFrame, value_col: str, title: str, ax: plt.Axes) -> None:
    if df.empty:
        ax.set_title(title)
        return
    pivot = df.pivot_table(
        index="EtOH_pct",
        columns="flow_ratio_Qc_over_Qaq",
        values=value_col,
        aggfunc="median",
    ).sort_index(ascending=False)
    values = pivot.to_numpy(float)
    vmin, vmax = (0, 1) if value_col == "valid_frame_fraction" else (None, None)
    image = ax.imshow(values, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)), [f"{x:g}" for x in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index)), [f"{x:g}" for x in pivot.index])
    ax.set_xlabel("Qc / Qaq")
    ax.set_ylabel("EtOH (%)")
    ax.set_title(title)
    finite = values[np.isfinite(values)]
    threshold = np.nanmedian(finite) if len(finite) else np.nan
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            if np.isfinite(value):
                color = "white" if np.isfinite(threshold) and value < threshold else "black"
                ax.text(col_idx, row_idx, format_heatmap_value(value), ha="center", va="center", fontsize=7, color=color)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def format_heatmap_value(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def write_replicate_stats_csv(
    run_stats: pd.DataFrame,
    frame_stats: pd.DataFrame,
    pairwise_stats: pd.DataFrame,
    output_dir: Path,
) -> None:
    run_stats.to_csv(output_dir / "repeatability_run_stats.csv", index=False)
    frame_stats.to_csv(output_dir / "repeatability_frame_stats.csv", index=False)
    pairwise_stats.to_csv(output_dir / "repeatability_pairwise_stats.csv", index=False)


def stat_tests_text(
    droplet_df: pd.DataFrame,
    condition_df: pd.DataFrame | None = None,
    preset: str = "parameter_sweep",
) -> str:
    if droplet_df.empty or stats is None:
        return "No statistical test available."
    if preset == "repeatability":
        return repeatability_text(droplet_df, condition_df)
    if preset == "qc_only":
        return qc_text(droplet_df, condition_df)
    lines = ["Frame-level/non-parametric summary", ""]
    groups = []
    names = []
    for name, group in droplet_df.groupby(["experiment", "flow", "fraction"], dropna=False):
        values = group["equivalent_diameter_um"].dropna().to_numpy(float)
        if len(values) >= 2:
            names.append(" | ".join(map(str, name)))
            groups.append(values)
    if len(groups) >= 2:
        h, p = stats.kruskal(*groups)
        lines.append(f"Kruskal-Wallis across conditions: H={h:.3f}, p={p:.3g}")
    else:
        lines.append("Not enough groups for Kruskal-Wallis.")
    lines.append("")
    lines.append("Condition medians:")
    for name, values in zip(names, groups):
        lines.append(f"- {name}: n={len(values)}, median={np.median(values):.2f} um")
    return "\n".join(lines[:55])


def repeatability_text(droplet_df: pd.DataFrame, condition_df: pd.DataFrame | None) -> str:
    lines = [
        "Repeatability summary",
        "",
        "Recommended stats for repeated runs:",
        "- frame-level median with 95% bootstrap CI",
        "- pairwise median difference with bootstrap CI",
        "- relative shift, robust CV and frame-to-frame stability",
        "- Mann-Whitney / KS / Brown-Forsythe as descriptive diagnostics",
        "",
    ]
    if condition_df is None or condition_df.empty:
        return "\n".join(lines + ["No condition summary available."])
    condition_df, droplet_df, empty_frames = prepare_repeatability_frames(
        droplet_df,
        pd.DataFrame(columns=FRAME_COLUMNS),
        condition_df,
    )
    run_stats = compute_repeatability_run_stats(droplet_df, empty_frames, condition_df)
    pairwise_stats = compute_pairwise_repeatability_stats(
        droplet_df,
        pd.DataFrame(),
        run_stats,
    )
    for _, row in run_stats.iterrows():
        lines.append(
            f"{row['run_id']}: n={int(row['n_droplets'])}, "
            f"median={row['droplet_median_um']:.2f} um "
            f"[{row['droplet_ci_low']:.2f}, {row['droplet_ci_high']:.2f}], "
            f"robust CV={row['droplet_robust_cv']:.3f}"
        )
    if not pairwise_stats.empty:
        lines.append("")
        lines.append("Pairwise median differences:")
        for _, row in pairwise_stats.iterrows():
            lines.append(
                f"{row['run_b_id']} - {row['run_a_id']}: "
                f"{row['delta_b_minus_a_um']:+.2f} um "
                f"[{row['delta_ci_low_um']:+.2f}, {row['delta_ci_high_um']:+.2f}], "
                f"{row['relative_delta_pct']:+.1f}%, Cliff={row['cliffs_delta_b_vs_a']:+.2f}"
            )
    lines.append("")
    lines.append("Droplet-level p-values are descriptive; frame-level summaries are preferred for repeated-video comparison.")
    return "\n".join(lines[:55])


def qc_text(droplet_df: pd.DataFrame, condition_df: pd.DataFrame | None) -> str:
    lines = ["YOLO/QC summary", ""]
    lines.append(f"Total droplets: {len(droplet_df)}")
    if not droplet_df.empty:
        d = droplet_df["equivalent_diameter_um"].dropna()
        px = droplet_df["manual_diameter_px"].dropna()
        lines.append(f"Diameter median: {d.median():.2f} um")
        lines.append(f"Diameter IQR: {d.quantile(0.25):.2f} - {d.quantile(0.75):.2f} um")
        lines.append(f"Pixel diameter range: {px.min():.1f} - {px.max():.1f} px")
    if condition_df is not None and not condition_df.empty:
        lines.append("")
        lines.append("Conditions flagged:")
        for _, row in condition_df.iterrows():
            lines.append(
                f"{condition_label(row)}: {row['publication_status']}, "
                f"valid frames={row['valid_frame_fraction']:.2f}, n={int(row['n_droplets'])}"
            )
    return "\n".join(lines[:55])


def make_contact_sheet(
    frame_df: pd.DataFrame,
    output_dir: Path,
    stem: str,
    max_images: int,
    figures: list[Path],
) -> None:
    paths = [path for path, _row in overlay_records(frame_df)[:max_images]]
    if not paths:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "No overlay images found", ha="center", va="center")
        save_figure(fig, output_dir, stem, figures)
        return
    thumbs = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        img.thumbnail((360, 260))
        img = ImageOps.expand(img, border=8, fill="white")
        thumbs.append((path.name, img.copy()))
    cols = min(4, len(thumbs))
    rows = math.ceil(len(thumbs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.1, rows * 2.4))
    axes_arr = np.atleast_1d(axes).ravel()
    for ax, (name, img) in zip(axes_arr, thumbs):
        ax.imshow(img)
        ax.set_title(name, fontsize=7)
        ax.axis("off")
    for ax in axes_arr[len(thumbs) :]:
        ax.axis("off")
    save_figure(fig, output_dir, stem, figures)


def overlay_records(frame_df: pd.DataFrame) -> list[tuple[Path, pd.Series]]:
    records: list[tuple[Path, pd.Series]] = []
    seen: set[Path] = set()
    if frame_df.empty:
        return records
    for _, row in frame_df.iterrows():
        for column in ["manual_overlay_frame", "overlay_frame"]:
            value = row.get(column, "")
            if pd.isna(value) or not str(value):
                continue
            path = Path(str(value))
            if path.exists() and path not in seen:
                records.append((path, row))
                seen.add(path)
                break
    return records


def make_contact_sheet_pages(
    frame_df: pd.DataFrame,
    output_dir: Path,
    stem_prefix: str,
    max_images: int,
    figures: list[Path],
) -> None:
    records = overlay_records(frame_df)
    if not records:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "No overlay images found", ha="center", va="center")
        save_figure(fig, output_dir, f"{stem_prefix}_01", figures)
        return

    chunks = [records[i : i + max_images] for i in range(0, len(records), max_images)]
    for page_idx, chunk in enumerate(chunks, start=1):
        cols = min(4, len(chunk))
        rows = math.ceil(len(chunk) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.1, rows * 2.45))
        axes_arr = np.atleast_1d(axes).ravel()
        for ax, (path, row) in zip(axes_arr, chunk):
            img = Image.open(path).convert("RGB")
            img.thumbnail((360, 260))
            img = ImageOps.expand(img, border=8, fill="white")
            ax.imshow(img)
            ax.set_title(contact_sheet_title(row, path), fontsize=6.5, pad=2)
            ax.axis("off")
        for ax in axes_arr[len(chunk) :]:
            ax.axis("off")
        fig.suptitle(f"Annotation overlays - page {page_idx}/{len(chunks)}", y=1.01)
        save_figure(fig, output_dir, f"{stem_prefix}_{page_idx:02d}", figures)


def contact_sheet_title(row: pd.Series, path: Path) -> str:
    condition_id = str(row.get("condition_id", "") or "")
    etoh = _clean_number(row.get("EtOH_pct"))
    water = _clean_number(row.get("W_pct"))
    qc = _clean_number(row.get("Qc_uLh"))
    qaq = _clean_number(row.get("Qaq_uLh"))
    n_droplets = _clean_number(row.get("n_droplets"))
    frame_name = str(row.get("frame_name", "") or path.name)
    parts = [condition_id] if condition_id else []
    if np.isfinite(etoh) and np.isfinite(water):
        parts.append(f"{_fmt_int(etoh)}/{_fmt_int(water)}")
    if np.isfinite(qc) and np.isfinite(qaq):
        parts.append(f"{_fmt_int(qc)};{_fmt_int(qaq)} uL/h")
    n_text = f", n={_fmt_int(n_droplets)}" if np.isfinite(n_droplets) else ""
    return f"{' '.join(parts)}\n{frame_name}{n_text}"


def bundle_pdf(output_dir: Path, figure_pdfs: list[Path]) -> None:
    bundle = output_dir / "publication_figures_python_summary.pdf"
    with PdfPages(bundle) as pdf:
        for fig_pdf in figure_pdfs:
            png = fig_pdf.with_suffix(".png")
            fig, ax = plt.subplots(figsize=(8.27, 11.69))
            ax.axis("off")
            if png.exists():
                img = Image.open(png).convert("RGB")
                ax.imshow(img)
                ax.set_title(fig_pdf.stem, fontsize=12, pad=10)
            else:
                ax.text(0.5, 0.5, fig_pdf.name, ha="center", va="center", fontsize=18)
            pdf.savefig(fig)
            plt.close(fig)


def write_readme(output_dir: Path, preset: str = "parameter_sweep") -> None:
    (output_dir / "README_publication_figures.md").write_text(
        f"""# Publication figures (YOLO compiled version)

Reproducible generation:

```bash
streamlit run apps/publication_compiler/compile_droplet_publication_app.py
```

Analysis preset: `{preset}` - {ANALYSIS_PRESETS.get(preset, "")}

Main outputs follow the same names as the manual publication folder:

- `condition_stats_publication.csv`: clean table of plotted values, including `median_ci_low` / `median_ci_high`.
- `compiled_data/droplet_measurements_manual.csv`: YOLO measurements converted to the manual-analysis column layout.
- `compiled_data/frame_summary_manual.csv`: frame-level summary converted to the manual-analysis column layout.
- `fig00_method_qc_summary.*`: annotation, coverage and QC summary.
- `fig01_size_vs_flow.*` to `fig10b_annotation_coverage_qaq.*`: publication-style parameter figures as PNG/PDF.
- `fig11_manual_overlay_contact_sheet_*.pdf`: paginated overlay contact sheets.
- `repeatability_run_stats.csv`, `repeatability_frame_stats.csv`, `repeatability_pairwise_stats.csv`: only generated by the repeatability preset.
- `parameter_sweep_condition_stats.csv`, `parameter_sweep_frame_stats.csv`, `parameter_sweep_model_stats.csv`, `parameter_sweep_pairwise_stats.csv`: only generated by the parameter-sweep preset.

Publication status criterion:

- `usable`: at least 10 detected droplets and pile-up fraction < 0.6.
- `to_confirm`: 5 to 9 droplets, or majority pile-up.
- `excluded`: fewer than 5 droplets retained or condition manually marked bad in the UI.

Absolute sizes use the calibration already present in the YOLO `droplet_measurements.csv` files.
""",
        encoding="utf-8",
    )


def run_streamlit() -> None:
    import streamlit as st

    st.set_page_config(page_title="Droplet publication compiler", layout="wide")
    st.title("Droplet publication compiler")

    root = Path(
        st.text_input(
            "Dossier racine contenant les sous-dossiers avec droplets_analysis",
            value=str(DEFAULT_ROOT),
        )
    )
    output_dir = Path(
        st.text_input(
            "Dossier de sortie",
            value=str(root / "figures_publication_yolo" / "en"),
        )
    )
    metadata_path = Path(
        st.text_input(
            "CSV de métadonnées (sauvegarde/chargement)",
            value=str(root / "droplet_experiment_metadata.csv"),
        )
    )
    preset = st.selectbox(
        "Preset de statistiques",
        options=list(ANALYSIS_PRESETS),
        format_func=lambda key: f"{key} - {ANALYSIS_PRESETS[key]}",
        index=0,
    )

    col_a, col_b, col_c = st.columns(3)
    load_existing = col_a.button("Charger le CSV de métadonnées", use_container_width=True)
    scan = col_b.button("Scanner le dossier racine", use_container_width=True)
    save_meta = col_c.button("Sauvegarder les métadonnées", use_container_width=True)

    if "metadata" not in st.session_state or scan:
        folders = discover_analysis_folders(root)
        st.session_state.metadata = pd.DataFrame([infer_metadata(folder) for folder in folders])
    if load_existing and metadata_path.exists():
        st.session_state.metadata = read_metadata_csv(metadata_path)
    if save_meta and "metadata" in st.session_state:
        st.session_state.metadata.to_csv(metadata_path, index=False)
        st.success(f"Métadonnées sauvegardées: {metadata_path}")

    metadata = st.session_state.get("metadata", pd.DataFrame())
    if metadata.empty:
        st.warning("Aucun sous-dossier droplets_analysis trouvé.")
        return

    st.caption(
        "Renseigne ici ce que chaque dossier représente: série expérimentale, flow rates, fraction volumique, notes QC."
    )
    st.info(
        "Le tableau est dans un formulaire: modifie plusieurs cellules, puis clique sur "
        "`Appliquer les modifications`. Cela évite le rechargement qui faisait perdre la dernière valeur saisie."
    )
    with st.form("metadata_editor", clear_on_submit=False):
        edited = st.data_editor(
            metadata,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "analysis_dir": st.column_config.TextColumn(disabled=True),
                "include": st.column_config.CheckboxColumn(default=True),
                "manual_bad": st.column_config.CheckboxColumn(default=False),
                "Qc_uLh": st.column_config.NumberColumn(format="%.0f"),
                "Qaq_uLh": st.column_config.NumberColumn(format="%.0f"),
                "EtOH_pct": st.column_config.NumberColumn(format="%.0f"),
                "W_pct": st.column_config.NumberColumn(format="%.0f"),
            },
            key="metadata_editor_table",
        )
        apply_edits = st.form_submit_button("Appliquer les modifications", type="primary")
    if apply_edits:
        st.session_state.metadata = edited
        edited.to_csv(metadata_path, index=False)
        st.success(f"Modifications appliquées et sauvegardées: {metadata_path}")
    current_metadata = st.session_state.get("metadata", edited)

    col_compile, col_preview = st.columns([1, 2])
    compile_now = col_compile.button("Compiler les CSV et figures", type="primary", use_container_width=True)
    if col_preview.checkbox("Prévisualiser le résumé après application", value=True):
        try:
            _, _, preview_condition_df = build_tables(current_metadata)
            col_preview.dataframe(preview_condition_df, use_container_width=True)
        except Exception as exc:
            col_preview.warning(f"Prévisualisation impossible: {exc}")

    if compile_now:
        droplet_df, frame_df, condition_df = build_tables(current_metadata)
        figures = export_all(droplet_df, frame_df, condition_df, output_dir, preset=preset)
        current_metadata.to_csv(metadata_path, index=False)
        st.success(f"Export terminé dans: {output_dir}")
        st.write(
            {
                "conditions": len(condition_df),
                "frames": len(frame_df),
                "droplets": len(droplet_df),
                "figures_pdf": len(figures),
                "preset": preset,
            }
        )
        st.dataframe(condition_df, use_container_width=True)


def run_cli(args: argparse.Namespace) -> None:
    root = Path(args.root)
    if args.metadata and Path(args.metadata).exists():
        metadata = read_metadata_csv(Path(args.metadata))
    else:
        metadata = pd.DataFrame([infer_metadata(folder) for folder in discover_analysis_folders(root)])
        if args.metadata:
            metadata.to_csv(args.metadata, index=False)
            print(f"Metadata template written to {args.metadata}")
            print("Edit it, then rerun with the same --metadata path.")
            return
    droplet_df, frame_df, condition_df = build_tables(metadata)
    figures = export_all(droplet_df, frame_df, condition_df, Path(args.output), preset=args.preset)
    print(f"Exported {len(condition_df)} conditions, {len(frame_df)} frames, {len(droplet_df)} droplets")
    print(f"Output: {args.output}")
    print(f"Figure PDFs: {len(figures)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--output", default="")
    parser.add_argument("--metadata", default="")
    parser.add_argument("--preset", choices=list(ANALYSIS_PRESETS), default="parameter_sweep")
    parser.add_argument("--no-ui", action="store_true")
    args = parser.parse_args()
    if not args.output:
        args.output = str(Path(args.root) / "figures_publication_yolo" / "en")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.no_ui:
        run_cli(args)
    else:
        run_streamlit()
