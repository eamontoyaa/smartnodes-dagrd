"""Build the static GitHub Pages MVP for SmartNodes DAGRD.

This script is intentionally self-contained: every time new readings are committed
under data/raw/**, GitHub Actions can run this file, regenerate the figures, and
publish the updated site directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sn_dagrd.drift_correction import correct_sensor_angles_moving_df
from sn_dagrd.inclinometer_io import read_inclinometer_jsonl, sample_inclinometer_df
from sn_dagrd.inclinometer_kinematics import compute_inclinometer_displacements
from sn_dagrd.inclinometer_plotting import (
    plot_angle_correction_over_time,
    plot_cumulative_displacement_profiles,
    plot_sensor_displacement_evolution,
    plot_single_displacement_profile,
)

ROOT = Path(__file__).resolve().parents[1]
SITE_DIR = ROOT / "site"
FIGURES_DIR = SITE_DIR / "figures"
TABLES_DIR = SITE_DIR / "tables"
ASSETS_DIR = SITE_DIR / "assets"


@dataclass(frozen=True)
class StationConfig:
    slug: str
    name: str
    code: str
    station_id: str
    raw_path: Path
    valid_sensors: Mapping[str, int]
    depths_m: np.ndarray
    azimuth_deg: float
    start_date: str
    sample_freq: str
    sensor_to_plot: str


COMMON_CORRECTION_PARAMS = dict(
    window_time="1D",
    method="median",
    jump_threshold=10.0,
    min_jump_distance_time="1D",
    max_jump_gap=1,
    anchor_points=4,
)

STATIONS = [
    StationConfig(
        slug="santa_rita_1",
        name="Santa Rita 1",
        code="SR-SN-01",
        station_id="SmartNode-SantaRita1",
        raw_path=ROOT / "data" / "raw" / "santa_rita_1" / "raw_InclinometroSantaRita1.txt",
        valid_sensors={f"{i}a": 1 for i in range(1, 16)},
        depths_m=np.linspace(1.0, 15.0, num=15),
        azimuth_deg=30.0,
        start_date="2026-05-01",
        sample_freq="6h",
        sensor_to_plot="1a",
    ),
    StationConfig(
        slug="la_palmera",
        name="La Palmera",
        code="VLP",
        station_id="SmartNode-LaPalmera",
        raw_path=ROOT / "data" / "raw" / "la_palmera" / "raw_InclinometroLaPalmera.txt",
        valid_sensors={
            "1a": 1,
            "2a": 0,
            "3a": 1,
            "4a": 1,
            "5a": 1,
            "6a": 1,
            "7a": 1,
            "8a": 0,
            "9a": 1,
            "10a": 1,
            "11a": 1,
            "12a": 1,
            "13a": 1,
            "14a": 1,
            "15a": 1,
        },
        depths_m=np.linspace(-0.25, 14.825, num=15),
        azimuth_deg=295.0,
        start_date="2026-04-06",
        sample_freq="6h",
        sensor_to_plot="3a",
    ),
]


def sensor_sort_key(sensor_id: str) -> int:
    text = str(sensor_id)
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else 10_000


def reset_output_dirs() -> None:
    for directory in [FIGURES_DIR, TABLES_DIR, ASSETS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    # Keep the directories but remove stale generated files.
    for pattern in ["**/*.png", "**/*.csv", "**/*.json", "*.html", "assets/*.css"]:
        for path in SITE_DIR.glob(pattern):
            if path.is_file():
                path.unlink()


def save_figure(fig: plt.Figure, path: Path, *, dpi: int = 180) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path.relative_to(SITE_DIR).as_posix()


def date_label(value) -> str:
    if pd.isna(value):
        return "—"
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")


def number_label(value, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "—"
    if float(value).is_integer():
        return f"{int(value):,}".replace(",", ".")
    return f"{float(value):,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def compute_availability(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["has_quaternion"] = data[["qr", "qi", "qj", "qk"]].notna().all(axis=1)
    availability = (
        data.groupby("sensor_id", dropna=False)
        .agg(
            n_rows=("timestamp", "size"),
            n_valid_quaternion=("has_quaternion", "sum"),
            first_timestamp=("timestamp", "min"),
            last_timestamp=("timestamp", "max"),
        )
        .reset_index()
    )
    availability["availability_pct"] = (
        availability["n_valid_quaternion"] / availability["n_rows"].replace(0, np.nan) * 100.0
    )
    availability = availability.sort_values(
        "sensor_id", key=lambda s: s.astype(str).map(sensor_sort_key)
    ).reset_index(drop=True)
    return availability


def plot_availability(availability: pd.DataFrame, cfg: StationConfig) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
    y = np.arange(len(availability))
    values = availability["availability_pct"].to_numpy(dtype=float)
    colors = plt.get_cmap("viridis")(np.clip(values / 100.0, 0.0, 1.0))
    ax.barh(y, values, color=colors, edgecolor="0.25", linewidth=0.4)
    ax.set_yticks(y, availability["sensor_id"].astype(str))
    ax.set_xlim(0, 100)
    ax.set_xlabel("Lecturas válidas [%]")
    ax.set_ylabel("Sensor")
    ax.set_title(f"{cfg.name}\nDisponibilidad de quaternion por sensor")
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.invert_yaxis()
    for i, value in enumerate(values):
        label = "—" if np.isnan(value) else f"{value:.1f}%"
        ax.text(min(value + 1.0, 99.0), i, label, va="center", fontsize=8)
    return fig


def plot_rssi(df_records: pd.DataFrame, cfg: StationConfig) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 3.8), layout="constrained")
    data = df_records.dropna(subset=["rssi"]).copy()
    if data.empty:
        ax.text(0.5, 0.5, "Sin datos RSSI", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.plot(data["timestamp"], data["rssi"], marker="o", markersize=2.0, linewidth=1.0)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
        fig.autofmt_xdate()
    ax.set_title(f"{cfg.name}\nIntensidad de señal RSSI")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("RSSI [dBm]")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    return fig


def read_records_summary(path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
            rows.append(
                {
                    "timestamp": pd.to_datetime(record.get("timestamp")),
                    "estacion_id": record.get("estacion_id"),
                    "uptime_s": record.get("uptime_s"),
                    "heap": record.get("heap"),
                    "rssi": record.get("rssi"),
                    "n_sensors_payload": len(record.get("data") or []),
                }
            )
    return pd.DataFrame(rows)


def latest_displacement_table(df_disp_corr: pd.DataFrame) -> pd.DataFrame:
    latest_ts = df_disp_corr["timestamp"].max()
    latest = (
        df_disp_corr.loc[df_disp_corr["timestamp"] == latest_ts]
        .sort_values("sensor_id", key=lambda s: s.astype(str).map(sensor_sort_key))
        .copy()
    )
    latest["resultant_mm"] = np.sqrt(
        latest["cum_disp_north_mm"].to_numpy(dtype=float) ** 2
        + latest["cum_disp_east_mm"].to_numpy(dtype=float) ** 2
    )
    return latest[
        [
            "timestamp",
            "sensor_id",
            "depth_m",
            "cum_disp_north_mm",
            "cum_disp_east_mm",
            "resultant_mm",
        ]
    ]


def process_station(cfg: StationConfig) -> dict:
    print(f"Building station: {cfg.name}")

    station_fig_dir = FIGURES_DIR / cfg.slug
    station_table_dir = TABLES_DIR / cfg.slug
    station_fig_dir.mkdir(parents=True, exist_ok=True)
    station_table_dir.mkdir(parents=True, exist_ok=True)

    df_records = read_records_summary(cfg.raw_path)

    df_raw = read_inclinometer_jsonl(
        path=cfg.raw_path,
        sensor_type="BNO",
        drop_duplicate_rows=True,
    )
    df_raw = (
        df_raw.loc[df_raw["timestamp"] >= pd.Timestamp(cfg.start_date)]
        .reset_index(drop=True)
    )
    if df_raw.empty:
        raise ValueError(f"{cfg.name}: no readings after start_date={cfg.start_date!r}.")

    df_sampled = sample_inclinometer_df(
        df_raw,
        freq=cfg.sample_freq,
        how="first",
    )

    availability = compute_availability(df_sampled)
    availability.to_csv(station_table_dir / "availability.csv", index=False)

    df_corr = correct_sensor_angles_moving_df(
        df=df_sampled,
        timestamp_col="timestamp",
        sensor_col="sensor_id",
        station_col="estacion_id",
        angle_cols=("roll_deg", "pitch_deg", "yaw_deg"),
        **COMMON_CORRECTION_PARAMS,
    )

    df_disp_raw = compute_inclinometer_displacements(
        df=df_corr,
        depths_m=cfg.depths_m,
        azimuth_deg=cfg.azimuth_deg,
        pitch_col="pitch_deg",
        roll_col="roll_deg",
        timestamp_col="timestamp",
        sensor_col="sensor_id",
        station_col="estacion_id",
        valid_sensors=cfg.valid_sensors,
        reference_sensor="deepest",
    )

    df_disp_corr = compute_inclinometer_displacements(
        df=df_corr,
        depths_m=cfg.depths_m,
        azimuth_deg=cfg.azimuth_deg,
        pitch_col="pitch_corr_deg",
        roll_col="roll_corr_deg",
        timestamp_col="timestamp",
        sensor_col="sensor_id",
        station_col="estacion_id",
        valid_sensors=cfg.valid_sensors,
        reference_sensor="deepest",
    )

    latest_disp = latest_displacement_table(df_disp_corr)
    latest_disp.to_csv(station_table_dir / "latest_displacement.csv", index=False)

    latest_resultant = float(latest_disp["resultant_mm"].max())
    first_ts = df_sampled["timestamp"].min()
    last_ts = df_sampled["timestamp"].max()
    latest_record = df_records.sort_values("timestamp").iloc[-1].to_dict() if not df_records.empty else {}

    figure_paths: dict[str, str] = {}

    figure_paths["availability"] = save_figure(
        plot_availability(availability, cfg),
        station_fig_dir / "availability.png",
    )

    figure_paths["rssi"] = save_figure(
        plot_rssi(df_records, cfg),
        station_fig_dir / "rssi.png",
    )

    figure_paths["pitch_correction"] = save_figure(
        plot_angle_correction_over_time(
            df_corr=df_corr,
            field="pitch",
            depths_m=cfg.depths_m,
            valid_sensors=cfg.valid_sensors,
            profile_freq=None,
            profile_selection="nearest",
            delta_angle=0.2,
            title=f"{cfg.name}\nPitch original vs corregido",
            cmap="viridis",
        ),
        station_fig_dir / "pitch_correction_over_time.png",
    )

    figure_paths["roll_correction"] = save_figure(
        plot_angle_correction_over_time(
            df_corr=df_corr,
            field="roll",
            depths_m=cfg.depths_m,
            valid_sensors=cfg.valid_sensors,
            profile_freq=None,
            profile_selection="nearest",
            delta_angle=0.2,
            title=f"{cfg.name}\nRoll original vs corregido",
            cmap="viridis",
        ),
        station_fig_dir / "roll_correction_over_time.png",
    )

    figure_paths["profiles_raw"] = save_figure(
        plot_cumulative_displacement_profiles(
            df_disp=df_disp_raw,
            depths_m=cfg.depths_m,
            valid_sensors=cfg.valid_sensors,
            components="ne",
            profile_freq="7D",
            profile_selection="nearest",
            xlims=(-30, 30),
            title=f"{cfg.name}\nDesplazamientos acumulados sin corrección N-E",
            cmap="viridis",
        ),
        station_fig_dir / "cumulative_displacement_profiles_raw.png",
    )

    figure_paths["profiles_corrected"] = save_figure(
        plot_cumulative_displacement_profiles(
            df_disp=df_disp_corr,
            depths_m=cfg.depths_m,
            valid_sensors=cfg.valid_sensors,
            components="ne",
            profile_freq="7D",
            profile_selection="nearest",
            xlims=(-30, 30),
            title=f"{cfg.name}\nDesplazamientos acumulados corregidos N-E",
            cmap="viridis",
        ),
        station_fig_dir / "cumulative_displacement_profiles_corr.png",
    )

    figure_paths["latest_profile"] = save_figure(
        plot_single_displacement_profile(
            df_disp=df_disp_corr,
            timestamp=last_ts,
            depths_m=cfg.depths_m,
            valid_sensors=cfg.valid_sensors,
            components="ne",
            timestamp_col="timestamp",
            sensor_col="sensor_id",
            station_col="estacion_id",
            station_id=cfg.station_id,
            xlims=(-30, 30),
            title=f"{cfg.name}\nPerfil corregido más reciente",
        ),
        station_fig_dir / "latest_corrected_profile.png",
    )

    figure_paths["sensor_evolution"] = save_figure(
        plot_sensor_displacement_evolution(
            df_disp=df_disp_corr,
            sensor=cfg.sensor_to_plot,
            depths_m=cfg.depths_m,
            valid_sensors=cfg.valid_sensors,
            components="ne",
            profile_freq="1D",
            profile_selection="nearest",
            ylims=(-30, 30),
            xlims_ab=(-30, 30),
            title=f"{cfg.name}\nDesplazamientos acumulados - Sensor {cfg.sensor_to_plot}",
            cmap="viridis",
        ),
        station_fig_dir / f"displacement_evolution_sensor_{cfg.sensor_to_plot}.png",
    )

    payload = {
        "slug": cfg.slug,
        "name": cfg.name,
        "code": cfg.code,
        "station_id": cfg.station_id,
        "azimuth_deg": cfg.azimuth_deg,
        "start_date": cfg.start_date,
        "sample_freq": cfg.sample_freq,
        "sensor_to_plot": cfg.sensor_to_plot,
        "n_raw_rows": int(len(df_raw)),
        "n_sampled_rows": int(len(df_sampled)),
        "n_payload_records": int(len(df_records)),
        "n_declared_sensors": int(len(cfg.valid_sensors)),
        "n_valid_sensors": int(sum(bool(v) for v in cfg.valid_sensors.values())),
        "first_timestamp": first_ts.isoformat(),
        "last_timestamp": last_ts.isoformat(),
        "latest_rssi": latest_record.get("rssi"),
        "latest_heap": latest_record.get("heap"),
        "latest_uptime_s": latest_record.get("uptime_s"),
        "latest_resultant_mm": latest_resultant,
        "figures": figure_paths,
        "tables": {
            "availability": (station_table_dir / "availability.csv").relative_to(SITE_DIR).as_posix(),
            "latest_displacement": (station_table_dir / "latest_displacement.csv").relative_to(SITE_DIR).as_posix(),
        },
    }
    (station_table_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def render_metric(label: str, value: str, hint: str | None = None) -> str:
    hint_html = f"<small>{escape(hint)}</small>" if hint else ""
    return f"""
    <div class=\"metric\">
      <span>{escape(label)}</span>
      <strong>{escape(value)}</strong>
      {hint_html}
    </div>
    """


def render_figure(src: str, title: str, caption: str) -> str:
    return f"""
    <figure class=\"figure-card\">
      <a href=\"{escape(src)}\" target=\"_blank\" rel=\"noreferrer\">
        <img src=\"{escape(src)}\" alt=\"{escape(title)}\" loading=\"lazy\">
      </a>
      <figcaption>
        <strong>{escape(title)}</strong>
        <span>{escape(caption)}</span>
      </figcaption>
    </figure>
    """


def render_station_section(summary: dict) -> str:
    figures = summary["figures"]
    latest_rssi = summary.get("latest_rssi")
    latest_rssi_label = "—" if latest_rssi is None or pd.isna(latest_rssi) else f"{latest_rssi} dBm"

    metrics = "".join(
        [
            render_metric("Periodo", f"{date_label(summary['first_timestamp'])} → {date_label(summary['last_timestamp'])}"),
            render_metric("Sensores válidos", f"{summary['n_valid_sensors']} / {summary['n_declared_sensors']}", "perfil declarado"),
            render_metric("Lecturas procesadas", number_label(summary["n_sampled_rows"], 0), f"submuestreo {summary['sample_freq']}"),
            render_metric("RSSI reciente", latest_rssi_label),
            render_metric("Desp. resultante máx.", f"{summary['latest_resultant_mm']:.2f} mm", "último perfil corregido"),
            render_metric("Azimut eje A", f"{summary['azimuth_deg']:.1f}°"),
        ]
    )

    figure_grid = "".join(
        [
            render_figure(
                figures["latest_profile"],
                "Perfil corregido más reciente",
                "Desplazamientos acumulados Norte-Este con corrección de deriva.",
            ),
            render_figure(
                figures["profiles_corrected"],
                "Evolución de perfiles corregidos",
                "Perfiles acumulados seleccionados cada 7 días.",
            ),
            render_figure(
                figures["pitch_correction"],
                "Pitch original vs corregido",
                "Comparación por sensor luego de remover la deriva lenta.",
            ),
            render_figure(
                figures["roll_correction"],
                "Roll original vs corregido",
                "Comparación por sensor luego de remover la deriva lenta.",
            ),
            render_figure(
                figures["sensor_evolution"],
                f"Evolución sensor {summary['sensor_to_plot']}",
                "Trayectoria temporal del desplazamiento acumulado corregido.",
            ),
            render_figure(
                figures["availability"],
                "Disponibilidad de datos",
                "Porcentaje de lecturas con quaternion completo en el periodo procesado.",
            ),
            render_figure(
                figures["rssi"],
                "Señal RSSI",
                "Seguimiento básico de comunicación del nodo.",
            ),
            render_figure(
                figures["profiles_raw"],
                "Perfiles sin corrección",
                "Referencia para comparar el efecto de la corrección aplicada.",
            ),
        ]
    )

    return f"""
    <section class=\"station-section\" id=\"{escape(summary['slug'])}\">
      <div class=\"station-heading\">
        <div>
          <p class=\"eyebrow\">{escape(summary['code'])} · {escape(summary['station_id'])}</p>
          <h2>{escape(summary['name'])}</h2>
        </div>
        <div class=\"station-actions\">
          <a href=\"{escape(summary['tables']['availability'])}\">Disponibilidad CSV</a>
          <a href=\"{escape(summary['tables']['latest_displacement'])}\">Último perfil CSV</a>
        </div>
      </div>
      <div class=\"metrics-grid\">{metrics}</div>
      <div class=\"figures-grid\">{figure_grid}</div>
    </section>
    """


def build_css() -> str:
    return """
    :root {
      --bg: #f4f7f6;
      --panel: #ffffff;
      --ink: #12201d;
      --muted: #5e716c;
      --brand: #0f766e;
      --brand-dark: #12453f;
      --brand-soft: #d9f4ee;
      --line: #d8e4e0;
      --shadow: 0 18px 55px rgba(16, 51, 47, 0.12);
    }

    * { box-sizing: border-box; }

    html { scroll-behavior: smooth; }

    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.16), transparent 32rem),
        linear-gradient(180deg, #f8fbfa 0%, var(--bg) 100%);
      color: var(--ink);
      line-height: 1.55;
    }

    a { color: inherit; }

    .hero {
      padding: 4.5rem 1.25rem 2.2rem;
      background: linear-gradient(135deg, var(--brand-dark), #14372f 62%, #20372d);
      color: white;
      border-bottom-left-radius: 2rem;
      border-bottom-right-radius: 2rem;
    }

    .hero-inner, main, .footer-inner {
      max-width: 1180px;
      margin: 0 auto;
    }

    .hero h1 {
      max-width: 820px;
      margin: 0.5rem 0 1rem;
      font-size: clamp(2.2rem, 5vw, 4.8rem);
      line-height: 0.95;
      letter-spacing: -0.06em;
    }

    .hero p {
      max-width: 780px;
      margin: 0;
      font-size: 1.1rem;
      color: rgba(255,255,255,0.82);
    }

    .eyebrow {
      margin: 0 0 0.5rem;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      font-size: 0.78rem;
      font-weight: 800;
      color: #93f1df;
    }

    .nav-pills {
      display: flex;
      flex-wrap: wrap;
      gap: 0.7rem;
      margin-top: 1.8rem;
    }

    .nav-pills a, .station-actions a {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 0.7rem 1rem;
      background: rgba(255,255,255,0.12);
      color: white;
      text-decoration: none;
      border: 1px solid rgba(255,255,255,0.2);
      font-weight: 700;
      font-size: 0.92rem;
    }

    main { padding: 2rem 1.25rem 4rem; }

    .intro-card, .station-section {
      background: rgba(255,255,255,0.88);
      border: 1px solid var(--line);
      border-radius: 1.5rem;
      box-shadow: var(--shadow);
      margin-bottom: 2rem;
    }

    .intro-card {
      padding: 1.4rem;
      display: grid;
      grid-template-columns: minmax(0, 1.5fr) minmax(260px, 0.8fr);
      gap: 1.4rem;
    }

    .intro-card h2, .station-heading h2 {
      margin: 0;
      font-size: clamp(1.7rem, 3vw, 2.4rem);
      letter-spacing: -0.04em;
    }

    .intro-card p { margin: 0.5rem 0 0; color: var(--muted); }

    .workflow-box {
      background: var(--brand-soft);
      border: 1px solid #b9e8df;
      border-radius: 1rem;
      padding: 1rem;
      color: var(--brand-dark);
      font-weight: 700;
    }

    .station-section { padding: 1.4rem; scroll-margin-top: 1rem; }

    .station-heading {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 1rem;
      margin-bottom: 1.2rem;
    }

    .station-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 0.6rem;
      justify-content: flex-end;
    }

    .station-actions a {
      background: var(--brand-dark);
      border-color: var(--brand-dark);
    }

    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 0.75rem;
      margin-bottom: 1.2rem;
    }

    .metric {
      border: 1px solid var(--line);
      border-radius: 1rem;
      padding: 0.85rem;
      background: #fbfefd;
      min-height: 6rem;
    }

    .metric span, .metric small {
      display: block;
      color: var(--muted);
      font-size: 0.78rem;
      font-weight: 700;
    }

    .metric strong {
      display: block;
      margin-top: 0.35rem;
      font-size: 1.05rem;
      letter-spacing: -0.02em;
    }

    .metric small { margin-top: 0.25rem; font-weight: 600; }

    .figures-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 1rem;
    }

    .figure-card {
      margin: 0;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 1.1rem;
      background: var(--panel);
    }

    .figure-card a {
      display: block;
      background: #f8faf9;
    }

    .figure-card img {
      width: 100%;
      display: block;
      aspect-ratio: 4 / 3;
      object-fit: contain;
      padding: 0.4rem;
    }

    .figure-card figcaption {
      padding: 0.9rem 1rem 1rem;
      border-top: 1px solid var(--line);
    }

    .figure-card figcaption strong,
    .figure-card figcaption span {
      display: block;
    }

    .figure-card figcaption span {
      margin-top: 0.2rem;
      color: var(--muted);
      font-size: 0.9rem;
    }

    footer {
      padding: 2rem 1.25rem 3rem;
      color: var(--muted);
    }

    code {
      background: #eaf3f0;
      padding: 0.1rem 0.35rem;
      border-radius: 0.35rem;
    }

    @media (max-width: 980px) {
      .intro-card, .figures-grid { grid-template-columns: 1fr; }
      .metrics-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
      .station-heading { flex-direction: column; }
      .station-actions { justify-content: flex-start; }
    }

    @media (max-width: 620px) {
      .hero { padding-top: 3.2rem; }
      main { padding-inline: 0.8rem; }
      .metrics-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .station-section, .intro-card { padding: 1rem; border-radius: 1.1rem; }
    }
    """


def build_index(summaries: list[dict]) -> None:
    build_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    nav = "".join(
        f"<a href=\"#{escape(item['slug'])}\">{escape(item['name'])}</a>" for item in summaries
    )
    total_records = sum(item["n_payload_records"] for item in summaries)
    total_sampled = sum(item["n_sampled_rows"] for item in summaries)

    station_sections = "\n".join(render_station_section(item) for item in summaries)

    html = f"""<!doctype html>
<html lang=\"es\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>SmartNodes DAGRD · MVP</title>
  <meta name=\"description\" content=\"MVP estático para visualizar lecturas de inclinómetros remotos SmartNode/DAGRD.\">
  <link rel=\"stylesheet\" href=\"assets/styles.css\">
</head>
<body>
  <header class=\"hero\">
    <div class=\"hero-inner\">
      <p class=\"eyebrow\">MVP · Inclinómetros remotos</p>
      <h1>SmartNodes DAGRD</h1>
      <p>
        Reporte web estático generado automáticamente desde las lecturas JSONL del repositorio.
        En cada commit a <code>main</code>, GitHub Actions reprocesa los datos, actualiza las figuras
        y publica esta página en GitHub Pages.
      </p>
      <nav class=\"nav-pills\" aria-label=\"Estaciones\">{nav}</nav>
    </div>
  </header>

  <main>
    <section class=\"intro-card\">
      <div>
        <p class=\"eyebrow\">Resumen del build</p>
        <h2>Monitoreo reproducible, sin backend por ahora.</h2>
        <p>
          El sitio publica figuras PNG y tablas CSV generadas con el paquete Python del repositorio.
          La página es intencionalmente simple: ideal para validar el flujo de datos, procesamiento,
          corrección de deriva y despliegue antes de evolucionar a un dashboard interactivo.
        </p>
      </div>
      <div class=\"workflow-box\">
        Datos crudos → lectura JSONL → ángulos Euler → corrección de deriva por mediana móvil → desplazamientos acumulados → figuras → GitHub Pages
        <br><br>
        Build: {escape(build_time)}<br>
        Registros crudos: {number_label(total_records, 0)}<br>
        Filas procesadas: {number_label(total_sampled, 0)}
      </div>
    </section>

    {station_sections}
  </main>

  <footer>
    <div class=\"footer-inner\">
      <p>
        Generado por <code>scripts/build_site.py</code>. Para actualizar el reporte, agrega nuevas lecturas
        en <code>data/raw/</code> y haz commit a la rama <code>main</code>.
      </p>
    </div>
  </footer>
</body>
</html>
"""
    (SITE_DIR / "index.html").write_text(html, encoding="utf-8")
    (ASSETS_DIR / "styles.css").write_text(build_css(), encoding="utf-8")
    (TABLES_DIR / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    (SITE_DIR / ".nojekyll").write_text("", encoding="utf-8")


def main() -> None:
    reset_output_dirs()
    summaries = [process_station(cfg) for cfg in STATIONS]
    build_index(summaries)
    print(f"Site built at: {SITE_DIR}")


if __name__ == "__main__":
    main()
