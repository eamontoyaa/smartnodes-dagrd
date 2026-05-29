"""
inclinometer_drift_correction.py

Utilities to correct slow instrumental drift in inclinometer time series while
preserving abrupt jumps detected automatically with a robust MAD-based detector.

Expected input format
---------------------
A pandas DataFrame with at least two columns:

    timestamp | angle_observed

Example
-------
from inclinometer_drift_correction import (
    correct_drift_by_segments_moving_df,
    plot_drift_correction_result,
    plot_jump_score,
)

result, metadata = correct_drift_by_segments_moving_df(
    df=df,
    timestamp_col="timestamp",
    value_col="angle_observed",
    window_time="7D",
    method="median",
    jump_threshold=7.0,
    min_jump_distance_time="1D",
    max_jump_gap=1,
    anchor_points=12,
)

plot_drift_correction_result(result, metadata)
"""

from __future__ import annotations

from typing import Any, Literal
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

MovingMethod = Literal["median", "mean"]


def _get_ssa_class(SSAClass=None):
    """
    Return the SSA class.

    The dependency `py_ssa_lib` is intentionally imported lazily so that the
    moving-median correction can be used even when SSA is not installed.
    """

    if SSAClass is not None:
        return SSAClass

    try:
        from py_ssa_lib.SSA import SSA as ImportedSSA
    except ModuleNotFoundError as exc:
        raise ImportError(
            "py_ssa_lib is required only for SSA-based correction. "
            "Install it or pass SSAClass explicitly."
        ) from exc

    return ImportedSSA



__all__ = [
    "generate_drifted_inc_data",
    "detect_jump_regions_mad",
    "smooth_segment_moving",
    "robust_segment_anchor",
    "correct_drift_by_segments_moving_df",
    "correct_drift_by_segments_ssa_df",
    "correct_one_segment_with_ssa",
    "plot_drift_correction_result",
    "plot_jump_score",
]


def generate_drifted_inc_data(
    *,
    start_time="2026-01-01 00:00:00",
    delta_time="10min",
    total_time="30D",
    angle0=0.0,
    noise_std=0.05,
    drift_type="linear",
    drift_magnitude=1.0,
    drift_power=2.0,
    jump_magnitude=2.0,
    jump_position=0.5,
    jump_permanent=True,
    seed=None,
    series_name="S1_pitch",
):
    """
    Genera una serie temporal sintética de ángulo de inclinación.

    Parameters
    ----------
    start_time : str
        Fecha inicial de la serie.

    delta_time : str
        Paso temporal. Ejemplos: "1min", "10min", "1H".

    total_time : str
        Tiempo total simulado. Ejemplos: "7D", "30D", "12H".

    angle0 : float
        Ángulo inicial.

    noise_std : float
        Desviación estándar del ruido instrumental gaussiano.

    drift_type : str
        Tipo de deriva instrumental:
        - "none"
        - "linear"
        - "concave_up"
        - "concave_down"

    drift_magnitude : float
        Magnitud total de la deriva al final de la serie.

    drift_power : float
        Controla la curvatura de las derivas cóncavas.
        Valores típicos: 2, 3, 4.

    jump_magnitude : float
        Magnitud del salto abrupto en el ángulo.

    jump_position : float
        Posición relativa del salto dentro de la serie.
        0.5 significa mitad de la serie.
        0.25 significa al 25% de la serie.

    jump_permanent : bool
        Si True, el salto permanece hasta el final.
        Si False, solo afecta una muestra puntual.

    seed : int or None
        Semilla aleatoria para reproducibilidad.

    series_name : str
        Nombre de la serie.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame largo con la serie simulada y sus componentes.
    """

    rng = np.random.default_rng(seed)

    start_time = pd.Timestamp(start_time)
    delta = pd.to_timedelta(delta_time)
    total = pd.to_timedelta(total_time)

    n = int(total / delta) + 1

    timestamps = pd.date_range(
        start=start_time,
        periods=n,
        freq=delta,
    )

    # Tiempo normalizado entre 0 y 1
    tau = np.linspace(0.0, 1.0, n)

    # -------------------------
    # Deriva instrumental
    # -------------------------
    if drift_type == "none":
        drift = np.zeros(n)

    elif drift_type == "linear":
        drift = drift_magnitude * tau

    elif drift_type == "concave_up":
        # Empieza suave y se acelera hacia el final
        drift = drift_magnitude * tau**drift_power

    elif drift_type == "concave_down":
        # Empieza rápido y luego se estabiliza
        drift = drift_magnitude * (1 - (1 - tau)**drift_power)

    else:
        raise ValueError(
            "drift_type debe ser uno de: "
            "'none', 'linear', 'concave_up', 'concave_down'"
        )

    # -------------------------
    # Salto abrupto
    # -------------------------
    jump = np.zeros(n)

    jump_idx = int(np.clip(jump_position, 0, 1) * (n - 1))

    if jump_permanent:
        jump[jump_idx:] = jump_magnitude
    else:
        jump[jump_idx] = jump_magnitude

    # -------------------------
    # Ruido instrumental
    # -------------------------
    noise = rng.normal(
        loc=0.0,
        scale=noise_std,
        size=n,
    )

    # -------------------------
    # Serie final
    # -------------------------
    angle_true = angle0 + drift + jump
    angle_observed = angle_true + noise

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "series": series_name,
            "t_index": np.arange(n),
            "t_normalized": tau,
            "angle_observed": angle_observed,
            "angle_true": angle_true,
            "drift": drift,
            "jump": jump,
            "noise": noise,
        }
    )

    return df


def _time_to_n_samples(time_value: str, dt_seconds: float) -> int:
    """
    Convert a time duration such as '1D', '12h', or '30min' into samples.
    """

    seconds = pd.to_timedelta(time_value).total_seconds()

    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"Invalid time duration: {time_value!r}")

    return max(1, int(round(seconds / dt_seconds)))


def _infer_dt_seconds(timestamps: pd.Series) -> float:
    """
    Infer the typical sampling interval in seconds using the median time step.
    """

    dt_seconds = timestamps.diff().dt.total_seconds().dropna().median()

    if dt_seconds <= 0 or not np.isfinite(dt_seconds):
        raise ValueError(
            "Could not infer a valid temporal spacing. "
            "Check that the timestamp column is valid and sorted."
        )

    return float(dt_seconds)


def robust_segment_anchor(y_segment: np.ndarray, n_anchor: int = 12) -> float:
    """
    Estimate the initial level of a stable segment using a robust median.
    """

    y_segment = np.asarray(y_segment, dtype=float)

    n = min(len(y_segment), int(n_anchor))

    if n <= 0:
        return 0.0

    return float(np.nanmedian(y_segment[:n]))


def smooth_segment_moving(
    y_segment: np.ndarray,
    window: int,
    method: MovingMethod = "median",
    min_periods: int | None = None,
) -> np.ndarray:
    """
    Estimate the slow trend of a stable segment using a moving mean or median.
    """

    y_segment = np.asarray(y_segment, dtype=float)

    n = len(y_segment)

    if n < 5:
        return y_segment.copy()

    window = int(window)
    window = min(window, n)

    # Prefer an odd window for centered rolling smoothing.
    if window % 2 == 0:
        window -= 1

    if window < 3:
        return y_segment.copy()

    if min_periods is None:
        min_periods = max(3, window // 5)

    s = pd.Series(y_segment)

    if method == "median":
        trend = (
            s.rolling(
                window=window,
                center=True,
                min_periods=min_periods,
            )
            .median()
        )

    elif method == "mean":
        trend = (
            s.rolling(
                window=window,
                center=True,
                min_periods=min_periods,
            )
            .mean()
        )

    else:
        raise ValueError("method must be 'median' or 'mean'.")

    trend = trend.interpolate(limit_direction="both")

    return trend.to_numpy(dtype=float)


def detect_jump_regions_mad(
    y,
    threshold=8.0,
    min_distance=10,
    max_gap=1,
    min_abs_change=0.05,
    robust_std_floor=0.005,
    return_score=False,
):
    y = np.asarray(y, dtype=float)
    n = len(y)

    if n < 3:
        if return_score:
            return [], np.array([])
        return []

    dy = np.diff(y)

    med = np.nanmedian(dy)
    mad = np.nanmedian(np.abs(dy - med))

    robust_std = 1.4826 * mad

    if robust_std == 0 or not np.isfinite(robust_std):
        robust_std = np.nanstd(dy)

    if robust_std == 0 or not np.isfinite(robust_std):
        robust_std = robust_std_floor

    robust_std = max(robust_std, robust_std_floor)

    score = np.abs(dy - med) / (robust_std + 1e-12)

    candidate_dy_indices = np.where(
        (score > threshold) &
        (np.abs(dy) >= min_abs_change)
    )[0]

    if len(candidate_dy_indices) == 0:
        if return_score:
            return [], score
        return []

    groups = []
    current_group = [candidate_dy_indices[0]]

    for idx in candidate_dy_indices[1:]:
        if idx - current_group[-1] <= max_gap:
            current_group.append(idx)
        else:
            groups.append(current_group)
            current_group = [idx]

    groups.append(current_group)

    raw_regions = []

    for group in groups:
        first_dy_idx = group[0]
        last_dy_idx = group[-1]

        start = first_dy_idx + 1
        end = last_dy_idx + 2

        start = max(1, min(start, n - 1))
        end = max(start + 1, min(end, n))

        raw_regions.append((start, end))

    regions = []

    for start, end in raw_regions:
        if len(regions) == 0:
            regions.append((start, end))
            continue

        prev_start, prev_end = regions[-1]

        if start - prev_end < min_distance:
            regions[-1] = (prev_start, max(prev_end, end))
        else:
            regions.append((start, end))

    if return_score:
        return regions, score

    return regions


def _build_stable_segments(
    n: int,
    jump_regions: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """
    Build stable segments from jump regions.

    Stable segments are:
        [0, jump_start_0), [jump_end_0, jump_start_1), ..., [last_jump_end, n)
    """

    stable_segments: list[tuple[int, int]] = []

    cursor = 0

    for jump_start, jump_end in jump_regions:
        if cursor < jump_start:
            stable_segments.append((cursor, jump_start))

        cursor = jump_end

    if cursor < n:
        stable_segments.append((cursor, n))

    return stable_segments


def correct_drift_by_segments_moving_df(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    value_col: str = "angle_observed",
    window: int | None = None,
    window_time: str | None = None,
    method: MovingMethod = "median",
    jump_threshold: float = 7.0,
    min_jump_distance: int | None = None,
    min_jump_distance_time: str | None = None,
    max_jump_gap: int = 1,
    max_jump_gap_time: str | None = None,
    anchor_points: int = 12,
    sort_by_time: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Correct slow instrumental drift while preserving automatically detected jumps.

    This function does not accept manual jump locations. Jump regions are
    detected automatically with `detect_jump_regions_mad`.
    """

    if timestamp_col not in df.columns:
        raise ValueError(f"timestamp_col={timestamp_col!r} not found in df.")

    if value_col not in df.columns:
        raise ValueError(f"value_col={value_col!r} not found in df.")

    data = df[[timestamp_col, value_col]].copy()
    data[timestamp_col] = pd.to_datetime(data[timestamp_col])
    data[value_col] = data[value_col].astype(float)

    if sort_by_time:
        data = data.sort_values(timestamp_col).reset_index(drop=True)
    else:
        data = data.reset_index(drop=True)

    y = data[value_col].to_numpy(dtype=float)
    n = len(y)

    if n < 5:
        raise ValueError("The time series is too short to correct drift.")

    dt_seconds = _infer_dt_seconds(data[timestamp_col])

    # Rolling smoothing window.
    if window is None:
        if window_time is None:
            raise ValueError("You must provide either window or window_time.")

        window = _time_to_n_samples(window_time, dt_seconds)

    window = int(window)

    if window < 3:
        raise ValueError("The rolling window must have at least 3 samples.")

    # Minimum distance between jump regions.
    if min_jump_distance is None:
        if min_jump_distance_time is not None:
            min_jump_distance = _time_to_n_samples(
                min_jump_distance_time,
                dt_seconds,
            )
        else:
            min_jump_distance = max(10, window // 5)

    min_jump_distance = int(min_jump_distance)

    # Internal gap allowed inside one jump region.
    if max_jump_gap_time is not None:
        max_jump_gap = _time_to_n_samples(max_jump_gap_time, dt_seconds)

    max_jump_gap = int(max_jump_gap)

    # Automatic jump-region detection.
    jump_regions, jump_score = detect_jump_regions_mad(
        y,
        threshold=jump_threshold,
        min_distance=min_jump_distance,
        max_gap=max_jump_gap,
        return_score=True,
    )

    stable_segments = _build_stable_segments(n, jump_regions)

    y_corrected = np.full(n, np.nan)
    drift_correction_full = np.full(n, np.nan)
    drift_estimated_full = np.full(n, np.nan)
    segment_id = np.full(n, -1, dtype=int)
    is_jump = np.zeros(n, dtype=bool)

    segment_outputs: list[dict[str, Any]] = []

    # Accumulated drift correction.
    # This makes the correction continue after jumps instead of restarting.
    drift_offset = 0.0

    last_filled = 0

    for seg_id, (i0, i1) in enumerate(stable_segments):
        # Fill jump region located between previous stable segment and current one.
        if last_filled < i0:
            js = last_filled
            je = i0

            y_corrected[js:je] = y[js:je] - drift_offset
            drift_correction_full[js:je] = drift_offset

            # Do not draw a drift line during jump transition.
            drift_estimated_full[js:je] = np.nan

            is_jump[js:je] = True
            segment_id[js:je] = seg_id - 1

        y_segment = y[i0:i1]

        if len(y_segment) < 5:
            drift_relative = np.zeros_like(y_segment)

            anchor = robust_segment_anchor(
                y_segment,
                n_anchor=anchor_points,
            )

            drift_for_plot = anchor + drift_relative

        else:
            trend_segment = smooth_segment_moving(
                y_segment,
                window=window,
                method=method,
            )

            # Relative drift within the stable segment.
            drift_relative = trend_segment - trend_segment[0]

            # Visual drift curve anchored to the stable post-jump level.
            anchor = robust_segment_anchor(
                y_segment,
                n_anchor=anchor_points,
            )

            drift_for_plot = anchor + drift_relative

        # Cumulative correction. Do not restart at zero after a jump.
        drift_correction_segment = drift_offset + drift_relative

        y_corrected[i0:i1] = y_segment - drift_correction_segment
        drift_correction_full[i0:i1] = drift_correction_segment
        drift_estimated_full[i0:i1] = drift_for_plot
        segment_id[i0:i1] = seg_id

        if len(drift_correction_segment) > 0:
            drift_offset = float(drift_correction_segment[-1])

        last_filled = i1

        segment_outputs.append(
            {
                "segment_id": seg_id,
                "start_index": i0,
                "end_index": i1,
                "n_points": i1 - i0,
                "start_timestamp": data.loc[i0, timestamp_col],
                "end_timestamp": data.loc[i1 - 1, timestamp_col],
                "anchor": anchor,
                "drift_offset_end": drift_offset,
            }
        )

    # If the series ends inside a jump region.
    if last_filled < n:
        y_corrected[last_filled:n] = y[last_filled:n] - drift_offset
        drift_correction_full[last_filled:n] = drift_offset
        drift_estimated_full[last_filled:n] = np.nan
        is_jump[last_filled:n] = True
        segment_id[last_filled:n] = len(stable_segments) - 1

    result = pd.DataFrame(
        {
            "timestamp": data[timestamp_col],
            "original": y,
            "corrected": y_corrected,
            "drift_estimated": drift_estimated_full,
            "drift_correction": drift_correction_full,
            "segment_id": segment_id,
            "is_jump": is_jump,
        }
    )

    metadata: dict[str, Any] = {
        "value_col": value_col,
        "timestamp_col": timestamp_col,
        "method": method,
        "window": window,
        "window_time": window_time,
        "dt_seconds": dt_seconds,
        "jump_threshold": jump_threshold,
        "min_jump_distance": min_jump_distance,
        "max_jump_gap": max_jump_gap,
        "jump_regions": jump_regions,
        "jump_start_indices": np.array([r[0] for r in jump_regions], dtype=int),
        "jump_end_indices": np.array([r[1] for r in jump_regions], dtype=int),
        "jump_start_timestamps": [
            result.loc[r[0], "timestamp"] for r in jump_regions
        ],
        "jump_end_timestamps": [
            result.loc[min(r[1], n - 1), "timestamp"] for r in jump_regions
        ],
        "stable_segments": stable_segments,
        "segment_outputs": segment_outputs,
        "jump_score": jump_score,
    }

    return result, metadata

def correct_one_segment_with_ssa(
    y_segment,
    time_cols_segment,
    window_size,
    decomposition="svd",
    min_n_components=8,
    drift_components=(0,),
    verbose=False,
    SSAClass=None,
):
    """
    Corrige la deriva de un tramo estable usando SSA.

    La deriva se estima con las componentes indicadas en drift_components.
    La deriva devuelta es relativa dentro del tramo, es decir, empieza en cero.

    Parameters
    ----------
    y_segment : array-like
        Valores del tramo estable.

    time_cols_segment : array-like
        Timestamps o etiquetas temporales del tramo. Se usan como columnas
        para construir el DataFrame requerido por py-ssa-lib.

    SSA : class
        Clase SSA de py-ssa-lib. Se pasa como argumento para no forzar
        la dependencia dentro del módulo.

    window_size : int
        Tamaño de ventana SSA, equivalente a L.

    decomposition : str
        Tipo de descomposición, por ejemplo 'svd' o 'rand_svd'.

    min_n_components : int
        Número máximo de componentes SSA a reconstruir.

    drift_components : tuple[int]
        Componentes SSA que se interpretan como deriva.

    verbose : bool
        Si True, activa salida verbose del modelo SSA.

    Returns
    -------
    y_segment_corrected : np.ndarray
        Segmento corregido localmente.

    drift_relative : np.ndarray
        Deriva relativa dentro del tramo.

    components : np.ndarray or None
        Componentes reconstruidas.

    model : object or None
        Modelo SSA ajustado.
    """

    y_segment = np.asarray(y_segment, dtype=float)
    n_segment = len(y_segment)

    if n_segment < 4:
        drift_relative = np.zeros(n_segment, dtype=float)
        return y_segment.copy(), drift_relative, None, None

    L_segment = min(int(window_size), max(2, n_segment // 2))

    if L_segment >= n_segment:
        L_segment = n_segment - 1

    if L_segment < 2:
        drift_relative = np.zeros(n_segment, dtype=float)
        return y_segment.copy(), drift_relative, None, None

    time_cols_segment = list(time_cols_segment)

    if len(time_cols_segment) != n_segment:
        time_cols_segment = list(range(n_segment))

    df_segment = pd.DataFrame(
        [["segment", *y_segment]],
        columns=["series", *time_cols_segment],
    )

    try:
        SSAClass = _get_ssa_class(SSAClass)
        model = SSAClass(Verbose=verbose)

        model.fit(
            df=df_segment,
            L=L_segment,
            ts=0,
            decomposition=decomposition,
            idx_start_ts=1,
        )

        max_components = min(int(min_n_components), model.d)

        components = np.vstack(
            [
                np.asarray(model.reconstruct_ts([i]), dtype=float).ravel()
                for i in range(max_components)
            ]
        )

        if components.shape[1] != n_segment:
            raise ValueError(
                "SSA returned components with unexpected length: "
                f"{components.shape[1]} instead of {n_segment}."
            )

        valid_drift_components = [
            i for i in drift_components
            if 0 <= int(i) < components.shape[0]
        ]

        if len(valid_drift_components) == 0:
            drift = np.zeros(n_segment, dtype=float)
        else:
            drift = components[valid_drift_components].sum(axis=0)

        drift_relative = drift - drift[0]

        y_segment_corrected = y_segment - drift_relative

        return y_segment_corrected, drift_relative, components, model

    except Exception as exc:
        print(f"SSA falló en un tramo. Se deja sin corregir. Error: {exc}")

        drift_relative = np.zeros(n_segment, dtype=float)
        y_segment_corrected = y_segment.copy()

        return y_segment_corrected, drift_relative, None, None

def correct_drift_by_segments_ssa_df(
    df,
    timestamp_col="timestamp",
    value_col="angle_observed",
    window_size=None,
    window_time=None,
    SSAClass=None,
    decomposition="svd",
    min_n_components=8,
    drift_components=(0,),
    jump_threshold=7.0,
    min_jump_distance=None,
    min_jump_distance_time=None,
    max_jump_gap=1,
    max_jump_gap_time=None,
    anchor_points=12,
    sort_by_time=True,
    verbose=False,
):
    """
    Corrige deriva instrumental con SSA por tramos, preservando saltos detectados.

    La función detecta automáticamente regiones de salto con MAD aplicado sobre
    las diferencias de la serie. Luego aplica SSA solo sobre los tramos estables.

    Durante las regiones de salto, la corrección de deriva se mantiene constante.
    Después del salto, la corrección acumulada continúa; no se reinicia en cero.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con columnas timestamp y angle_observed.

    SSA : class
        Clase SSA de py-ssa-lib.

    timestamp_col : str
        Nombre de la columna temporal.

    value_col : str
        Nombre de la columna con el ángulo observado.

    window_size : int or None
        Tamaño de ventana SSA en número de muestras.

    window_time : str or None
        Tamaño de ventana SSA en tiempo, por ejemplo '7D', '12h'.
        Se usa solo si window_size es None.

    decomposition : str
        'svd' o 'rand_svd'.

    min_n_components : int
        Número máximo de componentes SSA reconstruidas.

    drift_components : tuple[int]
        Componentes SSA usadas como deriva. Por ejemplo: (0,) o (0, 1).

    jump_threshold : float
        Umbral robusto del detector MAD.

    min_jump_distance : int or None
        Distancia mínima entre regiones de salto, en muestras.

    min_jump_distance_time : str or None
        Distancia mínima entre regiones de salto, en tiempo.

    max_jump_gap : int
        Separación máxima entre diferencias anómalas para agruparlas
        como una misma región de salto.

    max_jump_gap_time : str or None
        Igual que max_jump_gap, pero expresado en tiempo.

    anchor_points : int
        Número de puntos usados para anclar visualmente la deriva al inicio
        de cada tramo estable.

    sort_by_time : bool
        Si True, ordena el DataFrame por timestamp.

    verbose : bool
        Si True, activa Verbose en SSA.

    Returns
    -------
    result : pandas.DataFrame
        DataFrame con:
        timestamp, original, corrected, drift_estimated,
        drift_correction, segment_id, is_jump.

    metadata : dict
        Información diagnóstica del proceso.
    """

    if timestamp_col not in df.columns:
        raise ValueError(f"timestamp_col={timestamp_col!r} no está en df.")

    if value_col not in df.columns:
        raise ValueError(f"value_col={value_col!r} no está en df.")

    data = df[[timestamp_col, value_col]].copy()

    data[timestamp_col] = pd.to_datetime(data[timestamp_col])
    data[value_col] = data[value_col].astype(float)

    if sort_by_time:
        data = data.sort_values(timestamp_col).reset_index(drop=True)
    else:
        data = data.reset_index(drop=True)

    y = data[value_col].to_numpy(dtype=float)
    n = len(y)

    if n < 5:
        raise ValueError("La serie es demasiado corta para corregir deriva.")

    dt_seconds = _infer_dt_seconds(data[timestamp_col])

    if window_size is None:
        if window_time is None:
            raise ValueError("Debes proporcionar window_size o window_time.")

        window_size = _time_to_n_samples(window_time, dt_seconds)

    window_size = int(window_size)

    if window_size < 2:
        raise ValueError("window_size debe ser al menos 2.")

    if min_jump_distance is None:
        if min_jump_distance_time is not None:
            min_jump_distance = _time_to_n_samples(
                min_jump_distance_time,
                dt_seconds,
            )
        else:
            min_jump_distance = max(10, window_size // 5)

    min_jump_distance = int(min_jump_distance)

    if max_jump_gap_time is not None:
        max_jump_gap = _time_to_n_samples(
            max_jump_gap_time,
            dt_seconds,
        )

    max_jump_gap = int(max_jump_gap)

    jump_regions, jump_score = detect_jump_regions_mad(
        y,
        threshold=jump_threshold,
        min_distance=min_jump_distance,
        max_gap=max_jump_gap,
        return_score=True,
    )

    stable_segments = _build_stable_segments(
        n=n,
        jump_regions=jump_regions,
    )

    y_corrected = np.full(n, np.nan)
    drift_correction_full = np.full(n, np.nan)
    drift_estimated_full = np.full(n, np.nan)
    segment_id = np.full(n, -1, dtype=int)
    is_jump = np.zeros(n, dtype=bool)

    segment_outputs = []

    drift_offset = 0.0
    last_filled = 0

    timestamps = data[timestamp_col]

    for seg_id, (i0, i1) in enumerate(stable_segments):
        # Rellenar región de salto previa, si existe.
        if last_filled < i0:
            js = last_filled
            je = i0

            y_corrected[js:je] = y[js:je] - drift_offset
            drift_correction_full[js:je] = drift_offset

            # No dibujar deriva durante el salto/transición.
            drift_estimated_full[js:je] = np.nan

            is_jump[js:je] = True
            segment_id[js:je] = seg_id - 1

        y_segment = y[i0:i1]
        time_cols_segment = timestamps.iloc[i0:i1].tolist()

        (
            y_segment_corrected_local,
            drift_relative,
            components,
            model,
        ) = correct_one_segment_with_ssa(
            y_segment=y_segment,
            time_cols_segment=time_cols_segment,
            window_size=window_size,
            decomposition=decomposition,
            min_n_components=min_n_components,
            drift_components=drift_components,
            verbose=verbose,
            SSAClass=SSAClass,
        )

        anchor = robust_segment_anchor(
            y_segment,
            n_anchor=anchor_points,
        )

        drift_for_plot = anchor + drift_relative

        # Corrección acumulada:
        # no se reinicia después del salto.
        drift_correction_segment = drift_offset + drift_relative

        y_corrected[i0:i1] = y_segment - drift_correction_segment

        drift_correction_full[i0:i1] = drift_correction_segment
        drift_estimated_full[i0:i1] = drift_for_plot
        segment_id[i0:i1] = seg_id

        if len(drift_correction_segment) > 0:
            drift_offset = float(drift_correction_segment[-1])

        last_filled = i1

        segment_outputs.append(
            {
                "segment_id": seg_id,
                "start_index": i0,
                "end_index": i1,
                "n_points": i1 - i0,
                "start_timestamp": data.loc[i0, timestamp_col],
                "end_timestamp": data.loc[i1 - 1, timestamp_col],
                "anchor": anchor,
                "drift_offset_end": drift_offset,
                "components": components,
                "model": model,
            }
        )

    # Si la serie termina dentro de una región de salto.
    if last_filled < n:
        y_corrected[last_filled:n] = y[last_filled:n] - drift_offset
        drift_correction_full[last_filled:n] = drift_offset
        drift_estimated_full[last_filled:n] = np.nan
        is_jump[last_filled:n] = True
        segment_id[last_filled:n] = len(stable_segments) - 1

    result = pd.DataFrame(
        {
            "timestamp": data[timestamp_col],
            "original": y,
            "corrected": y_corrected,
            "drift_estimated": drift_estimated_full,
            "drift_correction": drift_correction_full,
            "segment_id": segment_id,
            "is_jump": is_jump,
        }
    )

    metadata = {
        "value_col": value_col,
        "timestamp_col": timestamp_col,
        "method": "ssa",
        "window_size": window_size,
        "window_time": window_time,
        "dt_seconds": dt_seconds,
        "decomposition": decomposition,
        "min_n_components": min_n_components,
        "drift_components": drift_components,
        "jump_threshold": jump_threshold,
        "min_jump_distance": min_jump_distance,
        "max_jump_gap": max_jump_gap,
        "jump_regions": jump_regions,
        "jump_start_indices": np.array([r[0] for r in jump_regions], dtype=int),
        "jump_end_indices": np.array([r[1] for r in jump_regions], dtype=int),
        "jump_start_timestamps": [
            result.loc[r[0], "timestamp"] for r in jump_regions
        ],
        "jump_end_timestamps": [
            result.loc[min(r[1], n - 1), "timestamp"] for r in jump_regions
        ],
        "stable_segments": stable_segments,
        "segment_outputs": segment_outputs,
        "jump_score": jump_score,
    }

    return result, metadata

def plot_drift_correction_result(
    result: pd.DataFrame,
    metadata: dict[str, Any] | None = None,
    title: str = "Drift correction preserving detected jumps",
    figsize: tuple[float, float] = (5.5, 4.5),
) -> plt.Figure:
    """
    Plot original series, estimated drift, corrected series, and jump regions.
    """

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    ax.plot(
        result["timestamp"],
        result["original"],
        label="Drifted data",
        alpha=0.6,
    )

    ax.plot(
        result["timestamp"],
        result["drift_estimated"],
        "--",
        label="Estimated drift",
    )

    ax.plot(
        result["timestamp"],
        result["corrected"],
        color="black",
        label="Corrected data",
        linewidth=1.5,
    )

    if metadata is not None:
        for i, (js, je) in enumerate(metadata["jump_regions"]):
            label_start = "Jump start" if i == 0 else None
            label_end = "Jump end" if i == 0 else None

            # ax.axvline(
            #     result.loc[js, "timestamp"],
            #     linestyle="--",
            #     linewidth=1.5,
            #     color="k",
            #     label=label_start,
            # )

            end_idx = min(je, len(result) - 1)

            # ax.axvline(
            #     result.loc[end_idx, "timestamp"],
            #     linestyle=":",
            #     linewidth=1.5,
            #     color="k",
            #     label=label_end,
            # )

            ax.axvspan(
                result.loc[js, "timestamp"],
                result.loc[end_idx, "timestamp"],
                alpha=0.50,
                color="C3",
                label="Jump region"
            )

    ax.set_title(title)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Angle [°]")
    ax.legend()
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_linewidth(1.5)
    ax.tick_params(width=1.5)
    # autoformat x-axis as dates 
    fig.autofmt_xdate()


    return fig


def plot_jump_score(
    result: pd.DataFrame,
    metadata: dict[str, Any],
    title: str = "MAD robust jump score",
    figsize: tuple[float, float] = (5.5, 4.5),
) -> plt.Figure:
    """
    Plot the robust MAD score used by the jump detector.

    The score has length n - 1 because it is computed over first differences.
    """

    score = metadata["jump_score"]
    threshold = metadata["jump_threshold"]

    # Score[k] corresponds to dy[k] = y[k + 1] - y[k].
    score_timestamps = result["timestamp"].iloc[1:].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    ax.plot(score_timestamps, score, label="Robust score", c="C0")
    ax.axhline(threshold, linestyle="--", label=f"Threshold = {threshold:g}", c="C1")

    for i, (js, je) in enumerate(metadata["jump_regions"]):
        label = "Detected jump" if i == 0 else None

        ax.axvspan(
            result.loc[js, "timestamp"],
            result.loc[min(je, len(result) - 1), "timestamp"],
            alpha=0.50,
            color="C3",
            label=label,
        )

    ax.set_title(title)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Score")
    ax.legend(loc="upper left")
    ax.grid(False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_linewidth(1.5)
    ax.tick_params(width=1.5)
    # autoformat x-axis as dates 
    fig.autofmt_xdate()


    return fig


def _corrected_angle_col_name(angle_col: str) -> str:
    """
    Convierte:
        roll_deg  -> roll_corr_deg
        pitch_deg -> pitch_corr_deg
        yaw_deg   -> yaw_corr_deg
    """

    if angle_col.endswith("_deg"):
        return angle_col.replace("_deg", "_corr_deg")

    return f"{angle_col}_corrected"

def _correct_sensor_angles_clean_template(
    df,
    corrector_func,
    angle_cols=("roll_deg", "pitch_deg", "yaw_deg"),
    timestamp_col="timestamp",
    sensor_col="sensor_id",
    station_col="estacion_id",
    min_points=10,
    raise_on_error=False,
    corrector_kwargs=None,
):
    """
    Aplica una función de corrección a todos los sensores y ángulos.

    Retorna únicamente el DataFrame original con columnas corregidas.
    No retorna metadata ni resumen.
    """

    if corrector_kwargs is None:
        corrector_kwargs = {}

    if timestamp_col not in df.columns:
        raise ValueError(f"La columna {timestamp_col!r} no existe en df.")

    if sensor_col not in df.columns:
        raise ValueError(f"La columna {sensor_col!r} no existe en df.")

    available_angle_cols = [col for col in angle_cols if col in df.columns]

    if len(available_angle_cols) == 0:
        raise ValueError(
            "No se encontró ninguna columna de ángulo. "
            f"Columnas buscadas: {angle_cols}"
        )

    group_cols = [sensor_col]

    if station_col is not None and station_col in df.columns:
        group_cols = [station_col, sensor_col]

    df_out = df.copy()
    df_out[timestamp_col] = pd.to_datetime(df_out[timestamp_col])

    for angle_col in available_angle_cols:
        corrected_col = _corrected_angle_col_name(angle_col)
        df_out[corrected_col] = np.nan

    grouped = df_out.groupby(group_cols, sort=True, dropna=False)

    for _, group in grouped:
        group_sorted = group.sort_values(timestamp_col)

        for angle_col in available_angle_cols:
            corrected_col = _corrected_angle_col_name(angle_col)

            valid = group_sorted[[timestamp_col, angle_col]].dropna()

            if len(valid) == 0:
                continue

            valid_indices = valid.index

            if len(valid) < min_points:
                df_out.loc[valid_indices, corrected_col] = valid[angle_col].to_numpy()
                continue

            series_df = pd.DataFrame(
                {
                    timestamp_col: valid[timestamp_col].to_numpy(),
                    "angle_observed": valid[angle_col].to_numpy(dtype=float),
                }
            )

            try:
                result, _ = corrector_func(
                    df=series_df,
                    timestamp_col=timestamp_col,
                    value_col="angle_observed",
                    **corrector_kwargs,
                )

                df_out.loc[valid_indices, corrected_col] = result[
                    "corrected"
                ].to_numpy(dtype=float)

            except Exception as exc:
                if raise_on_error:
                    raise

                warnings.warn(
                    f"No se pudo corregir {angle_col} para el grupo "
                    f"{group_cols}. Se copia la serie original. Error: {exc}",
                    RuntimeWarning,
                )

                df_out.loc[valid_indices, corrected_col] = valid[angle_col].to_numpy()

    return df_out

def correct_sensor_angles_moving_df(
    df,
    angle_cols=("roll_deg", "pitch_deg", "yaw_deg"),
    timestamp_col="timestamp",
    sensor_col="sensor_id",
    station_col="estacion_id",
    min_points=10,
    raise_on_error=False,
    **moving_kwargs,
):
    """
    Corrige roll, pitch y yaw para todos los sensores usando media/mediana móvil.

    Retorna el mismo DataFrame de entrada con columnas nuevas:
        roll_corr_deg
        pitch_corr_deg
        yaw_corr_deg
    """

    return _correct_sensor_angles_clean_template(
        df=df,
        corrector_func=correct_drift_by_segments_moving_df,
        angle_cols=angle_cols,
        timestamp_col=timestamp_col,
        sensor_col=sensor_col,
        station_col=station_col,
        min_points=min_points,
        raise_on_error=raise_on_error,
        corrector_kwargs=moving_kwargs,
    )

def correct_sensor_angles_ssa_df(
    df,
    angle_cols=("roll_deg", "pitch_deg", "yaw_deg"),
    timestamp_col="timestamp",
    sensor_col="sensor_id",
    station_col="estacion_id",
    min_points=10,
    raise_on_error=False,
    SSAClass=None,
    **ssa_kwargs,
):
    """
    Corrige roll, pitch y yaw para todos los sensores usando SSA segmentado.

    Retorna el mismo DataFrame de entrada con columnas nuevas:
        roll_corr_deg
        pitch_corr_deg
        yaw_corr_deg
    """

    def _ssa_corrector(df, timestamp_col, value_col, **kwargs):
        return correct_drift_by_segments_ssa_df(
            df=df,
            timestamp_col=timestamp_col,
            value_col=value_col,
            SSAClass=SSAClass,
            **kwargs,
        )

    return _correct_sensor_angles_clean_template(
        df=df,
        corrector_func=_ssa_corrector,
        angle_cols=angle_cols,
        timestamp_col=timestamp_col,
        sensor_col=sensor_col,
        station_col=station_col,
        min_points=min_points,
        raise_on_error=raise_on_error,
        corrector_kwargs=ssa_kwargs,
    )