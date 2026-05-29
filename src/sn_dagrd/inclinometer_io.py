"""
inclinometer_io.py

Lectura y tabulación de archivos JSON Lines de inclinómetros SmartNode/DAGRD.

Formato esperado por línea
--------------------------
Cada línea del archivo es un JSON independiente con esta estructura general:

{
    "estacion_id": "SmartNode-SantaRita1",
    "timestamp": "2026-04-22T23:15:55",
    "data": [
        {"id": "18a", "qr": 19976, "qi": 9375, "qj": 10282, "qk": 9982},
        ...
    ],
    "uptime_s": 18019,
    "heap": 103580,
    "rssi": -15
}

Uso típico
----------
from sn_dagrd.inclinometer_io import read_inclinometer_jsonl, to_angle_series

raw = read_inclinometer_jsonl("raw_InclinometroSantaRita1.txt", sensor_type="BNO")

serie = to_angle_series(
    raw,
    sensor_id="6a",
    angle_col="pitch_deg",
)

# serie queda con columnas:
# timestamp | angle_observed
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Generator, Iterable, Literal

import numpy as np
import pandas as pd


SensorType = Literal["BNO", "ISM", "RAW"]
JsonErrorMode = Literal["raise", "skip"]


__all__ = [
    "_quaternion_to_euler",
    "iter_inclinometer_jsonl_records",
    "read_inclinometer_jsonl",
    "add_euler_angles",
    "to_angle_series",
    "to_wide_angle_table",
]


def _quaternion_to_euler(
    qr: float,
    qi: float,
    qj: float,
    qk: float,
    s_type: str = "BNO",
) -> tuple[float, float, float]:
    """
    Convierte quaternion a ángulos de Euler en secuencia ZYX (Tait-Bryan).

    Convención ZYX:
        R = Rz(yaw) · Ry(pitch) · Rx(roll)

    Parámetros
    ----------
    qr, qi, qj, qk : float
        Componentes del quaternion q = [qr, qi, qj, qk], donde:
        qi = qx, qj = qy, qk = qz.

    s_type : {'BNO', 'ISM', 'RAW'}
        Tipo de dato/sensor.

        - 'BNO': se asume codificación offset:
          componente_real = valor_crudo / 10000 - 1.

        - 'ISM': se asume que no hay quaternion sino ángulos codificados:
          roll = qr / 100 - 180
          pitch = qi / 100 - 90
          yaw = NaN

        - 'RAW': se asume que qr, qi, qj, qk ya son componentes reales
          del quaternion.

    Returns
    -------
    roll_deg, pitch_deg, yaw_deg : tuple[float, float, float]
        Ángulos en grados.
    """

    try:
        qr = float(qr)
        qi = float(qi)
        qj = float(qj)
        qk = float(qk)
    except (TypeError, ValueError):
        return np.nan, np.nan, np.nan

    s_type = str(s_type).upper()

    if s_type == "BNO":
        qr = qr / 10000.0 - 1.0
        qi = qi / 10000.0 - 1.0
        qj = qj / 10000.0 - 1.0
        qk = qk / 10000.0 - 1.0

    elif s_type == "ISM":
        roll = qr / 100.0 - 180.0
        pitch = qi / 100.0 - 90.0
        return roll, pitch, np.nan

    elif s_type == "RAW":
        pass

    else:
        raise ValueError("s_type debe ser 'BNO', 'ISM' o 'RAW'.")

    norm = math.sqrt(qr**2 + qi**2 + qj**2 + qk**2)

    if norm == 0 or not math.isfinite(norm):
        return np.nan, np.nan, np.nan

    qr = qr / norm
    qi = qi / norm
    qj = qj / norm
    qk = qk / norm

    roll = math.atan2(
        2.0 * (qr * qi + qj * qk),
        1.0 - 2.0 * (qi**2 + qj**2),
    )

    sinp = 2.0 * (qr * qj - qk * qi)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    yaw = math.atan2(
        2.0 * (qr * qk + qi * qj),
        1.0 - 2.0 * (qj**2 + qk**2),
    )

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def _quaternion_arrays_to_euler(
    qr: np.ndarray,
    qi: np.ndarray,
    qj: np.ndarray,
    qk: np.ndarray,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Conversión vectorizada quaternion -> Euler ZYX.
    """

    qr = np.asarray(qr, dtype=float)
    qi = np.asarray(qi, dtype=float)
    qj = np.asarray(qj, dtype=float)
    qk = np.asarray(qk, dtype=float)

    if normalize:
        norm = np.sqrt(qr**2 + qi**2 + qj**2 + qk**2)
        valid = np.isfinite(norm) & (norm > 0)

        qr_n = np.full_like(qr, np.nan, dtype=float)
        qi_n = np.full_like(qi, np.nan, dtype=float)
        qj_n = np.full_like(qj, np.nan, dtype=float)
        qk_n = np.full_like(qk, np.nan, dtype=float)

        qr_n[valid] = qr[valid] / norm[valid]
        qi_n[valid] = qi[valid] / norm[valid]
        qj_n[valid] = qj[valid] / norm[valid]
        qk_n[valid] = qk[valid] / norm[valid]

        qr, qi, qj, qk = qr_n, qi_n, qj_n, qk_n

    roll = np.arctan2(
        2.0 * (qr * qi + qj * qk),
        1.0 - 2.0 * (qi**2 + qj**2),
    )

    sinp = 2.0 * (qr * qj - qk * qi)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    yaw = np.arctan2(
        2.0 * (qr * qk + qi * qj),
        1.0 - 2.0 * (qj**2 + qk**2),
    )

    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def add_euler_angles(
    df: pd.DataFrame,
    sensor_type: SensorType | str = "BNO",
    qr_col: str = "qr",
    qi_col: str = "qi",
    qj_col: str = "qj",
    qk_col: str = "qk",
    roll_col: str = "roll_deg",
    pitch_col: str = "pitch_deg",
    yaw_col: str = "yaw_deg",
    copy: bool = True,
) -> pd.DataFrame:
    """
    Agrega columnas roll, pitch y yaw a un DataFrame con columnas qr, qi, qj, qk.

    Esta función usa operaciones vectorizadas y es preferible a aplicar
    `_quaternion_to_euler` fila por fila.
    """

    out = df.copy() if copy else df

    missing = [col for col in [qr_col, qi_col, qj_col, qk_col] if col not in out.columns]
    if missing:
        raise ValueError(f"Faltan columnas para convertir orientación: {missing}")

    qr = pd.to_numeric(out[qr_col], errors="coerce").to_numpy(dtype=float)
    qi = pd.to_numeric(out[qi_col], errors="coerce").to_numpy(dtype=float)
    qj = pd.to_numeric(out[qj_col], errors="coerce").to_numpy(dtype=float)
    qk = pd.to_numeric(out[qk_col], errors="coerce").to_numpy(dtype=float)

    sensor_type = str(sensor_type).upper()

    if sensor_type == "BNO":
        qr = qr / 10000.0 - 1.0
        qi = qi / 10000.0 - 1.0
        qj = qj / 10000.0 - 1.0
        qk = qk / 10000.0 - 1.0

        roll, pitch, yaw = _quaternion_arrays_to_euler(
            qr,
            qi,
            qj,
            qk,
            normalize=True,
        )

    elif sensor_type == "ISM":
        roll = qr / 100.0 - 180.0
        pitch = qi / 100.0 - 90.0
        yaw = np.full_like(roll, np.nan, dtype=float)

    elif sensor_type == "RAW":
        roll, pitch, yaw = _quaternion_arrays_to_euler(
            qr,
            qi,
            qj,
            qk,
            normalize=True,
        )

    else:
        raise ValueError("sensor_type debe ser 'BNO', 'ISM' o 'RAW'.")

    out[roll_col] = roll
    out[pitch_col] = pitch
    out[yaw_col] = yaw

    return out


def iter_inclinometer_jsonl_records(
    path: str | Path,
    encoding: str = "utf-8",
    errors: JsonErrorMode = "raise",
) -> Generator[dict[str, Any], None, None]:
    """
    Itera eficientemente sobre un archivo JSON Lines de inclinómetro.

    Cada registro de salida corresponde a una lectura de un sensor individual
    dentro del arreglo `data` de una línea JSON.

    Parameters
    ----------
    path : str or pathlib.Path
        Ruta del archivo JSONL/TXT.

    encoding : str
        Codificación del archivo.

    errors : {'raise', 'skip'}
        Qué hacer si una línea no es JSON válido.

    Yields
    ------
    dict
        Registro plano con metadatos de estación, timestamp, id del sensor y
        componentes qr, qi, qj, qk.
    """

    path = Path(path)

    if errors not in {"raise", "skip"}:
        raise ValueError("errors debe ser 'raise' o 'skip'.")

    with path.open("r", encoding=encoding) as file:
        for line_no, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                if errors == "skip":
                    continue
                raise

            station_id = payload.get("estacion_id")
            timestamp = payload.get("timestamp")
            uptime_s = payload.get("uptime_s")
            heap = payload.get("heap")
            rssi = payload.get("rssi")

            sensor_rows = payload.get("data", [])

            if sensor_rows is None:
                continue

            for item in sensor_rows:
                if not isinstance(item, dict):
                    continue

                yield {
                    "line_no": line_no,
                    "estacion_id": station_id,
                    "timestamp": timestamp,
                    "sensor_id": item.get("id"),
                    "qr": item.get("qr"),
                    "qi": item.get("qi"),
                    "qj": item.get("qj"),
                    "qk": item.get("qk"),
                    "uptime_s": uptime_s,
                    "heap": heap,
                    "rssi": rssi,
                }


def read_inclinometer_jsonl(
    path: str | Path,
    sensor_type: SensorType | str = "BNO",
    encoding: str = "utf-8",
    errors: JsonErrorMode = "raise",
    add_angles: bool = True,
    sort: bool = True,
    drop_duplicate_rows: bool = False,
) -> pd.DataFrame:
    """
    Lee un archivo JSON Lines de inclinómetro y devuelve una tabla larga.

    Parameters
    ----------
    path : str or pathlib.Path
        Ruta del archivo.

    sensor_type : {'BNO', 'ISM', 'RAW'}
        Tipo/codificación de los datos de orientación.

    encoding : str
        Codificación del archivo.

    errors : {'raise', 'skip'}
        Manejo de líneas JSON inválidas.

    add_angles : bool
        Si True, agrega roll_deg, pitch_deg y yaw_deg.

    sort : bool
        Si True, ordena por timestamp y sensor_id.

    drop_duplicate_rows : bool
        Si True, elimina duplicados por timestamp y sensor_id, conservando
        el último registro.

    Returns
    -------
    pandas.DataFrame
        Tabla larga con una fila por timestamp y sensor.
    """

    records = list(
        iter_inclinometer_jsonl_records(
            path=path,
            encoding=encoding,
            errors=errors,
        )
    )

    columns = [
        "line_no",
        "estacion_id",
        "timestamp",
        "sensor_id",
        "qr",
        "qi",
        "qj",
        "qk",
        "uptime_s",
        "heap",
        "rssi",
    ]

    df = pd.DataFrame.from_records(records, columns=columns)

    if df.empty:
        if add_angles:
            for col in ["roll_deg", "pitch_deg", "yaw_deg"]:
                df[col] = pd.Series(dtype=float)
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    numeric_cols = ["qr", "qi", "qj", "qk", "uptime_s", "heap", "rssi", "line_no"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["sensor_id"] = df["sensor_id"].astype("string")
    df["estacion_id"] = df["estacion_id"].astype("string")

    if drop_duplicate_rows:
        df = df.drop_duplicates(
            subset=["timestamp", "sensor_id"],
            keep="last",
        )

    if add_angles:
        df = add_euler_angles(
            df,
            sensor_type=sensor_type,
            copy=False,
        )

    if sort:
        df = df.sort_values(["timestamp", "sensor_id"]).reset_index(drop=True)

    return df


def to_angle_series(
    df: pd.DataFrame,
    sensor_id: str,
    angle_col: str = "pitch_deg",
    timestamp_col: str = "timestamp",
    output_col: str = "angle_observed",
    sort: bool = True,
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Extrae una serie de un sensor y un ángulo específico para corregir deriva.

    Returns
    -------
    pandas.DataFrame
        DataFrame con columnas:
        timestamp | angle_observed
    """

    required = {"sensor_id", timestamp_col, angle_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    mask = df["sensor_id"].astype(str) == str(sensor_id)

    out = df.loc[mask, [timestamp_col, angle_col]].copy()
    out = out.rename(columns={angle_col: output_col})

    if dropna:
        out = out.dropna(subset=[timestamp_col, output_col])

    if sort:
        out = out.sort_values(timestamp_col).reset_index(drop=True)

    return out


def to_wide_angle_table(
    df: pd.DataFrame,
    angle_col: str = "pitch_deg",
    timestamp_col: str = "timestamp",
    sensor_col: str = "sensor_id",
    series_col: str = "series",
    aggfunc: str = "first",
) -> pd.DataFrame:
    """
    Convierte la tabla larga a formato wide compatible con flujos tipo SSA.

    Output:
        series | timestamp_0 | timestamp_1 | ...
    """

    required = {sensor_col, timestamp_col, angle_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")

    wide = (
        df.pivot_table(
            index=sensor_col,
            columns=timestamp_col,
            values=angle_col,
            aggfunc=aggfunc,
        )
        .reset_index()
        .rename(columns={sensor_col: series_col})
    )

    wide.columns.name = None

    return wide


def resample_inclinometer_df(
    df,
    freq="6h",
    timestamp_col="timestamp",
    group_cols=("estacion_id", "sensor_id"),
    method="median",
    numeric_cols=None,
    min_count=1,
    sort=True,
):
    """
    Reduce la resolución temporal de un DataFrame largo de inclinómetros.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame largo, por ejemplo el resultado de read_inclinometer_jsonl.

    freq : str
        Frecuencia de remuestreo. Ejemplos: "6h", "1D", "30min".

    timestamp_col : str
        Nombre de la columna temporal.

    group_cols : tuple[str]
        Columnas usadas para separar series independientes.
        Normalmente: ("estacion_id", "sensor_id").

    method : str
        Método de agregación:
        - "median"
        - "mean"
        - "first"
        - "last"

    numeric_cols : list[str] or None
        Columnas numéricas a remuestrear. Si None, usa todas las numéricas.

    min_count : int
        Número mínimo de datos requeridos en una ventana para conservarla.

    sort : bool
        Si True, ordena la salida por group_cols + timestamp_col.

    Returns
    -------
    pandas.DataFrame
        DataFrame remuestreado.
    """

    data = df.copy()

    if timestamp_col not in data.columns:
        raise ValueError(f"La columna {timestamp_col!r} no existe en df.")

    data[timestamp_col] = pd.to_datetime(data[timestamp_col])

    group_cols = tuple(col for col in group_cols if col in data.columns)

    if len(group_cols) == 0:
        raise ValueError("Ninguna columna de agrupación existe en df.")

    if numeric_cols is None:
        numeric_cols = (
            data
            .select_dtypes(include=[np.number])
            .columns
            .tolist()
        )
    else:
        numeric_cols = [col for col in numeric_cols if col in data.columns]

    if len(numeric_cols) == 0:
        raise ValueError("No hay columnas numéricas para remuestrear.")

    grouped = data.groupby(
        [
            *group_cols,
            pd.Grouper(key=timestamp_col, freq=freq),
        ],
        dropna=False,
    )

    if method == "median":
        out = grouped[numeric_cols].median()

    elif method == "mean":
        out = grouped[numeric_cols].mean()

    elif method == "first":
        out = grouped[numeric_cols].first()

    elif method == "last":
        out = grouped[numeric_cols].last()

    else:
        raise ValueError("method debe ser 'median', 'mean', 'first' o 'last'.")

    counts = grouped.size().rename("n_samples")

    out = out.join(counts).reset_index()

    if min_count is not None:
        out = out[out["n_samples"] >= min_count].copy()

    if sort:
        out = out.sort_values([*group_cols, timestamp_col]).reset_index(drop=True)

    return out


def sample_inclinometer_df(
    df: pd.DataFrame,
    freq: str = "6h",
    timestamp_col: str = "timestamp",
    group_cols: tuple[str, ...] = ("estacion_id", "sensor_id"),
    how: Literal["first", "last"] = "first",
    sort: bool = True,
) -> pd.DataFrame:
    """
    Submuestrea un DataFrame de inclinómetros conservando filas reales.

    A diferencia de `resample_inclinometer_df`, esta función no calcula
    medianas/promedios ni crea valores agregados. Divide el tiempo en bloques
    de tamaño `freq` y toma una fila existente por grupo y bloque.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame largo de inclinómetros.

    freq : str
        Frecuencia de submuestreo. Ejemplos: "6h", "12h", "1D".

    timestamp_col : str
        Columna temporal.

    group_cols : tuple[str, ...]
        Columnas para separar series independientes.

    how : {"first", "last"}
        Fila real que se toma dentro de cada bloque temporal.

    sort : bool
        Si True, ordena por group_cols + timestamp_col.

    Returns
    -------
    pandas.DataFrame
        DataFrame submuestreado conservando columnas originales.
    """

    if how not in {"first", "last"}:
        raise ValueError("how debe ser 'first' o 'last'.")

    data = df.copy()

    if timestamp_col not in data.columns:
        raise ValueError(f"La columna {timestamp_col!r} no existe en df.")

    data[timestamp_col] = pd.to_datetime(data[timestamp_col])

    group_cols = tuple(col for col in group_cols if col in data.columns)

    if len(group_cols) == 0:
        raise ValueError("Ninguna columna de agrupación existe en df.")

    if sort:
        data = data.sort_values([*group_cols, timestamp_col]).reset_index(drop=True)

    data["_sample_bin"] = data[timestamp_col].dt.floor(freq)

    grouped = data.groupby([*group_cols, "_sample_bin"], sort=True, dropna=False)

    if how == "first":
        out = grouped.head(1).copy()
    else:
        out = grouped.tail(1).copy()

    out = out.drop(columns=["_sample_bin"])

    if sort:
        out = out.sort_values([*group_cols, timestamp_col]).reset_index(drop=True)

    return out

