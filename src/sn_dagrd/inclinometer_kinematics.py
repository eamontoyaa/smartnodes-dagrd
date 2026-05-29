"""
inclinometer_kinematics.py

Kinematics utilities for inclinometer data.

Important design decision
-------------------------
This module no longer asks for `separations_m` in
`compute_inclinometer_displacements`.

Instead, you pass the physical sensor depths with `depths_m`, and the module
computes the effective separations internally.

For the usual inclinometer case, use:

    reference_sensor="deepest"

Then the deepest valid capsule is used as the reference and its cumulative
displacement is exactly zero.

Expected input DataFrame
------------------------
Long format:

    timestamp | sensor_id | pitch_corr_deg | roll_corr_deg

Optionally:

    estacion_id

Expected valid_sensors format
-----------------------------

    valid_sensors = {
        "1a": 1,
        "2a": 1,
        "3a": 0,
        ...
        "15a": 1,
    }

Only sensors declared in `valid_sensors` are considered part of the
inclinometer profile. This lets you exclude superficial sensors such as
"16a", "17a", "18a".

If a declared sensor has value 0/False, it is skipped. The separation is not
assigned arbitrarily to the previous or next numerical sensor. Instead, effective
separations are recomputed from the depths of the remaining valid sensors.

Example
-------
df_disp = compute_inclinometer_displacements(
    df=df_corr,
    depths_m=[1.2, 2.2, 3.2, ..., 15.2],
    azimuth_deg=40.0,
    pitch_col="pitch_corr_deg",
    roll_col="roll_corr_deg",
    valid_sensors=valid_sensors_inc,
    reference_sensor="deepest",
)
"""

from __future__ import annotations

import re
from typing import Iterable, Literal, Mapping

import numpy as np
import pandas as pd


__version__ = "0.4.0-depths"

ReferenceSensor = Literal["deepest", "shallowest"]

__all__ = [
    "__version__",
    "sensor_id_to_order",
    "build_depth_map",
    "compute_effective_separations_from_depths",
    "depths_to_separations",
    "compute_cathetuses_from_angles",
    "get_cathetuses",
    "compute_inclinometer_displacements",
    "get_displacement_profile",
]


def sensor_id_to_order(sensor_id: str) -> int:
    """Convert sensor IDs like '1a' or '18a' to integer order."""

    match = re.search(r"\d+", str(sensor_id))

    if match is None:
        raise ValueError(
            f"Could not extract a numeric sensor order from sensor_id={sensor_id!r}."
        )

    return int(match.group())


def _sort_sensor_ids(sensor_ids: Iterable[str]) -> list[str]:
    """Sort sensor IDs by their numeric order."""

    return sorted([str(sensor_id) for sensor_id in sensor_ids], key=sensor_id_to_order)


def _validate_valid_sensors(
    valid_sensors: Mapping[str, int | bool] | None,
) -> None:
    """
    Validate valid_sensors.

    Expected:
        {"1a": 1, "2a": 1, "3a": 0}
    """

    if valid_sensors is None:
        return

    if not isinstance(valid_sensors, Mapping):
        raise TypeError("valid_sensors must be a mapping, e.g. {'1a': 1, '2a': 0}.")

    invalid_keys = [key for key in valid_sensors.keys() if not isinstance(key, str)]

    if invalid_keys:
        raise ValueError(
            "valid_sensors must use sensor_id strings as keys, "
            f"for example '1a'. Invalid keys: {invalid_keys}"
        )


def _normalize_valid_sensors(
    sensor_ids: Iterable[str],
    valid_sensors: Mapping[str, int | bool] | None = None,
) -> dict[str, bool]:
    """
    Return a complete sensor_id -> bool map.

    If valid_sensors is None, every sensor in sensor_ids is treated as valid.
    """

    sensor_ids = _sort_sensor_ids(sensor_ids)

    if valid_sensors is None:
        return {sensor_id: True for sensor_id in sensor_ids}

    _validate_valid_sensors(valid_sensors)

    return {
        sensor_id: bool(valid_sensors.get(sensor_id, False))
        for sensor_id in sensor_ids
    }


def build_depth_map(
    sensor_ids: Iterable[str],
    depths_m,
) -> dict[str, float]:
    """
    Build sensor_id -> depth_m.

    Parameters
    ----------
    sensor_ids : iterable[str]
        Sensor IDs, for example ['1a', '2a', ..., '15a'].

    depths_m : dict, pandas.Series, or array-like
        Sensor depths in meters.

        If dict/Series:
            {'1a': 1.2, '2a': 2.2, ...}

        If array-like:
            Must have the same length as sorted sensor_ids. The assumed order is:
            1a, 2a, 3a, ...

    Returns
    -------
    dict[str, float]
        Mapping sensor_id -> depth_m.
    """

    sensor_ids = _sort_sensor_ids(sensor_ids)

    if depths_m is None:
        raise ValueError(
            "depths_m is required. Pass either a dict sensor_id -> depth "
            "or a vector in the same order as the sorted sensor IDs."
        )

    if isinstance(depths_m, pd.Series):
        depths_m = depths_m.to_dict()

    if isinstance(depths_m, Mapping):
        missing = [sensor_id for sensor_id in sensor_ids if sensor_id not in depths_m]

        if missing:
            raise ValueError(f"Missing depths for sensors: {missing}")

        return {sensor_id: float(depths_m[sensor_id]) for sensor_id in sensor_ids}

    depths_m = np.asarray(depths_m, dtype=float)

    if depths_m.ndim != 1:
        raise ValueError("depths_m must be one-dimensional.")

    if len(depths_m) != len(sensor_ids):
        raise ValueError(
            "If depths_m is array-like, it must have the same length as "
            f"the sensor list. Received {len(depths_m)} depths for "
            f"{len(sensor_ids)} sensors."
        )

    if not np.all(np.isfinite(depths_m)):
        raise ValueError("depths_m contains non-finite values.")

    return {
        sensor_id: float(depth)
        for sensor_id, depth in zip(sensor_ids, depths_m)
    }


def compute_effective_separations_from_depths(
    sensor_ids: Iterable[str],
    depths_m,
    valid_sensors: Mapping[str, int | bool] | None = None,
    reference_sensor: ReferenceSensor = "deepest",
) -> dict[str, float]:
    """
    Compute effective separations from depths for the valid inclinometer sensors.

    This function removes the ambiguity of `interval_alignment` and
    `cumulative_order`.

    If reference_sensor == "deepest":
        - sensors are accumulated from deepest to shallowest;
        - the deepest valid sensor gets separation 0.0;
        - each shallower valid sensor gets the distance to the next deeper
          valid sensor.

    If reference_sensor == "shallowest":
        - sensors are accumulated from shallowest to deepest;
        - the shallowest valid sensor gets separation 0.0;
        - each deeper valid sensor gets the distance to the next shallower
          valid sensor.

    Invalid sensors are skipped, and the effective separation is computed
    directly between the remaining valid depths.

    Returns
    -------
    dict[str, float]
        Mapping valid sensor_id -> effective separation_m.
    """

    if reference_sensor not in ("deepest", "shallowest"):
        raise ValueError("reference_sensor must be 'deepest' or 'shallowest'.")

    sensor_ids = _sort_sensor_ids(sensor_ids)
    depth_map = build_depth_map(sensor_ids, depths_m)

    valid_map = _normalize_valid_sensors(sensor_ids, valid_sensors)
    valid_sensor_ids = [sid for sid in sensor_ids if valid_map[sid]]

    if len(valid_sensor_ids) == 0:
        raise ValueError("No valid sensors remain after applying valid_sensors.")

    # Sort by physical depth, not by sensor number.
    # Assumption: larger depth_m means deeper sensor.
    reverse = reference_sensor == "deepest"
    ordered = sorted(valid_sensor_ids, key=lambda sid: depth_map[sid], reverse=reverse)

    separation_map: dict[str, float] = {}
    previous_depth = None

    for sid in ordered:
        depth = float(depth_map[sid])

        if previous_depth is None:
            separation_map[sid] = 0.0
        else:
            separation_map[sid] = abs(previous_depth - depth)

        previous_depth = depth

    return separation_map


def depths_to_separations(depths_m) -> np.ndarray:
    """
    Backward-compatible helper.

    Converts ordered depths into differences between consecutive depths.

    New code should usually call `compute_inclinometer_displacements` with
    `depths_m` directly instead of passing separations.
    """

    depths_m = np.asarray(depths_m, dtype=float)

    if depths_m.ndim != 1:
        raise ValueError("depths_m must be one-dimensional.")

    if len(depths_m) < 2:
        raise ValueError("At least two depths are required.")

    if not np.all(np.isfinite(depths_m)):
        raise ValueError("depths_m contains non-finite values.")

    return np.abs(np.diff(depths_m))


def compute_cathetuses_from_angles(
    pitch_deg,
    roll_deg,
    separations_m,
    azimuth_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute local and geographic cathetus components.

    Parameters
    ----------
    pitch_deg : array-like
        Pitch angle in degrees.

    roll_deg : array-like
        Roll angle in degrees.

    separations_m : array-like or float
        Effective segment/capsule separation in meters.

    azimuth_deg : float
        Azimuth of local A axis in degrees.

    Returns
    -------
    cat_a_mm, cat_b_mm, cat_north_mm, cat_east_mm : tuple[np.ndarray, ...]
        Cathetus components in millimeters.
    """

    pitch_rad = np.deg2rad(np.asarray(pitch_deg, dtype=float))
    roll_rad = np.deg2rad(np.asarray(roll_deg, dtype=float))
    azimuth_rad = np.deg2rad(float(azimuth_deg))
    separations_m = np.asarray(separations_m, dtype=float)

    cat_a_mm = 1000.0 * separations_m * np.sin(pitch_rad)
    cat_b_mm = 1000.0 * separations_m * np.sin(roll_rad)

    cat_north_mm = (
        -np.sin(azimuth_rad) * cat_b_mm
        + np.cos(azimuth_rad) * cat_a_mm
    )

    cat_east_mm = (
        np.cos(azimuth_rad) * cat_b_mm
        + np.sin(azimuth_rad) * cat_a_mm
    )

    return cat_a_mm, cat_b_mm, cat_north_mm, cat_east_mm


def get_cathetuses(
    pitch,
    roll,
    separations,
    azimuth,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Backward-compatible alias for compute_cathetuses_from_angles."""

    return compute_cathetuses_from_angles(
        pitch_deg=pitch,
        roll_deg=roll,
        separations_m=separations,
        azimuth_deg=azimuth,
    )


def compute_inclinometer_displacements(
    df: pd.DataFrame,
    depths_m,
    azimuth_deg: float,
    pitch_col: str = "pitch_corr_deg",
    roll_col: str = "roll_corr_deg",
    timestamp_col: str = "timestamp",
    sensor_col: str = "sensor_id",
    station_col: str | None = "estacion_id",
    valid_sensors: Mapping[str, int | bool] | None = None,
    reference_sensor: ReferenceSensor = "deepest",
    keep_sensor_order_col: bool = True,
    keep_valid_sensor_col: bool = True,
) -> pd.DataFrame:
    """
    Compute cathetuses, relative cathetuses, and cumulative displacements.

    Parameters
    ----------
    df : pandas.DataFrame
        Long DataFrame with one row per timestamp and sensor.

    depths_m : dict, Series, or array-like
        Sensor depths in meters. If array-like, the order is the sorted declared
        sensors, typically 1a, 2a, ..., 15a.

    azimuth_deg : float
        Azimuth of local A axis in degrees.

    pitch_col : str
        Pitch column to use. Default: 'pitch_corr_deg'.

    roll_col : str
        Roll column to use. Default: 'roll_corr_deg'.

    valid_sensors : dict or None
        Dict with sensor IDs as keys, for example:

            {"1a": 1, "2a": 1, "3a": 0, ..., "15a": 1}

        If passed, only declared sensors are part of the inclinometer profile.
        Sensors with value 0/False are excluded.

    reference_sensor : {'deepest', 'shallowest'}
        Which sensor is the fixed reference.

        For a conventional inclinometer, use 'deepest'. The deepest valid
        sensor will have zero effective separation and therefore zero cumulative
        displacement.

    Returns
    -------
    pandas.DataFrame
        DataFrame with additional columns:

        - depth_m
        - separation_m
        - cat_a_mm, cat_b_mm, cat_north_mm, cat_east_mm
        - rel_cat_a_mm, rel_cat_b_mm, rel_cat_north_mm, rel_cat_east_mm
        - cum_disp_a_mm, cum_disp_b_mm, cum_disp_north_mm, cum_disp_east_mm
    """

    required_cols = [timestamp_col, sensor_col, pitch_col, roll_col]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column {col!r} does not exist in df.")

    if reference_sensor not in ("deepest", "shallowest"):
        raise ValueError("reference_sensor must be 'deepest' or 'shallowest'.")

    _validate_valid_sensors(valid_sensors)

    data = df.copy()
    data[timestamp_col] = pd.to_datetime(data[timestamp_col])
    data[sensor_col] = data[sensor_col].astype(str)
    data["_sensor_order"] = data[sensor_col].map(sensor_id_to_order)

    all_sensor_ids = _sort_sensor_ids(data[sensor_col].dropna().unique())

    if valid_sensors is not None:
        declared_sensor_ids = _sort_sensor_ids(valid_sensors.keys())
        profile_sensor_ids = [
            sensor_id for sensor_id in declared_sensor_ids
            if sensor_id in all_sensor_ids
        ]
    else:
        profile_sensor_ids = all_sensor_ids

    if len(profile_sensor_ids) == 0:
        raise ValueError("No profile sensors were found in the input DataFrame.")

    depth_map = build_depth_map(profile_sensor_ids, depths_m)

    valid_map = _normalize_valid_sensors(
        sensor_ids=profile_sensor_ids,
        valid_sensors=valid_sensors,
    )

    separation_map = compute_effective_separations_from_depths(
        sensor_ids=profile_sensor_ids,
        depths_m=depth_map,
        valid_sensors=valid_map,
        reference_sensor=reference_sensor,
    )

    valid_sensor_ids = [sid for sid in profile_sensor_ids if valid_map[sid]]

    data = data[data[sensor_col].isin(valid_sensor_ids)].copy()
    data["is_valid_sensor"] = data[sensor_col].map(valid_map).astype(bool)
    data["depth_m"] = data[sensor_col].map(depth_map).astype(float)
    data["separation_m"] = data[sensor_col].map(separation_map).astype(float)

    if data.empty:
        raise ValueError("After excluding invalid sensors, no data remains.")

    (
        data["cat_a_mm"],
        data["cat_b_mm"],
        data["cat_north_mm"],
        data["cat_east_mm"],
    ) = compute_cathetuses_from_angles(
        pitch_deg=data[pitch_col].to_numpy(dtype=float),
        roll_deg=data[roll_col].to_numpy(dtype=float),
        separations_m=data["separation_m"].to_numpy(dtype=float),
        azimuth_deg=azimuth_deg,
    )

    cat_cols = ["cat_a_mm", "cat_b_mm", "cat_north_mm", "cat_east_mm"]
    rel_cols = ["rel_cat_a_mm", "rel_cat_b_mm", "rel_cat_north_mm", "rel_cat_east_mm"]

    group_cols = [sensor_col]
    if station_col is not None and station_col in data.columns:
        group_cols = [station_col, sensor_col]

    # Relative cathetuses with respect to first reading of each valid sensor.
    data = data.sort_values([*group_cols, timestamp_col]).reset_index(drop=True)

    first_values = (
        data
        .groupby(group_cols, dropna=False)[cat_cols]
        .transform("first")
    )

    data[rel_cols] = (
        data[cat_cols].to_numpy(dtype=float)
        - first_values.to_numpy(dtype=float)
    )

    # Cumulative displacement by timestamp along the physical profile.
    # For reference_sensor="deepest", the deepest valid sensor has separation 0,
    # so its cumulative displacement is exactly zero.
    if station_col is not None and station_col in data.columns:
        profile_group_cols = [station_col, timestamp_col]
    else:
        profile_group_cols = [timestamp_col]

    ascending_depth = reference_sensor == "shallowest"

    data = data.sort_values(
        [*profile_group_cols, "depth_m"],
        ascending=[True] * len(profile_group_cols) + [ascending_depth],
    ).reset_index(drop=True)

    data["_profile_order"] = (
        data
        .groupby(profile_group_cols, dropna=False)
        .cumcount()
    )

    cumulative_cols = {
        "rel_cat_a_mm": "cum_disp_a_mm",
        "rel_cat_b_mm": "cum_disp_b_mm",
        "rel_cat_north_mm": "cum_disp_north_mm",
        "rel_cat_east_mm": "cum_disp_east_mm",
    }

    for rel_col, cum_col in cumulative_cols.items():
        data[cum_col] = (
            data
            .groupby(profile_group_cols, dropna=False)[rel_col]
            .cumsum()
        )

    data = data.sort_values([*group_cols, timestamp_col]).reset_index(drop=True)

    if not keep_sensor_order_col:
        data = data.drop(columns=["_sensor_order", "_profile_order"], errors="ignore")

    if not keep_valid_sensor_col:
        data = data.drop(columns=["is_valid_sensor"], errors="ignore")

    return data


def get_displacement_profile(
    df_disp: pd.DataFrame,
    timestamp=None,
    timestamp_col: str = "timestamp",
    sensor_col: str = "sensor_id",
    station_col: str | None = "estacion_id",
    station_id=None,
    method: Literal["exact", "nearest"] = "nearest",
) -> pd.DataFrame:
    """
    Extract a displacement profile at a selected timestamp.

    If timestamp is None, the latest timestamp is used.
    """

    data = df_disp.copy()
    data[timestamp_col] = pd.to_datetime(data[timestamp_col])

    if station_col is not None and station_col in data.columns and station_id is not None:
        data = data[data[station_col] == station_id].copy()

    if data.empty:
        return data

    if timestamp is None:
        selected_timestamp = data[timestamp_col].max()
    else:
        target = pd.Timestamp(timestamp)

        if method == "exact":
            selected_timestamp = target

        elif method == "nearest":
            available_times = data[timestamp_col].drop_duplicates().sort_values()

            if available_times.empty:
                return data.iloc[0:0].copy()

            idx = np.argmin(np.abs(available_times - target))
            selected_timestamp = available_times.iloc[int(idx)]

        else:
            raise ValueError("method must be 'exact' or 'nearest'.")

    profile = data[data[timestamp_col] == selected_timestamp].copy()

    if "depth_m" in profile.columns:
        profile = profile.sort_values("depth_m").reset_index(drop=True)
    elif "_sensor_order" in profile.columns:
        profile = profile.sort_values("_sensor_order").reset_index(drop=True)
    else:
        profile["_sensor_order"] = profile[sensor_col].map(sensor_id_to_order)
        profile = profile.sort_values("_sensor_order").reset_index(drop=True)

    return profile
