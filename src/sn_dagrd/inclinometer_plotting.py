"""
inclinometer_plotting.py

Plotting utilities for inclinometer displacement and correction results.

The plotting API uses explicit sensor depths. It does not accept `spacing_m`.

Main functions
--------------
- plot_cumulative_displacement_profiles
- plot_single_displacement_profile
- plot_sensor_displacement_evolution
- plot_angle_correction_over_time

All functions return only the matplotlib Figure object.
"""

from __future__ import annotations

import re
from typing import Literal, Mapping

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


__version__ = "0.2.0-depths"

Components = Literal["ab", "ne"]

__all__ = [
    "__version__",
    "sensor_id_to_order",
    "build_depth_map",
    "plot_cumulative_displacement_profiles",
    "plot_single_displacement_profile",
    "plot_sensor_displacement_evolution",
    "plot_angle_correction_over_time",
]


def sensor_id_to_order(sensor_id: str) -> int:
    """Convert a sensor ID such as '1a' or '18a' to its integer order."""

    match = re.search(r"\d+", str(sensor_id))

    if match is None:
        raise ValueError(
            f"Could not extract a numeric sensor order from sensor_id={sensor_id!r}."
        )

    return int(match.group())


def _sort_sensor_ids(sensor_ids) -> list[str]:
    """Sort sensor IDs by their numeric order."""

    return sorted(
        [str(sensor_id) for sensor_id in sensor_ids],
        key=sensor_id_to_order,
    )


def build_depth_map(
    sensor_ids,
    depths_m=None,
    data: pd.DataFrame | None = None,
    sensor_col: str = "sensor_id",
    depth_col: str = "depth_m",
) -> dict[str, float]:
    """
    Build sensor_id -> depth_m.

    Parameters
    ----------
    sensor_ids : iterable
        Sensor IDs, e.g. ['1a', '2a', ..., '15a'].

    depths_m : dict, pandas.Series, array-like, or None
        Sensor depths.

        If dict/Series:
            {'1a': 1.2, '2a': 2.2, ...}

        If array-like:
            Must have the same length as sorted sensor_ids.

        If None, the function tries to read depth_col from data.

    data : pandas.DataFrame or None
        Optional DataFrame with sensor_col and depth_col.

    Returns
    -------
    dict[str, float]
        Mapping sensor_id -> depth_m.
    """

    sensor_ids = _sort_sensor_ids(sensor_ids)

    if depths_m is None:
        if data is not None and depth_col in data.columns:
            depth_by_sensor = (
                data[[sensor_col, depth_col]]
                .dropna()
                .assign(**{sensor_col: lambda d: d[sensor_col].astype(str)})
                .drop_duplicates(subset=[sensor_col])
                .set_index(sensor_col)[depth_col]
                .to_dict()
            )

            missing = [sensor_id for sensor_id in sensor_ids if sensor_id not in depth_by_sensor]

            if missing:
                raise ValueError(f"Missing depths in data for sensors: {missing}")

            return {sensor_id: float(depth_by_sensor[sensor_id]) for sensor_id in sensor_ids}

        raise ValueError(
            "depths_m is required unless df_disp already contains a depth_m column."
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
            f"the declared sensor list. Received {len(depths_m)} depths for "
            f"{len(sensor_ids)} sensors."
        )

    return {
        sensor_id: float(depth)
        for sensor_id, depth in zip(sensor_ids, depths_m)
    }


def _get_component_columns(components: Components) -> tuple[str, str, list[str]]:
    """Resolve displacement component columns and labels."""

    if components == "ab":
        return (
            "cum_disp_a_mm",
            "cum_disp_b_mm",
            ["A", "B", "Resultante"],
        )

    if components == "ne":
        return (
            "cum_disp_north_mm",
            "cum_disp_east_mm",
            ["Norte", "Este", "Resultante"],
        )

    raise ValueError("components must be 'ab' or 'ne'.")


def _get_declared_valid_invalid_sensors(
    data: pd.DataFrame,
    sensor_col: str,
    valid_sensors: Mapping[str, int | bool] | None,
) -> tuple[list[str], list[str], list[str]]:
    """
    Return declared, valid, and invalid sensor IDs.

    If valid_sensors is provided:
    - declared sensors are the keys in valid_sensors;
    - valid sensors are those with truthy values;
    - invalid sensors are those with falsy values.

    If valid_sensors is None:
    - declared sensors are those present in data;
    - all declared sensors are valid.
    """

    if valid_sensors is not None:
        declared_sensor_ids = _sort_sensor_ids(valid_sensors.keys())

        valid_sensor_ids = _sort_sensor_ids(
            [sensor_id for sensor_id, is_valid in valid_sensors.items() if bool(is_valid)]
        )

        invalid_sensor_ids = _sort_sensor_ids(
            [sensor_id for sensor_id, is_valid in valid_sensors.items() if not bool(is_valid)]
        )

        return declared_sensor_ids, valid_sensor_ids, invalid_sensor_ids

    declared_sensor_ids = _sort_sensor_ids(data[sensor_col].dropna().unique())

    return declared_sensor_ids, declared_sensor_ids, []


def _select_timestamps(
    timestamps,
    profile_freq: str | None = None,
    profile_selection: str = "first",
    max_profiles: int | None = None,
):
    """
    Select timestamps to plot.

    profile_freq examples:
        "6h", "12h", "1D", "7D"

    profile_selection:
        "first", "last", or "nearest".
    """

    ts = (
        pd.Series(pd.to_datetime(timestamps))
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    if len(ts) == 0:
        return np.array([], dtype="datetime64[ns]")

    if profile_freq is not None:
        tmp = pd.DataFrame({"timestamp": ts})
        tmp["_bin"] = tmp["timestamp"].dt.floor(profile_freq)

        if profile_selection == "first":
            selected = tmp.groupby("_bin", sort=True)["timestamp"].first().reset_index(drop=True)

        elif profile_selection == "last":
            selected = tmp.groupby("_bin", sort=True)["timestamp"].last().reset_index(drop=True)

        elif profile_selection == "nearest":
            freq_delta = pd.to_timedelta(profile_freq)
            tmp["_target"] = tmp["_bin"] + freq_delta / 2
            tmp["_distance"] = (tmp["timestamp"] - tmp["_target"]).abs()

            selected = (
                tmp.sort_values(["_bin", "_distance"])
                .groupby("_bin", sort=True)["timestamp"]
                .first()
                .reset_index(drop=True)
            )

        else:
            raise ValueError("profile_selection must be 'first', 'last', or 'nearest'.")

        ts = selected.drop_duplicates().sort_values().reset_index(drop=True)

    if max_profiles is not None and len(ts) > max_profiles:
        idx = np.linspace(0, len(ts) - 1, int(max_profiles), dtype=int)
        ts = ts.iloc[idx].reset_index(drop=True)

    return ts.to_numpy()


def _prepare_xlims(xlims):
    """Normalize x limits to a list with three axis limits."""

    if isinstance(xlims, tuple):
        if len(xlims) != 2:
            raise ValueError("If xlims is a tuple, it must be (xmin, xmax).")
        return [xlims, xlims, xlims]

    if isinstance(xlims, list):
        if len(xlims) != 3:
            raise ValueError("If xlims is a list, it must contain three (xmin, xmax) tuples.")

        for item in xlims:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("Each xlims item must be a tuple (xmin, xmax).")

        return xlims

    raise ValueError("xlims must be a tuple or a list of three tuples.")


def _signed_resultant(d_1: np.ndarray, d_2: np.ndarray) -> np.ndarray:
    """Compute resultant with sign inherited from component 1."""
    d_resultant = np.sqrt(d_1**2 + d_2**2)
    return d_resultant #* np.where(d_1 < 0, -1.0, 1.0)


def _format_date_axes(axs):
    """Format date axes in a compact way."""

    locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
    formatter = mdates.DateFormatter("%m-%d %H")

    for ax in axs:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        plt.setp(
            ax.get_xticklabels(),
            rotation=30,
            ha="right",
            va="top",
            rotation_mode="anchor",
        )
        ax.set_xlabel("Fecha")


def plot_cumulative_displacement_profiles(
    df_disp: pd.DataFrame,
    depths_m=None,
    valid_sensors: Mapping[str, int | bool] | None = None,
    components: Components = "ab",
    timestamp_col: str = "timestamp",
    sensor_col: str = "sensor_id",
    station_col: str | None = "estacion_id",
    station_id=None,
    profile_freq: str | None = None,
    profile_selection: str = "first",
    max_profiles: int | None = None,
    xlims=(-50, 50),
    cmap="viridis",
    figsize=(7.5, 7.5),
    title: str | None = None,
    show_invalid: bool = True,
    invalid_label: str = "Sensores en revisión",
):
    """
    Plot cumulative displacement profiles over time.

    `depths_m` is required unless df_disp already contains `depth_m`.

    profile_freq controls how often a profile curve is plotted:
        "6h", "12h", "1D", "7D", etc.

    Returns
    -------
    matplotlib.figure.Figure
    """

    data = df_disp.copy()

    for col in [timestamp_col, sensor_col]:
        if col not in data.columns:
            raise ValueError(f"Column {col!r} does not exist in df_disp.")

    data[timestamp_col] = pd.to_datetime(data[timestamp_col])
    data[sensor_col] = data[sensor_col].astype(str)

    if station_col is not None and station_col in data.columns and station_id is not None:
        data = data[data[station_col] == station_id].copy()

    if data.empty:
        raise ValueError("df_disp is empty after filtering.")

    if "_sensor_order" not in data.columns:
        data["_sensor_order"] = data[sensor_col].map(sensor_id_to_order)

    comp_1_col, comp_2_col, labels = _get_component_columns(components)

    required_cols = [timestamp_col, sensor_col, "_sensor_order", comp_1_col, comp_2_col]
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Column {col!r} does not exist in df_disp.")

    declared_sensor_ids, valid_sensor_ids, invalid_sensor_ids = _get_declared_valid_invalid_sensors(
        data=data,
        sensor_col=sensor_col,
        valid_sensors=valid_sensors,
    )

    depth_map = build_depth_map(
        sensor_ids=declared_sensor_ids,
        depths_m=depths_m,
        data=data,
        sensor_col=sensor_col,
    )

    data = data[data[sensor_col].isin(valid_sensor_ids)].copy()

    if data.empty:
        raise ValueError("No valid sensors remain to plot.")

    data["depth_m"] = data[sensor_col].map(depth_map)

    timestamps = _select_timestamps(
        timestamps=data[timestamp_col],
        profile_freq=profile_freq,
        profile_selection=profile_selection,
        max_profiles=max_profiles,
    )

    if len(timestamps) == 0:
        raise ValueError("No timestamps available to plot.")

    cmap_obj = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
    time_numbers = mdates.date2num(pd.to_datetime(timestamps))

    if len(time_numbers) == 1:
        vmin = time_numbers[0] - 0.5
        vmax = time_numbers[0] + 0.5
    else:
        vmin = np.nanmin(time_numbers)
        vmax = np.nanmax(time_numbers)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    fig, axs = plt.subplots(
        1,
        3,
        figsize=figsize,
        sharex=False,
        sharey=True,
        layout="constrained",
    )

    for ts in timestamps:
        ts = pd.Timestamp(ts)

        profile = data[data[timestamp_col] == ts].sort_values("depth_m").copy()

        if profile.empty:
            continue

        d_1 = profile[comp_1_col].to_numpy(dtype=float)
        d_2 = profile[comp_2_col].to_numpy(dtype=float)
        depths = profile["depth_m"].to_numpy(dtype=float)
        d_resultant = _signed_resultant(d_1, d_2)

        color = cmap_obj(norm(mdates.date2num(ts)))

        kw = dict(marker="", ls="-", color=color, lw=1.5, alpha=1.0)

        axs[0].plot(d_1, depths, **kw)
        axs[1].plot(d_2, depths, **kw)
        axs[2].plot(d_resultant, depths, **kw)

    axs[0].invert_yaxis()
    xlims = _prepare_xlims(xlims)

    invalid_depths = np.array([], dtype=float)
    if show_invalid and invalid_sensor_ids:
        invalid_depths = np.asarray(
            [depth_map[sid] for sid in invalid_sensor_ids if sid in depth_map],
            dtype=float,
        )
        zeros_invalid = np.zeros_like(invalid_depths)

    for ax, label, xlim in zip(axs, labels, xlims):
        if show_invalid and len(invalid_depths) > 0:
            ax.scatter(
                zeros_invalid,
                invalid_depths,
                marker="o",
                color="k",
                facecolor="none",
                zorder=10,
            )

        ax.set_xlim(xlim)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))
        ax.grid(True, ls="--", lw=0.5, alpha=0.6)
        ax.xaxis.set_label_position("top")
        ax.xaxis.set_ticks_position("top")
        ax.tick_params(which="both", top=True, bottom=False, width=1.2)
        ax.spines[["left", "top"]].set_linewidth(1.5)
        ax.spines[["right", "bottom"]].set_visible(False)
        ax.axvline(0.0, color="k", lw=1.2, ls="--", zorder=0)
        ax.set_xlabel(f"Desp. {label} [mm]")
        ax.set_facecolor("whitesmoke")
    axs[-1].set_xlim((0.1* xlim[0], xlim[1]))
    axs[0].set_ylabel("Profundidad [m]")

    sm = mpl.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=axs, orientation="horizontal", fraction=0.04, pad=0.08)
    cbar.set_label("Fecha")

    locator = mdates.AutoDateLocator()
    cbar.ax.xaxis.set_major_locator(locator)
    cbar.ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    plt.setp(
        cbar.ax.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    if title is not None:
        fig.suptitle(title, fontweight="bold")

    if show_invalid and len(invalid_depths) > 0:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="k",
                linestyle="None",
                fillstyle="none",
                label=invalid_label,
            )
        ]

        fig.legend(
            handles=handles,
            loc="lower right",
            bbox_to_anchor=(1.0, 0.17),
            ncol=1,
            frameon=False,
            handletextpad=0.2,
        )

    return fig


def plot_single_displacement_profile(
    df_disp: pd.DataFrame,
    timestamp=None,
    depths_m=None,
    valid_sensors: Mapping[str, int | bool] | None = None,
    components: Components = "ab",
    timestamp_col: str = "timestamp",
    sensor_col: str = "sensor_id",
    station_col: str | None = "estacion_id",
    station_id=None,
    method: Literal["exact", "nearest"] = "nearest",
    xlims=(-50, 50),
    figsize=(7.5, 7.5),
    title: str | None = None,
):
    """
    Plot a single cumulative displacement profile.

    `depths_m` is required unless df_disp already contains `depth_m`.

    Returns
    -------
    matplotlib.figure.Figure
    """

    data = df_disp.copy()
    data[timestamp_col] = pd.to_datetime(data[timestamp_col])
    data[sensor_col] = data[sensor_col].astype(str)

    if station_col is not None and station_col in data.columns and station_id is not None:
        data = data[data[station_col] == station_id].copy()

    if data.empty:
        raise ValueError("df_disp is empty after filtering.")

    if timestamp is None:
        selected_timestamp = data[timestamp_col].max()
    else:
        target = pd.Timestamp(timestamp)

        if method == "exact":
            selected_timestamp = target

        elif method == "nearest":
            available_times = data[timestamp_col].drop_duplicates().sort_values()
            if available_times.empty:
                raise ValueError("No timestamps available.")
            idx = np.argmin(np.abs(available_times - target))
            selected_timestamp = available_times.iloc[int(idx)]

        else:
            raise ValueError("method must be 'exact' or 'nearest'.")

    comp_1_col, comp_2_col, labels = _get_component_columns(components)

    declared_sensor_ids, valid_sensor_ids, invalid_sensor_ids = _get_declared_valid_invalid_sensors(
        data=data,
        sensor_col=sensor_col,
        valid_sensors=valid_sensors,
    )

    depth_map = build_depth_map(
        sensor_ids=declared_sensor_ids,
        depths_m=depths_m,
        data=data,
        sensor_col=sensor_col,
    )

    profile = (
        data[
            (data[timestamp_col] == selected_timestamp)
            & (data[sensor_col].isin(valid_sensor_ids))
        ]
        .copy()
    )

    if profile.empty:
        raise ValueError("No valid sensor data found for the selected timestamp.")

    profile["depth_m"] = profile[sensor_col].map(depth_map)
    profile = profile.sort_values("depth_m")

    d_1 = profile[comp_1_col].to_numpy(dtype=float)
    d_2 = profile[comp_2_col].to_numpy(dtype=float)
    depths = profile["depth_m"].to_numpy(dtype=float)
    d_resultant = _signed_resultant(d_1, d_2)

    fig, axs = plt.subplots(
        1,
        3,
        figsize=figsize,
        sharex=False,
        sharey=True,
        layout="constrained",
    )

    for ax, x, label in zip(axs, [d_1, d_2, d_resultant], labels):
        ax.plot(x, depths, marker="o", ls="-", lw=1.2, ms=3, color="black", mfc="white")
        ax.set_xlim(xlims)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4))
        ax.grid(True, ls="--", lw=0.5, alpha=0.6)
        ax.xaxis.set_label_position("top")
        ax.xaxis.set_ticks_position("top")
        ax.tick_params(which="both", top=True, bottom=False, width=1.2)
        ax.spines[["left", "top"]].set_linewidth(1.5)
        ax.spines[["right", "bottom"]].set_visible(False)
        ax.axvline(0.0, color="k", lw=1.2, ls="--", zorder=0)
        ax.set_xlabel(f"Desp. {label} [mm]")
        ax.set_facecolor("whitesmoke")

    axs[0].invert_yaxis()
    axs[0].set_ylabel("Profundidad [m]")

    if title is None:
        title = f"Perfil de desplazamiento — {selected_timestamp}"

    fig.suptitle(title, fontweight="bold")

    return fig


def plot_sensor_displacement_evolution(
    df_disp: pd.DataFrame,
    sensor,
    depths_m=None,
    valid_sensors: Mapping[str, int | bool] | None = None,
    components: Components = "ab",
    timestamp_col: str = "timestamp",
    sensor_col: str = "sensor_id",
    station_col: str | None = "estacion_id",
    station_id=None,
    profile_freq: str | None = None,
    profile_selection: str = "first",
    max_points: int | None = None,
    ylims=(-50, 50),
    xlims_ab=(-50, 50),
    circle_radii=(10, 20, 30),
    cmap="viridis",
    figsize=(7.5, 7.5),
    title: str | None = None,
):
    """
    Plot time evolution of cumulative displacement for one sensor.

    The figure includes:
    - component 1 vs time;
    - component 2 vs time;
    - signed resultant vs time;
    - component 2 vs component 1 trajectory.

    Returns
    -------
    matplotlib.figure.Figure
    """

    data = df_disp.copy()

    for col in [timestamp_col, sensor_col]:
        if col not in data.columns:
            raise ValueError(f"Column {col!r} does not exist in df_disp.")

    data[timestamp_col] = pd.to_datetime(data[timestamp_col])
    data[sensor_col] = data[sensor_col].astype(str)

    sensor_id = f"{sensor}a" if isinstance(sensor, int) else str(sensor)

    if station_col is not None and station_col in data.columns and station_id is not None:
        data = data[data[station_col] == station_id].copy()

    if data.empty:
        raise ValueError("df_disp is empty after filtering.")

    comp_1_col, comp_2_col, labels = _get_component_columns(components)

    for col in [comp_1_col, comp_2_col]:
        if col not in data.columns:
            raise ValueError(f"Column {col!r} does not exist in df_disp.")

    declared_sensor_ids, valid_sensor_ids, invalid_sensor_ids = _get_declared_valid_invalid_sensors(
        data=data,
        sensor_col=sensor_col,
        valid_sensors=valid_sensors,
    )

    if sensor_id not in declared_sensor_ids:
        raise ValueError(f"Sensor {sensor_id!r} is not declared in valid_sensors/data.")

    if sensor_id in invalid_sensor_ids:
        raise ValueError(f"Sensor {sensor_id!r} is marked as invalid.")

    depth_map = build_depth_map(
        sensor_ids=declared_sensor_ids,
        depths_m=depths_m,
        data=data,
        sensor_col=sensor_col,
    )

    depth = depth_map[sensor_id]

    sensor_data = (
        data[data[sensor_col] == sensor_id]
        .sort_values(timestamp_col)
        .copy()
    )

    if sensor_data.empty:
        raise ValueError(f"No data found for sensor {sensor_id!r}.")

    selected_timestamps = _select_timestamps(
        timestamps=sensor_data[timestamp_col],
        profile_freq=profile_freq,
        profile_selection=profile_selection,
        max_profiles=max_points,
    )

    sensor_data = sensor_data[sensor_data[timestamp_col].isin(selected_timestamps)].copy()

    if sensor_data.empty:
        raise ValueError("No data remains after temporal selection.")

    timestamps = sensor_data[timestamp_col]
    d_1 = sensor_data[comp_1_col].to_numpy(dtype=float)
    d_2 = sensor_data[comp_2_col].to_numpy(dtype=float)
    d_resultant = _signed_resultant(d_1, d_2)

    cmap_obj = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
    time_numbers = mdates.date2num(pd.to_datetime(timestamps))

    if len(time_numbers) == 1:
        vmin = time_numbers[0] - 0.5
        vmax = time_numbers[0] + 0.5
    else:
        vmin = np.nanmin(time_numbers)
        vmax = np.nanmax(time_numbers)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    colors = cmap_obj(norm(time_numbers))

    fig, axs = plt.subplots(
        2,
        2,
        figsize=figsize,
        sharex=False,
        sharey=False,
        layout="constrained",
    )

    kw = dict(s=25, ec="k", lw=0.2, alpha=0.75, marker="o", c=colors)

    axs[0, 0].scatter(timestamps, d_1, **kw)
    axs[0, 0].set_title(f"Sentido {labels[0]}")
    axs[0, 0].set_ylim(ylims)

    axs[1, 0].scatter(timestamps, d_2, **kw)
    axs[1, 0].set_title(f"Sentido {labels[1]}")
    axs[1, 0].set_ylim(ylims)

    axs[1, 1].scatter(timestamps, d_resultant, **kw)
    axs[1, 1].set_title("Resultante")
    axs[1, 1].set_ylim((0.1 * ylims[0], ylims[1]))

    axs[0, 1].scatter(d_2, d_1, **kw)
    axs[0, 1].set_title(f"{labels[0]} vs {labels[1]}")
    axs[0, 1].set_xlabel(f"Desp. {labels[1]} [mm]")
    axs[0, 1].set_ylabel(f"Desp. {labels[0]} [mm]")
    axs[0, 1].set_xlim(xlims_ab)
    axs[0, 1].set_ylim(xlims_ab)
    axs[0, 1].set_aspect("equal", adjustable="box")

    line_styles = ["-", "--", ":"]
    for radius, ls in zip(circle_radii, line_styles):
        circle = plt.Circle((0, 0), radius, fill=False, color="k", linestyle=ls, label=f"{radius:g} mm")
        axs[0, 1].add_patch(circle)

    axs[0, 1].legend(
        loc="center left",
        frameon=False,
        bbox_to_anchor=(1.0, 0.5),
        title="Radio",
        handlelength=1.0,
    )

    for ax in axs.flat:
        ax.grid(True, ls="--", lw=0.5, alpha=0.6)
        ax.spines[["left", "bottom"]].set_linewidth(1.5)
        ax.spines[["right", "top"]].set_visible(False)
        ax.axhline(0.0, color="k", lw=1.2, ls="--", zorder=0)
        ax.set_facecolor("whitesmoke")

    axs[0, 1].axvline(0.0, color="k", lw=1.2, ls="--", zorder=0)

    _format_date_axes([axs[0, 0], axs[1, 0], axs[1, 1]])

    sm = mpl.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=axs, orientation="horizontal", fraction=0.05, pad=0.08)
    cbar.set_label("Fecha")

    cbar_locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    cbar.ax.xaxis.set_major_locator(cbar_locator)
    cbar.ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S"))

    plt.setp(cbar.ax.get_xticklabels(), rotation=45, ha="right", va="top", rotation_mode="anchor")

    if title is None:
        title = f"Desplazamientos acumulados - Sensor {sensor_id} ({depth:.1f} m)"

    fig.suptitle(title, fontweight="bold")

    return fig


def plot_angle_correction_over_time(
    df_corr: pd.DataFrame,
    field: Literal["roll", "pitch", "yaw"] = "pitch",
    depths_m=None,
    valid_sensors: Mapping[str, int | bool] | None = None,
    timestamp_col: str = "timestamp",
    sensor_col: str = "sensor_id",
    station_col: str | None = "estacion_id",
    station_id=None,
    original_col: str | None = None,
    corrected_col: str | None = None,
    profile_freq: str | None = None,
    profile_selection: str = "first",
    max_points: int | None = None,
    reference: Literal["first", "median"] = "first",
    delta_angle: float | None = None,
    angle_scale: float = 1.0,
    cmap="viridis",
    figsize=(5.5, 5.5),
    title: str | None = None,
    invalid_label: str = "Sensores en revisión",
):
    """
    Plot temporal angle variation before and after drift correction.

    This is the new version of the stacked angle plot used in the notebook.

    The vertical baseline is each sensor depth. The plotted curves are:

        depth + angle_scale * (angle - reference_angle)

    Parameters
    ----------
    df_corr : pandas.DataFrame
        Long DataFrame containing original and corrected angle columns.

    field : {'roll', 'pitch', 'yaw'}
        Angle to plot.

    depths_m : dict, Series, array-like, or None
        Sensor depths. If None, the function tries to use `depth_m` from df_corr.

    valid_sensors : dict or None
        Declared/valid sensors.

    original_col : str or None
        Defaults to f"{field}_deg".

    corrected_col : str or None
        Defaults to f"{field}_corr_deg" if available; otherwise f"{field}_corrected_deg".

    profile_freq : str or None
        Optional temporal thinning, e.g. "6h", "12h", "1D".

    reference : {'first', 'median'}
        Reference angle used to plot variations.

    delta_angle : float or None
        If provided, draws a horizontal strip around each sensor baseline with
        half-height angle_scale * delta_angle.

    angle_scale : float
        Visual scaling factor. With angle_scale=1.0, one degree is plotted as
        one vertical depth unit. This is a visualization offset, not a physical
        depth conversion.

    Returns
    -------
    matplotlib.figure.Figure
    """

    data = df_corr.copy()

    for col in [timestamp_col, sensor_col]:
        if col not in data.columns:
            raise ValueError(f"Column {col!r} does not exist in df_corr.")

    data[timestamp_col] = pd.to_datetime(data[timestamp_col])
    data[sensor_col] = data[sensor_col].astype(str)

    if station_col is not None and station_col in data.columns and station_id is not None:
        data = data[data[station_col] == station_id].copy()

    if data.empty:
        raise ValueError("df_corr is empty after filtering.")

    if original_col is None:
        original_col = f"{field}_deg"

    if corrected_col is None:
        preferred = f"{field}_corr_deg"
        fallback = f"{field}_corrected_deg"

        if preferred in data.columns:
            corrected_col = preferred
        elif fallback in data.columns:
            corrected_col = fallback
        else:
            raise ValueError(
                f"Could not infer corrected column. Expected {preferred!r} or {fallback!r}."
            )

    for col in [original_col, corrected_col]:
        if col not in data.columns:
            raise ValueError(f"Column {col!r} does not exist in df_corr.")

    declared_sensor_ids, valid_sensor_ids, invalid_sensor_ids = _get_declared_valid_invalid_sensors(
        data=data,
        sensor_col=sensor_col,
        valid_sensors=valid_sensors,
    )

    depth_map = build_depth_map(
        sensor_ids=declared_sensor_ids,
        depths_m=depths_m,
        data=data,
        sensor_col=sensor_col,
    )

    valid_sensor_ids = [sid for sid in valid_sensor_ids if sid in data[sensor_col].unique()]
    invalid_sensor_ids = [sid for sid in invalid_sensor_ids if sid in depth_map]

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    cmap_obj = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
    sensor_colors = cmap_obj(np.linspace(0.05, 0.95, max(len(valid_sensor_ids), 2)))

    for i, sensor_id in enumerate(_sort_sensor_ids(valid_sensor_ids)):
        sensor_data = (
            data[data[sensor_col] == sensor_id]
            .sort_values(timestamp_col)
            .copy()
        )

        if sensor_data.empty:
            continue

        selected_timestamps = _select_timestamps(
            timestamps=sensor_data[timestamp_col],
            profile_freq=profile_freq,
            profile_selection=profile_selection,
            max_profiles=max_points,
        )

        sensor_data = sensor_data[sensor_data[timestamp_col].isin(selected_timestamps)].copy()

        if sensor_data.empty:
            continue
        
        baseline = int(sensor_id[:-1])
        if reference == "first":
            ref_angle = float(sensor_data[original_col].dropna().iloc[0])
        elif reference == "median":
            ref_angle = float(sensor_data[original_col].median())
        else:
            raise ValueError("reference must be 'first' or 'median'.")

        if delta_angle is not None:
            ax.axhspan(
                baseline - angle_scale * delta_angle,
                baseline + angle_scale * delta_angle,
                color="0.75",
                alpha=0.3,
                zorder=0,
            )

        y_original = baseline + angle_scale * (sensor_data[original_col].to_numpy(dtype=float) - ref_angle)
        y_corrected = baseline + angle_scale * (sensor_data[corrected_col].to_numpy(dtype=float) - ref_angle)

        ax.plot(
            sensor_data[timestamp_col],
            y_original,
            ls="--",
            lw=2,
            color=sensor_colors[i],
            alpha=1,
            zorder=2,
        )

        ax.plot(
            sensor_data[timestamp_col],
            y_corrected,
            ls="-",
            lw=1.0,
            color="black",
            alpha=1,
            zorder=3,
        )

    ax.invert_yaxis()
    ax.set_xlabel("Fecha")
    ax.set_ylabel("ID sensor")
    ax.set_yticks(range(1, len(declared_sensor_ids) + 1))
    ax.set_title(title or f"Variación temporal de {field}: original vs corregida", fontweight="bold")

    ax.grid(True, ls="--", lw=0.5, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_linewidth(1.3)
    ax.tick_params(width=1.3)

    locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)

    legend_handles = [
        Line2D([0], [0], color="k", lw=2.0, ls="--", label="Original"),
        Line2D([0], [0], color="k", lw=1.5, ls="-", label="Corregida"),
    ]

    if delta_angle is not None:
        legend_handles.insert(
            0,
            Patch(
                facecolor="0.75",
                edgecolor="none",
                alpha=0.3,
                label=f"Franja ±{delta_angle:g}°",
            ),
        )

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=min(len(legend_handles), 3),
        frameon=False,
    )

    return fig
