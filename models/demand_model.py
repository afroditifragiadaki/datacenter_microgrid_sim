"""
Demand-side model for a collocated 50 MW datacenter.

PUE Model
---------
The methodology document defines four discrete cooling regimes:

    T < 10°C   → Free Cooling (Economizer)         PUE = 1.12
    10–25°C    → Standard Liquid Cooling            PUE = 1.20
    25–35°C    → Hybrid / Assisted Cooling          PUE = 1.32
    T > 35°C   → Max Mechanical Chilling            PUE = 1.45

A naive step-function implementation creates discontinuous load jumps at
threshold crossings (e.g., 60.0 MW → 66.0 MW at exactly 25 °C). This is
not physically realistic: datacenter cooling plants transition gradually as
outside-air conditions change.

Fix applied: piecewise-linear interpolation between the boundary temperatures
using the table PUE values as knots. The function is flat (clamped) outside
the defined range. This preserves all four table values at the regime
boundaries while producing smooth, continuous load profiles.

Total Load = IT Load × PUE(T_amb)
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IT_LOAD_MW: float = 50.0  # constant server / IT load (MW)

# Piecewise-linear PUE control points (ambient_temp_°C, PUE).
# The flat segments at each end are implemented by duplicating the boundary
# value at an extreme temperature so np.interp clamps correctly.
_PUE_KNOTS: list[tuple[float, float]] = [
    (-50.0, 1.12),  # floor  – free cooling below 10 °C
    (10.0,  1.12),  # knot 1 – bottom of free-cooling regime
    (25.0,  1.20),  # knot 2 – transition: liquid → hybrid
    (35.0,  1.32),  # knot 3 – transition: hybrid → max chilling
    (60.0,  1.45),  # ceiling – max mechanical chilling above 35 °C
]

_T_KNOTS   = np.array([k[0] for k in _PUE_KNOTS], dtype=float)
_PUE_VALS  = np.array([k[1] for k in _PUE_KNOTS], dtype=float)

# Cooling mode labels for reporting (step boundaries kept for labelling only)
_MODE_THRESHOLDS = [
    (10.0, "Free Cooling (Economizer)"),
    (25.0, "Standard Liquid Cooling"),
    (35.0, "Hybrid / Assisted Cooling"),
    (np.inf, "Max Mechanical Chilling"),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pue_from_temp(temp_c: "np.ndarray | float") -> "np.ndarray | float":
    """
    Compute PUE as a piecewise-linear function of ambient temperature.

    Extrapolation is clamped to 1.12 (below −50 °C) and 1.45 (above 60 °C),
    which covers any realistic location on Earth.

    Parameters
    ----------
    temp_c : float or array-like
        Ambient dry-bulb temperature in °C.

    Returns
    -------
    float or np.ndarray
        PUE value(s), same shape as input.
    """
    return np.interp(temp_c, _T_KNOTS, _PUE_VALS)


def cooling_mode_label(temp_c: float) -> str:
    """Return the dominant cooling-mode label for a given temperature."""
    for threshold, label in _MODE_THRESHOLDS:
        if temp_c < threshold:
            return label
    return _MODE_THRESHOLDS[-1][1]


def build_demand_timeseries(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build an hourly demand timeseries from a weather DataFrame.

    Parameters
    ----------
    weather_df : pd.DataFrame
        Must have a DatetimeIndex and a column ``temp_c`` (ambient °C).
        Any extra columns are ignored.

    Returns
    -------
    pd.DataFrame
        Index     : same DatetimeIndex as input
        Columns:
            temp_c          – ambient temperature (°C)
            pue             – Power Usage Effectiveness (dimensionless)
            cooling_mode    – dominant cooling regime label
            it_load_mw      – IT / server load (constant 50 MW)
            total_load_mw   – facility total load = IT load × PUE (MW)
    """
    if "temp_c" not in weather_df.columns:
        raise ValueError("weather_df must contain a 'temp_c' column.")

    df = weather_df[["temp_c"]].copy()
    df["pue"]           = pue_from_temp(df["temp_c"].to_numpy())
    df["cooling_mode"]  = df["temp_c"].apply(cooling_mode_label)
    df["it_load_mw"]    = IT_LOAD_MW
    df["total_load_mw"] = df["it_load_mw"] * df["pue"]
    return df


def demand_summary(demand_df: pd.DataFrame) -> dict:
    """
    Compute summary statistics for a demand timeseries.

    Returns
    -------
    dict with keys: hours, avg_pue, min_load_mw, max_load_mw,
                    avg_load_mw, annual_energy_mwh, cooling_mode_hours
    """
    d = demand_df
    mode_hours = d["cooling_mode"].value_counts().to_dict()
    return {
        "hours":              len(d),
        "avg_pue":            d["pue"].mean(),
        "min_pue":            d["pue"].min(),
        "max_pue":            d["pue"].max(),
        "min_load_mw":        d["total_load_mw"].min(),
        "max_load_mw":        d["total_load_mw"].max(),
        "avg_load_mw":        d["total_load_mw"].mean(),
        "annual_energy_mwh":  d["total_load_mw"].sum(),   # hourly MW = MWh
        "cooling_mode_hours": mode_hours,
    }
