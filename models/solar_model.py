"""
Solar PV generation model for a collocated datacenter microgrid.

Physics
-------
The model follows the governing equation from the methodology document:

    P_out = P_DC × (POA / 1000) × [1 + γ × (T_cell − 25)] × η_sys

Where:
    P_DC    : Installed DC capacity in MW          (decision variable for optimizer)
    POA     : Plane-of-Array irradiance  W/m²      (computed via pvlib transposition)
    γ       : Temperature coefficient   -0.0035/°C  (N-type high-efficiency cells, 2026)
    T_cell  : Cell temperature  °C                  (T_amb + POA × 0.025)
    η_sys   : System derate factor  0.85            (inverter losses, soiling, wiring)

Since P_DC is the optimizer's decision variable, this module computes a
normalized *capacity factor* timeseries:

    solar_cf_t  =  (POA_t / 1000) × [1 + γ × (T_cell_t − 25)] × η_sys

So the dispatcher can trivially compute:  P_solar_t  =  P_DC  ×  solar_cf_t

Array configuration (fixed-tilt, south-facing — optimal for annual energy):
    Tilt    = latitude   (32.78° for Dallas/Fort Worth)
    Azimuth = 180°       (true south in the northern hemisphere)
    Albedo  = 0.20       (typical mixed terrain / suburban)

POA Transposition:
    Uses pvlib's isotropic sky model: POA = beam + sky_diffuse + ground_reflected
    Beam component accounts for the angle of incidence on the tilted surface.
    pvlib's SPA algorithm handles solar geometry (declination, hour angle, zenith).
"""

import numpy as np
import pandas as pd
import pvlib

# ---------------------------------------------------------------------------
# Physics constants (from methodology document)
# ---------------------------------------------------------------------------

GAMMA: float   = -0.0035   # temperature coefficient [/°C]  (-0.35 %/°C for N-type)
ETA_SYS: float = 0.85      # system derate factor (inverter + soiling + wiring)

# Array geometry — fixed-tilt, south-facing at latitude-optimal tilt
TILT: float    = 32.78     # surface tilt [degrees]  = DFW latitude
AZIMUTH: float = 180.0     # surface azimuth [degrees, N=0, E=90, S=180, W=270]
ALBEDO: float  = 0.20      # ground reflectance (dimensionless)

# DFW elevation (used for more accurate solar position & air mass)
ELEVATION_M: float = 137.0  # metres above sea level


# ---------------------------------------------------------------------------
# Core physics functions
# ---------------------------------------------------------------------------

def compute_poa(
    weather_df: pd.DataFrame,
    lat: float,
    lon: float,
    tilt: float   = TILT,
    azimuth: float = AZIMUTH,
    albedo: float  = ALBEDO,
    elevation: float = ELEVATION_M,
) -> pd.Series:
    """
    Transpose GHI / DNI / DHI to Plane-of-Array (POA) irradiance using pvlib.

    Parameters
    ----------
    weather_df : pd.DataFrame
        Must have a timezone-aware DatetimeIndex and columns: ghi, dhi, dni (W/m²).
    lat, lon   : float  — site coordinates in decimal degrees.
    tilt       : float  — panel tilt angle from horizontal [degrees].
    azimuth    : float  — panel azimuth [degrees, south = 180].
    albedo     : float  — ground reflectance.
    elevation  : float  — site elevation [metres].

    Returns
    -------
    pd.Series  — POA global irradiance [W/m²], same index as weather_df.
    """
    location = pvlib.location.Location(
        latitude=lat,
        longitude=lon,
        tz=weather_df.index.tz,
        altitude=elevation,
    )

    solar_pos = location.get_solarposition(weather_df.index)

    poa_components = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=solar_pos["apparent_zenith"],
        solar_azimuth=solar_pos["azimuth"],
        dni=weather_df["dni"],
        ghi=weather_df["ghi"],
        dhi=weather_df["dhi"],
        albedo=albedo,
        model="isotropic",
    )

    poa = poa_components["poa_global"].fillna(0.0).clip(lower=0.0)
    poa.name = "poa"
    return poa


def compute_cell_temp(temp_c: "np.ndarray | pd.Series", poa: "np.ndarray | pd.Series") -> "np.ndarray | pd.Series":
    """
    Estimate PV cell temperature from the simplified NOCT-like model.

        T_cell = T_ambient + POA × 0.025

    The 0.025 coefficient is derived from the nominal NOCT assumption
    (800 W/m² → +20 °C rise above ambient). Returns same type as input.
    """
    return temp_c + poa * 0.025


def compute_solar_cf(poa: "np.ndarray | pd.Series", tcell: "np.ndarray | pd.Series") -> "np.ndarray | pd.Series":
    """
    Compute normalized solar capacity factor per MW of installed DC capacity.

        solar_cf = (POA / 1000) × [1 + γ × (T_cell − 25)] × η_sys

    At STC (POA = 1000 W/m², T_cell = 25 °C): solar_cf = 1.0 × 1.0 × 0.85 = 0.85
    At night: solar_cf = 0.0

    The dispatcher uses: P_solar_MW_t  =  P_DC_MW  ×  solar_cf_t

    Returns
    -------
    Non-negative array/Series of the same shape as poa.
    """
    cf = (poa / 1000.0) * (1.0 + GAMMA * (tcell - 25.0)) * ETA_SYS
    if isinstance(cf, pd.Series):
        return cf.clip(lower=0.0)
    return np.maximum(cf, 0.0)


# ---------------------------------------------------------------------------
# High-level builder
# ---------------------------------------------------------------------------

def build_solar_timeseries(
    weather_df: pd.DataFrame,
    lat: float,
    lon: float,
    tilt: float    = TILT,
    azimuth: float = AZIMUTH,
) -> pd.DataFrame:
    """
    Build the full hourly solar capacity-factor timeseries from raw weather.

    Parameters
    ----------
    weather_df : pd.DataFrame
        Timezone-aware DatetimeIndex; columns: ghi, dhi, dni [W/m²], temp_c [°C].
    lat, lon   : float — site coordinates.
    tilt       : float — panel tilt [degrees].
    azimuth    : float — panel azimuth [degrees].

    Returns
    -------
    pd.DataFrame with columns:
        ghi         W/m²    Global Horizontal Irradiance (from NSRDB)
        dhi         W/m²    Diffuse Horizontal Irradiance
        dni         W/m²    Direct Normal Irradiance
        temp_c      °C      Ambient temperature
        poa         W/m²    Plane-of-Array irradiance (computed)
        tcell       °C      Estimated cell temperature (computed)
        solar_cf    --      Normalized capacity factor ∈ [0, ~0.85]

    Usage in dispatcher:
        P_solar_t [MW]  =  P_DC [MW]  ×  solar_cf_t
    """
    required = {"ghi", "dhi", "dni", "temp_c"}
    missing = required - set(weather_df.columns)
    if missing:
        raise ValueError(f"weather_df missing columns: {missing}")

    df = weather_df[["ghi", "dhi", "dni", "temp_c"]].copy()

    df["poa"]      = compute_poa(df, lat, lon, tilt=tilt, azimuth=azimuth)
    df["tcell"]    = compute_cell_temp(df["temp_c"], df["poa"])
    df["solar_cf"] = compute_solar_cf(df["poa"], df["tcell"])

    return df


def solar_summary(solar_df: pd.DataFrame) -> dict:
    """Compute key statistics for a solar capacity factor timeseries."""
    cf = solar_df["solar_cf"]
    poa = solar_df["poa"]
    daylight = poa > 1.0   # W/m² threshold to exclude numerical night noise
    return {
        "total_hours":          len(cf),
        "daylight_hours":       int(daylight.sum()),
        "peak_poa_wm2":         float(poa.max()),
        "peak_solar_cf":        float(cf.max()),
        "mean_cf_annual":       float(cf.mean()),            # capacity factor (8760/8784 basis)
        "mean_cf_daylight":     float(cf[daylight].mean()),  # during daylight only
        "annual_cf_pct":        float(cf.mean() * 100),
        "peak_hour":            str(cf.idxmax()),
    }
