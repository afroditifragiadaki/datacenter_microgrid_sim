"""
Build the 2024 hourly solar PV capacity-factor timeseries for ERCOT.

Data source (in priority order)
--------------------------------
1. NREL NSRDB PSM3  — https://nsrdb.nrel.gov/
   Resolution : 60-min, 4 km spatial grid, ERA5 + satellite-corrected
   NOTE: Requires separate dataset activation at nsrdb.nrel.gov after
   obtaining an NREL API key. If unavailable, falls back to source 2.

2. Open-Meteo ERA5 archive  — https://open-meteo.com/  (no API key)
   Resolution : 60-min, ~9 km (ERA5 native), same reanalysis backbone
   Variables  : shortwave_radiation (GHI), diffuse_radiation (DHI),
                direct_normal_irradiance (DNI), temperature_2m

Both sources provide equivalent quality for hourly energy modelling.
NSRDB PSM3 adds satellite-derived corrections for the US; ERA5 alone is
the global standard for reanalysis-based solar resource assessment.

API key security
----------------
The NREL API key is loaded from config/secrets.env (gitignored). It is
never printed in full — only the first / last 4 chars appear in logs.

Location
--------
Dallas / Fort Worth, TX  (32.78N, 96.80W) — same grid point as demand model.

Outputs
-------
    data/raw/ercot_nsrdb_<year>.csv        raw NSRDB response (cached, if used)
    data/raw/ercot_openmeteo_solar_<yr>.csv raw Open-Meteo response (cached)
    data/processed/ercot_solar_2024.csv    hourly solar_cf timeseries
    data/processed/ercot_solar_2024.png    diagnostic plots
"""

import io
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.solar_model import (
    build_solar_timeseries,
    solar_summary,
    TILT, AZIMUTH, GAMMA, ETA_SYS, ALBEDO,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LAT           = 32.78
LON           = -96.80
TARGET_YEAR   = 2024
LOCATION_NAME = "Dallas-Fort Worth, TX"

YEAR_CANDIDATES = [2024, 2023, 2022]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# NREL API secrets (gitignored config/secrets.env)
# ---------------------------------------------------------------------------
load_dotenv(PROJECT_ROOT / "config" / "secrets.env")
_API_KEY = os.environ.get("NREL_API_KEY", "")
_EMAIL   = os.environ.get("NREL_EMAIL", "research@microgrid-sim.local")


def _mask_key(key: str) -> str:
    if not key or len(key) < 10:
        return "[NOT SET]"
    return f"{key[:4]}{'*' * (len(key) - 8)}{key[-4:]}"


# ---------------------------------------------------------------------------
# Source 1: NREL NSRDB PSM3
# ---------------------------------------------------------------------------
NSRDB_URL = "https://developer.nrel.gov/api/nsrdb/v2/solar/psm3-download.csv"

COL_RENAME_NSRDB = {
    "GHI":              "ghi",
    "DHI":              "dhi",
    "DNI":              "dni",
    "Air Temperature":  "temp_c",
    "Wind Speed":       "wind_speed",
    "Surface Pressure": "pressure",
}


def _fetch_nsrdb(year: int) -> str:
    """Try to download PSM3 CSV for the given year. Returns raw text or raises."""
    params = {
        "api_key":       _API_KEY,
        "email":         _EMAIL,
        "lat":           LAT,
        "lon":           LON,
        "names":         str(year),
        "interval":      "60",
        "attributes":    "ghi,dhi,dni,air_temperature,wind_speed,surface_pressure",
        "full_name":     "Microgrid Researcher",
        "affiliation":   "Research",
        "reason":        "Research",
        "mailing_list":  "false",
        "utc":           "false",
    }
    resp = requests.get(NSRDB_URL, params=params, timeout=120)

    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}")

    text = resp.text.strip()
    if text.startswith("{") or text.startswith("["):
        # JSON error body
        try:
            err = resp.json()
            msg = err.get("errors", err.get("error", str(err)))
        except Exception:
            msg = text[:200]
        raise RuntimeError(f"API error: {msg}")

    if not text or "Source" not in text.splitlines()[0]:
        raise RuntimeError(f"Unexpected response: {text[:100]}")

    return resp.text


def _parse_nsrdb_csv(raw_csv: str) -> tuple[pd.DataFrame, dict]:
    lines = raw_csv.splitlines()
    meta_parts = lines[0].split(",")
    meta = {}
    try:
        meta = {
            "source":    meta_parts[0].strip(),
            "latitude":  float(meta_parts[5]),
            "longitude": float(meta_parts[6]),
            "timezone":  float(meta_parts[7]),
            "elevation": float(meta_parts[8]),
            "city":      meta_parts[2].strip(),
            "state":     meta_parts[3].strip(),
        }
    except (IndexError, ValueError):
        pass

    df = pd.read_csv(io.StringIO("\n".join(lines[1:])))

    tz_offset = int(meta.get("timezone", -6))
    tz_str = f"Etc/GMT+{abs(tz_offset)}" if tz_offset < 0 else f"Etc/GMT-{tz_offset}"

    dt_index = pd.to_datetime({"year": df["Year"], "month": df["Month"],
                                "day": df["Day"], "hour": df["Hour"]})
    df.index = dt_index.dt.tz_localize(tz_str)
    df.index.name = "datetime"
    df = df.rename(columns=COL_RENAME_NSRDB)

    keep = [c for c in COL_RENAME_NSRDB.values() if c in df.columns]
    df = df[keep]
    for col in ["ghi", "dhi", "dni"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0.0)

    return df, meta


def _try_nsrdb() -> tuple[pd.DataFrame, dict, int] | None:
    """Attempt NSRDB fetch for all candidate years. Returns None on all failures."""
    if not _API_KEY:
        print("  NSRDB skipped: no API key configured.")
        return None

    print(f"  NREL API key: {_mask_key(_API_KEY)}")
    for year in YEAR_CANDIDATES:
        raw_path = RAW_DIR / f"ercot_nsrdb_{year}.csv"

        if raw_path.exists():
            print(f"  Loading cached NSRDB {year} ...")
            raw_csv = raw_path.read_text(encoding="utf-8")
        else:
            try:
                print(f"  Requesting NSRDB PSM3 year={year} ...")
                raw_csv = _fetch_nsrdb(year)
                raw_path.write_text(raw_csv, encoding="utf-8")
                print(f"  Cached -> {raw_path.name}")
            except RuntimeError as exc:
                print(f"  Year {year} unavailable ({exc})")
                continue

        weather, meta = _parse_nsrdb_csv(raw_csv)
        expected = 8784 if year % 4 == 0 else 8760
        if len(weather) < expected * 0.95:
            print(f"  Only {len(weather)} rows for {year}, expected {expected}. Skipping.")
            continue

        return weather, meta, year

    return None


# ---------------------------------------------------------------------------
# Source 2: Open-Meteo ERA5 (fallback)
# ---------------------------------------------------------------------------
OM_SOLAR_VARS = [
    "shortwave_radiation",       # GHI  (W/m^2)
    "diffuse_radiation",         # DHI  (W/m^2)
    "direct_normal_irradiance",  # DNI  (W/m^2)
    "temperature_2m",            # ambient temp (deg C)
]

COL_RENAME_OM = {
    "shortwave_radiation":      "ghi",
    "diffuse_radiation":        "dhi",
    "direct_normal_irradiance": "dni",
    "temperature_2m":           "temp_c",
}


def _fetch_openmeteo_solar(year: int) -> pd.DataFrame:
    """
    Fetch hourly solar irradiance + temperature from Open-Meteo ERA5 archive.

    Requests data in UTC to avoid all DST ambiguity. pvlib solar position
    calculations work correctly with UTC timestamps.
    Returns a UTC-aware DataFrame with 8784 rows for 2024 (leap year).
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":          LAT,
        "longitude":         LON,
        "start_date":        f"{year}-01-01",
        "end_date":          f"{year}-12-31",
        "hourly":            ",".join(OM_SOLAR_VARS),
        "timezone":          "UTC",          # clean UTC, no DST issues
        "temperature_unit":  "celsius",
        "wind_speed_unit":   "ms",
    }
    print(f"  Requesting Open-Meteo ERA5 solar for {year} (UTC) ...")
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    times = pd.to_datetime(data["hourly"]["time"], utc=True)
    df = pd.DataFrame(
        {var: data["hourly"][var] for var in OM_SOLAR_VARS},
        index=times,
    )
    df.index.name = "datetime"
    df = df.rename(columns=COL_RENAME_OM)

    # Clip irradiance to non-negative (ERA5 may have tiny negatives at dusk/dawn)
    for col in ["ghi", "dhi", "dni"]:
        df[col] = df[col].clip(lower=0.0)

    return df


def _load_or_fetch_openmeteo(year: int) -> pd.DataFrame:
    raw_path = RAW_DIR / f"ercot_openmeteo_solar_{year}.csv"
    if raw_path.exists():
        print(f"  Loading cached Open-Meteo solar {year} ...")
        df = pd.read_csv(raw_path, index_col="datetime", parse_dates=True)
        # Re-attach timezone if stripped by CSV round-trip
        if df.index.tz is None:
            df.index = df.index.tz_localize("Etc/GMT+6")
        return df

    df = _fetch_openmeteo_solar(year)
    df.to_csv(raw_path)
    print(f"  Cached -> {raw_path.name}")
    return df


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------
def load_weather() -> tuple[pd.DataFrame, dict, str, int]:
    """
    Load irradiance + temperature from the best available source.
    Returns (weather_df, meta, source_label, year_used).
    """
    # --- Try NSRDB first ---
    result = _try_nsrdb()
    if result is not None:
        weather, meta, year = result
        if year != TARGET_YEAR:
            print(f"  NOTE: Using NSRDB {year} irradiance profiles (2024 not yet published).")
        return weather, meta, "NSRDB PSM3", year

    # --- Fallback: Open-Meteo ERA5 ---
    print(f"\n  Falling back to Open-Meteo ERA5 (ERA5 reanalysis, same backbone as NSRDB PSM3).")
    print(f"  To enable NSRDB: activate PSM3 dataset access at https://nsrdb.nrel.gov/\n")

    weather = _load_or_fetch_openmeteo(TARGET_YEAR)
    meta = {
        "latitude":  LAT,
        "longitude": LON,
        "elevation": 137.0,
        "city":      "Dallas-Fort Worth",
        "state":     "TX",
        "source":    "Open-Meteo ERA5",
        "timezone":  0,    # UTC
    }
    return weather, meta, "Open-Meteo ERA5", TARGET_YEAR


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_solar(solar: pd.DataFrame, source_label: str, weather_year: int) -> Path:
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(
        f"ERCOT Solar PV Model  |  {LOCATION_NAME}  |  Source: {source_label} {weather_year}\n"
        f"Array: {TILT:.0f}deg tilt (latitude-optimal), South-facing  |  "
        f"gamma={GAMMA*100:.2f}%/C  eta_sys={ETA_SYS}",
        fontsize=11, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax_poa  = fig.add_subplot(gs[0, :])
    ax_cf   = fig.add_subplot(gs[1, :])
    ax_week = fig.add_subplot(gs[2, 0])
    ax_mon  = fig.add_subplot(gs[2, 1])

    MFMT = mdates.DateFormatter("%b")
    MLOC = mdates.MonthLocator()
    C1, C2 = "#e5c07b", "#98c379"

    # Panel 1: POA
    ax_poa.fill_between(solar.index, solar["poa"], alpha=0.35, color=C1)
    ax_poa.plot(solar.index, solar["poa"], color=C1, lw=0.4)
    ax_poa.set_ylabel("POA (W/m2)")
    ax_poa.set_title("Plane-of-Array Irradiance — Fixed Tilt, South-Facing")
    ax_poa.xaxis.set_major_formatter(MFMT); ax_poa.xaxis.set_major_locator(MLOC)
    ax_poa.grid(True, alpha=0.25)

    # Panel 2: Capacity factor
    ax_cf.fill_between(solar.index, solar["solar_cf"], alpha=0.35, color=C2)
    ax_cf.plot(solar.index, solar["solar_cf"], color=C2, lw=0.4)
    ax_cf.axhline(ETA_SYS, ls="--", color="#abb2bf", lw=0.9,
                  label=f"STC ceiling (eta_sys = {ETA_SYS})")
    ax_cf.set_ylabel("Capacity Factor (--)")
    ax_cf.set_ylim(-0.02, 0.95)
    ax_cf.set_title("Solar Capacity Factor  |  P_solar [MW] = P_DC [MW] x CF")
    ax_cf.legend(fontsize=8); ax_cf.xaxis.set_major_formatter(MFMT)
    ax_cf.xaxis.set_major_locator(MLOC); ax_cf.grid(True, alpha=0.25)

    # Panel 3: Peak week detail
    weekly_poa = solar["poa"].resample("W").mean()
    peak_week_start = weekly_poa.idxmax() - pd.Timedelta(days=3)
    peak_week_end   = peak_week_start + pd.Timedelta(days=7)
    week = solar.loc[peak_week_start:peak_week_end]
    ax_week.fill_between(week.index, week["solar_cf"], alpha=0.4, color=C2)
    ax_week.plot(week.index, week["solar_cf"], color=C2, lw=1.2)
    ax_week.set_ylabel("Capacity Factor (--)")
    ax_week.set_title(f"Peak Week: {peak_week_start.strftime('%b %d')} - "
                      f"{peak_week_end.strftime('%b %d')}")
    ax_week.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%Hh"))
    ax_week.xaxis.set_major_locator(mdates.DayLocator()); ax_week.grid(True, alpha=0.25)

    # Panel 4: Monthly CF bar chart
    monthly_cf = solar["solar_cf"].resample("ME").mean()
    months = [t.strftime("%b") for t in monthly_cf.index]
    bars = ax_mon.bar(months, monthly_cf.values, color=C2, alpha=0.75, edgecolor="white")
    for bar, val in zip(bars, monthly_cf.values):
        ax_mon.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    ax_mon.set_ylabel("Mean Capacity Factor (--)")
    ax_mon.set_title("Monthly Average Capacity Factor")
    ax_mon.grid(True, axis="y", alpha=0.3); ax_mon.set_ylim(0, 0.35)

    plot_path = OUT_DIR / f"ercot_solar_{TARGET_YEAR}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved -> {plot_path}")
    plt.show()
    return plot_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> pd.DataFrame:
    print("=" * 60)
    print("ERCOT Solar PV Timeseries Builder")
    print("=" * 60)

    # 1. Load weather (NSRDB or Open-Meteo ERA5)
    print(f"\n[1/4] Loading solar irradiance data ...")
    weather, meta, source_label, weather_year = load_weather()

    print(f"  Source           : {source_label}")
    print(f"  Location         : {meta.get('city','?')}, {meta.get('state','?')}")
    print(f"  Grid point       : {meta.get('latitude', LAT):.4f}N, "
          f"{abs(meta.get('longitude', LON)):.4f}W")
    print(f"  Elevation        : {meta.get('elevation','?')} m")
    print(f"  Rows loaded      : {len(weather)} ({len(weather)/24:.0f} days)")

    if weather.isna().any().any():
        n_nan = int(weather.isna().sum().sum())
        print(f"  Interpolating {n_nan} missing values ...")
        weather = weather.interpolate(method="time")

    # 2. Build solar timeseries
    print(f"\n[2/4] Computing solar capacity factor ...")
    print(f"  Tilt     : {TILT}deg  (= site latitude, annual-optimal)")
    print(f"  Azimuth  : {AZIMUTH}deg  (south-facing)")
    print(f"  Albedo   : {ALBEDO}")
    print(f"  gamma    : {GAMMA}/degC  ({GAMMA*100:.2f}%/degC, N-type cells)")
    print(f"  eta_sys  : {ETA_SYS}  (inverter + soiling + wiring derate)")

    solar = build_solar_timeseries(
        weather,
        lat=meta.get("latitude", LAT),
        lon=meta.get("longitude", LON),
    )

    # 3. Save
    out_path = OUT_DIR / f"ercot_solar_{TARGET_YEAR}.csv"
    solar.to_csv(out_path)
    print(f"\n[3/4] Saved -> {out_path}")

    # 4. Summary
    print(f"\n[4/4] Summary statistics")
    print("-" * 50)
    stats = solar_summary(solar)

    print(f"  Total hours            : {stats['total_hours']:,}")
    print(f"  Daylight hours (POA>1) : {stats['daylight_hours']:,} "
          f"({stats['daylight_hours']/stats['total_hours']*100:.1f}%)")
    print(f"  Peak POA               : {stats['peak_poa_wm2']:.0f} W/m2")
    print(f"  Peak capacity factor   : {stats['peak_solar_cf']:.4f} ({stats['peak_solar_cf']*100:.1f}%)")
    print(f"  Peak CF hour           : {stats['peak_hour']}")
    print(f"  Annual capacity factor : {stats['mean_cf_annual']*100:.2f}%")
    print(f"  Mean CF (daylight only): {stats['mean_cf_daylight']*100:.2f}%")

    print(f"\n  Optimizer usage:")
    print(f"    P_solar_t [MW]  =  P_DC [MW]  x  solar_cf_t")
    print(f"    100 MW DC  ->  {100*solar['solar_cf'].sum():,.0f} MWh/year")
    print(f"    150 MW DC  ->  {150*solar['solar_cf'].sum():,.0f} MWh/year")
    print(f"    200 MW DC  ->  {200*solar['solar_cf'].sum():,.0f} MWh/year")
    print(f"    Datacenter demand  :  522,524 MWh/year")

    print(f"\n  Monthly capacity factors:")
    monthly_cf = solar["solar_cf"].resample("ME").mean()
    for ts, cf in monthly_cf.items():
        bar = "#" * int(cf * 200)
        print(f"    {ts.strftime('%b'):3s}  {cf:.4f}  ({cf*100:5.2f}%)  {bar}")

    # Cross-check row count against demand timeseries
    demand_path = OUT_DIR / "ercot_demand_2024.csv"
    if demand_path.exists():
        demand = pd.read_csv(demand_path, index_col="datetime", parse_dates=True)
        print(f"\n  Row count check vs demand timeseries:")
        print(f"    Demand : {len(demand)}")
        print(f"    Solar  : {len(solar)}")
        status = "OK" if len(demand) == len(solar) else "MISMATCH - will align by index in dispatcher"
        print(f"    Status : {status}")

    # 5. Plot
    print("\n  Generating plots ...")
    plot_solar(solar, source_label, weather_year)

    print("\nDone.")
    return solar


if __name__ == "__main__":
    main()
