"""
Build the 2024 hourly solar PV capacity-factor timeseries for ERCOT.

Primary source: NREL PVWatts V8 API
------------------------------------
https://developer.nrel.gov/api/pvwatts/v8.json

PVWatts V8 runs NREL's SSC (System Advisor Model Simulation Core) engine,
which is the gold standard for solar production simulation. It uses NSRDB
TMY (Typical Meteorological Year) data internally — no separate weather
download needed. This replaces the manual physics model + separate weather
fetch with a single, authoritative API call.

Key PVWatts parameters matched to the methodology document:
    module_type = 1  (Premium)   -> γ = -0.35%/°C  (N-type, matches spec)
    array_type  = 0  (Fixed Open Rack, utility scale)
    tilt        = 32.78°         (latitude-optimal for annual energy)
    azimuth     = 180°           (south-facing)
    losses      = 15%            (soiling + wiring + other, ~1 - eta_sys)
    inv_eff     = 96%            (standard utility inverter)
    dc_ac_ratio = 1.2            (standard utility configuration)

Output: solar_cf timeseries
    P_solar_t [MW]  =  P_DC [MW]  x  solar_cf_t
    solar_cf is derived as: ac_output_W / (system_capacity_kW * 1000)
    At STC (1 kW ref): solar_cf = ac_W / 1000

Leap-year handling
------------------
PVWatts returns 8760 TMY hours (non-leap year). The demand timeseries is
2024 (8784 hours, leap year). We expand the TMY to 8784 by duplicating
Feb 28 as Feb 29 — standard practice when mapping TMY to a leap year.

Fallback
--------
If PVWatts is unavailable, falls back to Open-Meteo ERA5 + pvlib transposition
(the original manual physics approach), which gives consistent results.

Outputs
-------
    data/raw/ercot_pvwatts_tmy.json          raw PVWatts response (cached)
    data/raw/ercot_openmeteo_solar_2024.csv  Open-Meteo fallback (cached)
    data/processed/ercot_solar_2024.csv      hourly solar_cf timeseries
    data/processed/ercot_solar_2024.png      diagnostic plots
"""

import io
import json
import os
import sys
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

from models.solar_model import solar_summary, TILT, AZIMUTH, GAMMA, ETA_SYS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LAT           = 32.78
LON           = -96.80
TARGET_YEAR   = 2024
LOCATION_NAME = "Dallas-Fort Worth, TX"

RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# PVWatts parameters — aligned with methodology document
PVWATTS_PARAMS = {
    "system_capacity": 1,      # 1 kW reference: CF = ac_W / 1000
    "module_type":     1,      # Premium (γ = -0.35%/°C, matches N-type spec)
    "array_type":      0,      # Fixed - Open Rack (utility scale)
    "tilt":            TILT,   # 32.78° = latitude-optimal
    "azimuth":         AZIMUTH,# 180° south-facing
    "losses":          15,     # soiling + wiring + other (~1 - eta_sys = 15%)
    "inv_eff":         96.0,   # inverter efficiency %
    "dc_ac_ratio":     1.2,    # standard utility configuration
    "dataset":         "nsrdb",# NSRDB TMY (best available for US)
    "timeframe":       "hourly",
}

# ---------------------------------------------------------------------------
# Secrets (gitignored config/secrets.env)
# ---------------------------------------------------------------------------
load_dotenv(PROJECT_ROOT / "config" / "secrets.env")
_API_KEY = os.environ.get("NREL_API_KEY", "")
_EMAIL   = os.environ.get("NREL_EMAIL", "research@microgrid-sim.local")


def _mask_key(key: str) -> str:
    if not key or len(key) < 10:
        return "[NOT SET]"
    return f"{key[:4]}{'*' * (len(key) - 8)}{key[-4:]}"


# ---------------------------------------------------------------------------
# Source 1: PVWatts V8
# ---------------------------------------------------------------------------
PVWATTS_URL = "https://developer.nrel.gov/api/pvwatts/v8.json"


def _fetch_pvwatts() -> dict:
    """Call PVWatts V8 API and return the parsed JSON response."""
    params = {"api_key": _API_KEY, "lat": LAT, "lon": LON, **PVWATTS_PARAMS}
    print(f"  Calling PVWatts V8 (key: {_mask_key(_API_KEY)}) ...")
    resp = requests.get(PVWATTS_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    errors = data.get("errors", [])
    if errors:
        raise RuntimeError(f"PVWatts API errors: {errors}")

    if "outputs" not in data or "ac" not in data["outputs"]:
        raise RuntimeError(f"Unexpected PVWatts response structure: {list(data.keys())}")

    return data


def _load_or_fetch_pvwatts() -> dict:
    cache_path = RAW_DIR / "ercot_pvwatts_tmy.json"
    if cache_path.exists():
        print(f"  Loading cached PVWatts response ...")
        with cache_path.open(encoding="utf-8") as f:
            return json.load(f)

    data = _fetch_pvwatts()
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"  Cached -> {cache_path.name}")
    return data


def _tmy_8760_to_2024_8784(tmy_array: np.ndarray) -> np.ndarray:
    """
    Expand an 8760-element TMY array to 8784 elements for leap year 2024.

    Strategy: duplicate Feb 28 as Feb 29 (the standard approach for mapping
    TMY to a leap year in energy simulation).

    TMY index layout (0-based hours):
        Jan (31d): hours   0 – 743
        Feb (28d): hours 744 – 1415
          Feb 28:  hours 1392 – 1415  (last 24h of Feb)
        Mar – Dec: hours 1416 – 8759
    """
    feb28_start = (31 + 27) * 24   # = 1392
    feb28_end   = feb28_start + 24  # = 1416
    feb28_vals  = tmy_array[feb28_start:feb28_end]

    expanded = np.concatenate([
        tmy_array[:feb28_end],   # Jan 1 – Feb 28  (hours 0–1415)
        feb28_vals,              # Feb 29 duplicate (hours 1416–1439)
        tmy_array[feb28_end:],  # Mar 1 – Dec 31   (hours 1440–8783)
    ])
    assert len(expanded) == 8784, f"Expected 8784, got {len(expanded)}"
    return expanded


def parse_pvwatts(data: dict) -> tuple[pd.DataFrame, dict]:
    """
    Convert a PVWatts V8 JSON response into a solar timeseries DataFrame.

    Returns
    -------
    solar_df : pd.DataFrame (8784 rows, UTC index, 2024)
        Columns: poa (W/m2), tamb (°C), tcell (°C), ac_w (W per kW installed),
                 solar_cf (dimensionless capacity factor)

    meta : dict  — station info from PVWatts
    """
    outputs = data["outputs"]
    station = data.get("station_info", {})
    warnings = data.get("warnings", [])
    if warnings:
        for w in warnings:
            print(f"  PVWatts WARNING: {w}")

    # Extract hourly arrays (8760 TMY hours)
    ac_tmy   = np.array(outputs["ac"],    dtype=float)   # W
    poa_tmy  = np.array(outputs["poa"],   dtype=float)   # W/m²
    tamb_tmy = np.array(outputs["tamb"],  dtype=float)   # °C
    tcell_tmy= np.array(outputs["tcell"], dtype=float)   # °C
    dc_tmy   = np.array(outputs["dc"],    dtype=float)   # W

    # Expand from TMY 8760 -> 2024 8784 (leap year)
    ac    = _tmy_8760_to_2024_8784(ac_tmy)
    poa   = _tmy_8760_to_2024_8784(poa_tmy)
    tamb  = _tmy_8760_to_2024_8784(tamb_tmy)
    tcell = _tmy_8760_to_2024_8784(tcell_tmy)
    dc    = _tmy_8760_to_2024_8784(dc_tmy)

    # system_capacity = 1 kW, so CF = ac_W / 1000
    system_kw = PVWATTS_PARAMS["system_capacity"]
    solar_cf = np.maximum(ac / (system_kw * 1000.0), 0.0)

    # Build 2024 UTC datetime index (8784 hours)
    idx = pd.date_range("2024-01-01", periods=8784, freq="h", tz="UTC")

    solar_df = pd.DataFrame({
        "poa":      poa,
        "tamb":     tamb,
        "tcell":    tcell,
        "ac_w":     ac,
        "dc_w":     dc,
        "solar_cf": solar_cf,
    }, index=idx)
    solar_df.index.name = "datetime"

    meta = {
        "source":      "PVWatts V8 / NSRDB TMY",
        "latitude":    station.get("lat", LAT),
        "longitude":   station.get("lon", LON),
        "elevation":   station.get("elev", 137.0),
        "city":        station.get("city") or "Dallas-Fort Worth",
        "state":       station.get("state", "TX"),
        "nsrdb_file":  station.get("solar_resource_file", ""),
        "distance_km": round(station.get("distance", 0) / 1000, 1),  # metres -> km
        "ac_annual_kwh_per_kw": outputs.get("ac_annual"),
        "pvwatts_cf_pct":       outputs.get("capacity_factor"),
    }
    return solar_df, meta


# ---------------------------------------------------------------------------
# Source 2: Open-Meteo ERA5 + pvlib physics (fallback)
# ---------------------------------------------------------------------------
OM_SOLAR_VARS = [
    "shortwave_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
    "temperature_2m",
]
COL_RENAME_OM = {
    "shortwave_radiation":      "ghi",
    "diffuse_radiation":        "dhi",
    "direct_normal_irradiance": "dni",
    "temperature_2m":           "temp_c",
}


def _fetch_openmeteo_solar(year: int) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT, "longitude": LON,
        "start_date": f"{year}-01-01", "end_date": f"{year}-12-31",
        "hourly": ",".join(OM_SOLAR_VARS),
        "timezone": "UTC", "temperature_unit": "celsius",
    }
    print(f"  Requesting Open-Meteo ERA5 solar for {year} (UTC) ...")
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    times = pd.to_datetime(data["hourly"]["time"], utc=True)
    df = pd.DataFrame({v: data["hourly"][v] for v in OM_SOLAR_VARS}, index=times)
    df.index.name = "datetime"
    df = df.rename(columns=COL_RENAME_OM)
    for col in ["ghi", "dhi", "dni"]:
        df[col] = df[col].clip(lower=0.0)
    return df


def _build_openmeteo_fallback(year: int) -> tuple[pd.DataFrame, dict]:
    """Build solar CF via Open-Meteo ERA5 + pvlib physics (fallback path)."""
    from models.solar_model import build_solar_timeseries

    raw_path = RAW_DIR / f"ercot_openmeteo_solar_{year}.csv"
    if raw_path.exists():
        print(f"  Loading cached Open-Meteo solar {year} ...")
        df = pd.read_csv(raw_path, index_col="datetime", parse_dates=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
    else:
        df = _fetch_openmeteo_solar(year)
        df.to_csv(raw_path)
        print(f"  Cached -> {raw_path.name}")

    solar = build_solar_timeseries(df, lat=LAT, lon=LON)
    # Rename to match PVWatts column names for consistency
    solar = solar.rename(columns={"temp_c": "tamb"})
    solar["ac_w"]  = solar["solar_cf"] * 1000.0   # per 1 kW system
    solar["dc_w"]  = solar["ac_w"] / 0.96          # approx DC from AC

    meta = {
        "source":    "Open-Meteo ERA5 + pvlib physics",
        "latitude":  LAT, "longitude": LON, "elevation": 137.0,
        "city": "Dallas-Fort Worth", "state": "TX",
    }
    return solar, meta


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------
def load_solar() -> tuple[pd.DataFrame, dict]:
    """Try PVWatts V8 first; fall back to Open-Meteo ERA5 + pvlib."""
    if not _API_KEY:
        print("  NREL API key not set — using Open-Meteo ERA5 fallback.")
        return _build_openmeteo_fallback(TARGET_YEAR)

    try:
        data = _load_or_fetch_pvwatts()
        solar, meta = parse_pvwatts(data)
        return solar, meta
    except Exception as exc:
        print(f"  PVWatts unavailable ({exc})")
        print(f"  Falling back to Open-Meteo ERA5 + pvlib ...")
        return _build_openmeteo_fallback(TARGET_YEAR)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_solar(solar: pd.DataFrame, meta: dict) -> Path:
    source = meta.get("source", "")
    nsrdb_file = meta.get("nsrdb_file", "")
    cf_pct = meta.get("pvwatts_cf_pct")
    cf_label = f"PVWatts CF: {cf_pct:.2f}%" if cf_pct else ""

    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(
        f"ERCOT Solar PV Model  |  {LOCATION_NAME}  |  Source: {source}\n"
        f"Array: {TILT:.0f}deg tilt (latitude), S-facing  |  "
        f"module_type=Premium (gamma=-0.35%/C)  |  {cf_label}",
        fontsize=10, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.35)

    ax_poa   = fig.add_subplot(gs[0, :])
    ax_cf    = fig.add_subplot(gs[1, :])
    ax_week  = fig.add_subplot(gs[2, 0])
    ax_mon   = fig.add_subplot(gs[2, 1])

    MFMT = mdates.DateFormatter("%b")
    MLOC = mdates.MonthLocator()
    C1, C2 = "#e5c07b", "#98c379"

    # Panel 1: POA
    ax_poa.fill_between(solar.index, solar["poa"], alpha=0.3, color=C1)
    ax_poa.plot(solar.index, solar["poa"], color=C1, lw=0.4)
    ax_poa.set_ylabel("POA (W/m2)")
    ax_poa.set_title("Plane-of-Array Irradiance (from NSRDB TMY via PVWatts V8)")
    ax_poa.xaxis.set_major_formatter(MFMT); ax_poa.xaxis.set_major_locator(MLOC)
    ax_poa.grid(True, alpha=0.25)

    # Panel 2: Capacity factor
    ax_cf.fill_between(solar.index, solar["solar_cf"], alpha=0.3, color=C2)
    ax_cf.plot(solar.index, solar["solar_cf"], color=C2, lw=0.4)
    cf_ceiling = PVWATTS_PARAMS["inv_eff"] / 100 * (1 - PVWATTS_PARAMS["losses"] / 100)
    ax_cf.axhline(cf_ceiling, ls="--", color="#abb2bf", lw=0.9,
                  label=f"Theoretical ceiling: {cf_ceiling:.3f}")
    ax_cf.set_ylabel("Capacity Factor (--)")
    ax_cf.set_ylim(-0.02, 0.92)
    ax_cf.set_title("Solar Capacity Factor  |  P_solar [MW] = P_DC [MW] x CF")
    ax_cf.legend(fontsize=8)
    ax_cf.xaxis.set_major_formatter(MFMT); ax_cf.xaxis.set_major_locator(MLOC)
    ax_cf.grid(True, alpha=0.25)

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
    ax_week.xaxis.set_major_locator(mdates.DayLocator())
    ax_week.grid(True, alpha=0.25)

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
    print("ERCOT Solar PV Timeseries Builder  (PVWatts V8)")
    print("=" * 60)

    # 1. Load solar
    print(f"\n[1/4] Loading solar data ...")
    solar, meta = load_solar()

    print(f"  Source           : {meta['source']}")
    print(f"  Location         : {meta['city']}, {meta['state']}")
    print(f"  Grid point       : {meta['latitude']:.4f}N, {abs(meta['longitude']):.4f}W")
    if meta.get("nsrdb_file"):
        print(f"  NSRDB TMY file   : {meta['nsrdb_file']}")
    if meta.get("distance_km"):
        print(f"  Distance to site : {meta['distance_km']} km")
    print(f"  Rows             : {len(solar)} ({len(solar)/24:.0f} days)")

    # 2. (Physics already done by PVWatts / fallback)
    print(f"\n[2/4] Array configuration")
    print(f"  module_type  : 1 (Premium, gamma = -0.35%/degC, N-type equivalent)")
    print(f"  array_type   : 0 (Fixed Open Rack, utility scale)")
    print(f"  tilt         : {PVWATTS_PARAMS['tilt']}deg  (latitude-optimal)")
    print(f"  azimuth      : {PVWATTS_PARAMS['azimuth']}deg  (south-facing)")
    print(f"  losses       : {PVWATTS_PARAMS['losses']}%  (soiling + wiring)")
    print(f"  inv_eff      : {PVWATTS_PARAMS['inv_eff']}%")
    print(f"  dc_ac_ratio  : {PVWATTS_PARAMS['dc_ac_ratio']}")
    derate = (1 - PVWATTS_PARAMS["losses"]/100) * (PVWATTS_PARAMS["inv_eff"]/100)
    print(f"  Combined derate factor : {derate:.3f}  (vs eta_sys={ETA_SYS} in physics model)")

    # 3. Save
    out_path = OUT_DIR / f"ercot_solar_{TARGET_YEAR}.csv"
    solar.to_csv(out_path)
    print(f"\n[3/4] Saved -> {out_path}")

    # 4. Summary
    print(f"\n[4/4] Summary statistics")
    print("-" * 50)
    cf = solar["solar_cf"]
    poa = solar["poa"]

    daylight = poa > 1.0
    print(f"  Total hours            : {len(cf):,}")
    print(f"  Daylight hours (POA>1) : {daylight.sum():,} ({daylight.mean()*100:.1f}%)")
    print(f"  Peak POA               : {poa.max():.0f} W/m2")
    print(f"  Peak capacity factor   : {cf.max():.4f}  ({cf.max()*100:.1f}%)")
    print(f"  Peak CF hour           : {cf.idxmax()}")
    print(f"  Annual capacity factor : {cf.mean()*100:.2f}%")
    if meta.get("pvwatts_cf_pct"):
        print(f"  PVWatts reported CF    : {meta['pvwatts_cf_pct']:.2f}%  (TMY 8760h basis)")
    print(f"  Mean CF (daylight)     : {cf[daylight].mean()*100:.2f}%")
    if meta.get("ac_annual_kwh_per_kw"):
        print(f"  AC annual yield        : {meta['ac_annual_kwh_per_kw']:.1f} kWh/kW installed")

    print(f"\n  Optimizer usage  (P_solar_t = P_DC x solar_cf_t):")
    for mw in [100, 150, 200, 250]:
        ann = mw * cf.sum()
        pct = ann / 522_524 * 100
        print(f"    {mw:3d} MW DC  ->  {ann:,.0f} MWh/year  ({pct:.0f}% of demand)")
    print(f"    Datacenter demand  :  522,524 MWh/year")

    print(f"\n  Monthly capacity factors:")
    monthly_cf = cf.resample("ME").mean()
    for ts, v in monthly_cf.items():
        bar = "#" * int(v * 200)
        print(f"    {ts.strftime('%b'):3s}  {v:.4f}  ({v*100:5.2f}%)  {bar}")

    # Demand alignment check
    demand_path = OUT_DIR / "ercot_demand_2024.csv"
    if demand_path.exists():
        demand = pd.read_csv(demand_path, index_col="datetime", parse_dates=True)
        print(f"\n  Alignment with demand timeseries:")
        print(f"    Demand rows : {len(demand):,}")
        print(f"    Solar rows  : {len(solar):,}")
        status = "OK - ready for dispatch" if len(demand) == len(solar) else "MISMATCH"
        print(f"    Status      : {status}")

    # 5. Plot
    print("\n  Generating plots ...")
    plot_solar(solar, meta)

    print("\nDone.")
    return solar


if __name__ == "__main__":
    main()
