"""
ISO Pipeline Orchestrator
=========================
Runs the full simulation pipeline for any ISO/RTO on demand.

Steps
-----
1. Fetch weather (Open-Meteo)   → {iso}_demand_{year}.csv
2. Fetch solar (PVWatts V8)     → {iso}_solar_{year}.csv
3. Build BESS params            → {iso}_bess_params_{year}.csv
4. Solve reliability surface    → {iso}_reliability_surface_{year}.csv
5. Optimise sLCOE               → {iso}_slcoe_surface_{year}.csv

All outputs are cached to data/processed/. Subsequent calls skip steps
whose output files already exist (unless force=True).
"""

import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED    = PROJECT_ROOT / "data" / "processed"
RAW          = PROJECT_ROOT / "data" / "raw"

PROCESSED.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

def _iso_prefix(iso_id: str, year: int) -> str:
    return iso_id.lower()


def output_files(iso_id: str, year: int) -> dict[str, Path]:
    p = iso_id.lower()
    return {
        "demand":       PROCESSED / f"{p}_demand_{year}.csv",
        "solar":        PROCESSED / f"{p}_solar_{year}.csv",
        "bess":         PROCESSED / f"{p}_bess_params_{year}.csv",
        "reliability":  PROCESSED / f"{p}_reliability_surface_{year}.csv",
        "slcoe":        PROCESSED / f"{p}_slcoe_surface_{year}.csv",
    }


def pipeline_status(iso_id: str, year: int) -> dict[str, bool]:
    """Return which output files exist for this ISO."""
    return {k: v.exists() for k, v in output_files(iso_id, year).items()}


def is_complete(iso_id: str, year: int) -> bool:
    """True if all five output files exist."""
    return all(pipeline_status(iso_id, year).values())


# ---------------------------------------------------------------------------
# Step 1: Demand (Open-Meteo + PUE model)
# ---------------------------------------------------------------------------

def _build_demand(iso_id: str, lat: float, lon: float, year: int,
                  log: Callable) -> pd.DataFrame:
    from models.demand_model import build_demand_timeseries

    out_path = output_files(iso_id, year)["demand"]
    raw_path = RAW / f"{iso_id.lower()}_weather_{year}.csv"

    if out_path.exists():
        log(f"  demand: loaded from cache ({out_path.name})")
        return pd.read_csv(out_path, index_col="datetime", parse_dates=True)

    # Fetch weather
    log(f"  demand: fetching Open-Meteo for {lat}°N {abs(lon):.2f}°W ...")
    url = "https://archive-api.open-meteo.com/v1/archive"
    resp = requests.get(url, params={
        "latitude": lat, "longitude": lon,
        "start_date": f"{year}-01-01", "end_date": f"{year}-12-31",
        "hourly": "temperature_2m",
        "timezone": "UTC",
        "temperature_unit": "celsius",
    }, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    times = pd.to_datetime(data["hourly"]["time"], utc=True)
    df_w = pd.DataFrame({"temp_c": data["hourly"]["temperature_2m"]}, index=times)
    df_w.index = df_w.index.tz_localize(None)
    df_w.index.name = "datetime"

    # Leap year check / interpolate NaN
    expected = 8784 if year % 4 == 0 else 8760
    if len(df_w) != expected:
        df_w = df_w.iloc[:expected]
    df_w["temp_c"] = df_w["temp_c"].interpolate(method="time")
    df_w.to_csv(raw_path)

    demand = build_demand_timeseries(df_w)
    demand.to_csv(out_path)
    log(f"  demand: saved {len(demand):,} rows -> {out_path.name}")
    return demand


# ---------------------------------------------------------------------------
# Step 2: Solar (PVWatts V8)
# ---------------------------------------------------------------------------

def _build_solar(iso_id: str, lat: float, lon: float,
                 tilt: float, year: int, log: Callable) -> pd.DataFrame:
    from dotenv import load_dotenv
    import os, json

    out_path  = output_files(iso_id, year)["solar"]
    raw_path  = RAW / f"{iso_id.lower()}_pvwatts_tmy.json"

    if out_path.exists():
        log(f"  solar:  loaded from cache ({out_path.name})")
        return pd.read_csv(out_path, index_col="datetime", parse_dates=True)

    load_dotenv(PROJECT_ROOT / "config" / "secrets.env")
    api_key = os.getenv("NREL_API_KEY", "DEMO_KEY")
    email   = os.getenv("NREL_EMAIL", "demo@example.com")

    log(f"  solar:  calling PVWatts V8 for {lat}°N {abs(lon):.2f}°W ...")
    resp = requests.get(
        "https://developer.nrel.gov/api/pvwatts/v8.json",
        params=dict(
            api_key=api_key, email=email,
            lat=lat, lon=lon,
            system_capacity=1, module_type=1, array_type=0,
            tilt=round(tilt, 2), azimuth=180,
            losses=15, inv_eff=96, dc_ac_ratio=1.2,
            dataset="nsrdb", timeframe="hourly",
        ),
        timeout=60,
    )
    resp.raise_for_status()
    payload = resp.json()
    raw_path.write_text(json.dumps(payload, indent=2))

    outputs  = payload["outputs"]
    ac_w     = outputs["ac"]          # 8760-length list, Wh per W
    poa      = outputs.get("poa", [0]*8760)
    tamb     = outputs.get("tamb", [20]*8760)
    tcell    = outputs.get("tcell", [25]*8760)
    dc_w     = outputs.get("dc", [0]*8760)
    solar_cf = [v / 1000.0 for v in ac_w]   # CF = ac_Wh / (1 kW * 1000)

    # Expand TMY 8760 → 8784 by duplicating Feb 28 as Feb 29
    def _expand(lst):
        arr = list(lst)
        return arr[:1416] + arr[1392:1416] + arr[1416:]

    index_2024 = pd.date_range(f"{year}-01-01", periods=8784, freq="h")
    df_s = pd.DataFrame({
        "poa": _expand(poa), "tamb": _expand(tamb),
        "tcell": _expand(tcell), "ac_w": _expand(ac_w),
        "dc_w": _expand(dc_w), "solar_cf": _expand(solar_cf),
    }, index=index_2024)
    df_s.index.name = "datetime"
    df_s.to_csv(out_path)
    log(f"  solar:  saved {len(df_s):,} rows -> {out_path.name}")
    return df_s


# ---------------------------------------------------------------------------
# Step 3: BESS params
# ---------------------------------------------------------------------------

def _build_bess(iso_id: str, year: int, demand: pd.DataFrame,
                log: Callable) -> pd.DataFrame:
    from models.bess_model import build_bess_params_timeseries

    out_path = output_files(iso_id, year)["bess"]
    if out_path.exists():
        log(f"  bess:   loaded from cache ({out_path.name})")
        return pd.read_csv(out_path, index_col="datetime", parse_dates=True)

    bess_params = build_bess_params_timeseries(demand[["temp_c"]])
    bess_params.to_csv(out_path)
    log(f"  bess:   saved {len(bess_params):,} rows -> {out_path.name}")
    return bess_params


# ---------------------------------------------------------------------------
# Step 4: Reliability surface
# ---------------------------------------------------------------------------

def _build_reliability(iso_id: str, year: int, log: Callable) -> pd.DataFrame:
    from models.dispatcher import dispatch, load_timeseries as _load_ts
    from models.bess_model  import B_UNIT_MWH, N_UNITS_MAX
    from itertools import product

    out_path = output_files(iso_id, year)["reliability"]
    if out_path.exists():
        log(f"  reliability: loaded from cache ({out_path.name})")
        return pd.read_csv(out_path)

    ts = _load_ts(PROCESSED, year, iso_id=iso_id)
    S_GRID = np.arange(0, 301, 25)
    N_GRID = np.arange(0, N_UNITS_MAX + 1, 10, dtype=int)
    B_GRID = N_GRID * B_UNIT_MWH
    G_INF  = 1_000_000.0

    rows = []
    n_total = len(S_GRID) * len(N_GRID)
    t0 = time.time()
    for i, (S, (N, B)) in enumerate(product(S_GRID, zip(N_GRID, B_GRID))):
        results, _ = dispatch(float(S), float(B), G_INF, ts)
        G_min     = float(results["gas_gen_mw"].max())
        gas_hours = float((results["gas_gen_mw"] > 0.1).sum())
        rows.append({"S_mw": S, "N_units": int(N),
                     "B_mwh": B, "G_min_mw": round(G_min, 2),
                     "gas_hours_yr": round(gas_hours, 0)})
        if (i + 1) % max(1, n_total // 5) == 0 or (i + 1) == n_total:
            pct = (i + 1) / n_total * 100
            log(f"  reliability: {pct:.0f}% ({i+1}/{n_total}) "
                f"elapsed {time.time()-t0:.0f}s")

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    log(f"  reliability: saved {len(df)} rows -> {out_path.name}")
    return df


# ---------------------------------------------------------------------------
# Step 5: sLCOE optimisation
# ---------------------------------------------------------------------------

def _build_slcoe(iso_id: str, year: int, iso_cfg: dict,
                 log: Callable) -> pd.DataFrame:
    from models.dispatcher   import dispatch, load_timeseries as _load_ts
    from models.iso_registry import get_costs, compute_slcoe
    from models.bess_model   import B_UNIT_MWH, N_UNITS_MAX

    out_path = output_files(iso_id, year)["slcoe"]
    if out_path.exists():
        log(f"  slcoe:  loaded from cache ({out_path.name})")
        return pd.read_csv(out_path)

    reliability = pd.read_csv(output_files(iso_id, year)["reliability"])
    ts          = _load_ts(PROCESSED, year, iso_id=iso_id)
    costs       = get_costs()
    annual_load = ts["load_mw"].sum()

    S_GRID = np.arange(0, 301, 25)
    N_GRID = np.arange(0, N_UNITS_MAX + 1, 10, dtype=int)
    B_GRID = N_GRID * B_UNIT_MWH

    rel_lookup = reliability.set_index(["S_mw", "B_mwh"])["G_min_mw"]

    rows = []
    for S in S_GRID:
        for N, B in zip(N_GRID, B_GRID):
            G_min = float(rel_lookup.get((S, B), ts["load_mw"].max()))
            results, summary = dispatch(float(S), float(B), G_min, ts)
            gas_gen_mwh = float(results["gas_gen_mw"].sum())

            fin = compute_slcoe(iso_cfg, costs, float(S), float(B),
                                G_min, gas_gen_mwh, annual_load)
            rows.append({
                "S_mw": S, "N_units": int(N), "B_mwh": B,
                "G_min_mw": G_min,
                "gas_gen_mwh": round(gas_gen_mwh, 0),
                "demand_mwh_yr": round(annual_load, 0),
                "slcoe_per_mwh": round(fin["slcoe_per_mwh"], 3),
                "total_cost_usd_yr": round(fin["total_annual_cost"], 0),
                "solar_cost_usd_yr": round(fin["solar_capex_ann"] + fin["solar_opex_ann"], 0),
                "bess_cost_usd_yr":  round(fin["bess_capex_ann"]  + fin["bess_opex_ann"],  0),
                "gas_cost_usd_yr":   round(fin["gas_capex_ann"]   + fin["gas_opex_ann"] +
                                           fin["fuel_cost_ann"]   + fin["gas_var_opex_ann"], 0),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    log(f"  slcoe:  saved {len(df)} rows -> {out_path.name}")
    return df


# ---------------------------------------------------------------------------
# Public: run full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(iso_id: str, year: int = 2024,
                 force: bool = False,
                 log: Callable = print) -> dict[str, pd.DataFrame]:
    """
    Run (or resume) the full simulation pipeline for one ISO.

    Parameters
    ----------
    iso_id : e.g. "CAISO", "PJM"
    year   : simulation year (default 2024)
    force  : re-run all steps even if outputs exist
    log    : callable for progress messages (default print, use st.write in Streamlit)

    Returns
    -------
    dict with keys: demand, solar, bess, reliability, slcoe
    """
    from models.iso_registry import get_iso
    cfg = get_iso(iso_id)

    if force:
        for p in output_files(iso_id, year).values():
            if p.exists():
                p.unlink()

    log(f"Pipeline: {iso_id} — {cfg['city']} — {year}")

    demand      = _build_demand(iso_id, cfg["lat"], cfg["lon"], year, log)
    solar       = _build_solar(iso_id, cfg["lat"], cfg["lon"],
                               cfg["pv_tilt_deg"], year, log)
    bess        = _build_bess(iso_id, year, demand, log)
    reliability = _build_reliability(iso_id, year, log)
    slcoe       = _build_slcoe(iso_id, year, cfg, log)

    log(f"Pipeline complete for {iso_id}.")
    return dict(demand=demand, solar=solar, bess=bess,
                reliability=reliability, slcoe=slcoe)
