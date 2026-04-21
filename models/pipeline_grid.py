"""
Grid-Connected Pipeline
=======================
Runs the grid-connected simulation for any ISO, building on the base
pipeline outputs (demand, solar, bess) produced by pipeline.py.

Steps
-----
1–3. Demand / Solar / BESS  — shared with base pipeline; run automatically
                               if outputs are missing.
4.   Grid prices             → {iso}_grid_prices_{year}.csv
5.   Grid sLCOE surface      → {iso}_grid_slcoe_surface_{year}.csv
         Searches over (S_mw, B_mwh, G_mw) with G unconstrained (grid is
         infinite backup), finds minimum sLCOE grid-connected configuration.

Comparison
----------
After running both pipelines, compare:
    pipeline.output_files(iso, year)["slcoe"]      — microgrid optimal
    grid_output_files(iso, year)["grid_slcoe"]     — grid-connected optimal
"""

import time
from itertools import product as iproduct
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED    = PROJECT_ROOT / "data" / "processed"

PROCESSED.mkdir(parents=True, exist_ok=True)

# G_mw search grid for grid-connected model (gas can be 0 — grid is backup)
G_GRID = np.array([0, 25, 50, 75, 100, 150, 200], dtype=float)


# ---------------------------------------------------------------------------
# Output file registry
# ---------------------------------------------------------------------------

def grid_output_files(iso_id: str, year: int) -> dict[str, Path]:
    p = iso_id.lower()
    return {
        "grid_prices": PROCESSED / f"{p}_grid_prices_{year}.csv",
        "grid_slcoe":  PROCESSED / f"{p}_grid_slcoe_surface_{year}.csv",
    }


def grid_pipeline_status(iso_id: str, year: int) -> dict[str, bool]:
    return {k: v.exists() for k, v in grid_output_files(iso_id, year).items()}


def is_grid_complete(iso_id: str, year: int) -> bool:
    return all(grid_pipeline_status(iso_id, year).values())


# ---------------------------------------------------------------------------
# Step 4: Grid prices
# ---------------------------------------------------------------------------

def _build_grid_prices(iso_id: str, year: int, log: Callable) -> pd.DataFrame:
    from models.grid_prices_model import build_grid_price_timeseries

    out_path = grid_output_files(iso_id, year)["grid_prices"]
    if out_path.exists():
        log(f"  grid prices: loaded from cache ({out_path.name})")
        return pd.read_csv(out_path, index_col="datetime", parse_dates=True)

    df = build_grid_price_timeseries(iso_id, year, log=log)
    df.to_csv(out_path)
    log(f"  grid prices: cached -> {out_path.name}")
    return df


# ---------------------------------------------------------------------------
# Step 5: Grid-connected sLCOE surface
# ---------------------------------------------------------------------------

def _build_grid_slcoe(
    iso_id: str, year: int, iso_cfg: dict, log: Callable
) -> pd.DataFrame:
    from models.dispatcher   import dispatch_grid, load_timeseries as _load_ts
    from models.iso_registry import get_costs, compute_slcoe_grid
    from models.bess_model   import B_UNIT_MWH, N_UNITS_MAX
    from models.gas_model    import HEAT_RATE_MMBTU_PER_MWH

    out_path = grid_output_files(iso_id, year)["grid_slcoe"]
    if out_path.exists():
        log(f"  grid slcoe: loaded from cache ({out_path.name})")
        return pd.read_csv(out_path)

    ts = _load_ts(PROCESSED, year, iso_id=iso_id)
    grid_prices = pd.read_csv(
        grid_output_files(iso_id, year)["grid_prices"],
        index_col="datetime", parse_dates=True,
    )

    # Align grid prices positionally (same convention as base pipeline)
    ts = ts.copy()
    ts["grid_price_per_mwh"] = grid_prices["grid_price_per_mwh"].values

    costs       = get_costs()
    annual_load = float(ts["load_mw"].sum())

    # ISO-specific gas short-run marginal cost ($/MWh)
    gas_marginal = (
        iso_cfg["gas_price_per_mmbtu"] * HEAT_RATE_MMBTU_PER_MWH
        + costs["gas_rice"]["opex_variable_per_mwh"]
    )

    S_GRID = np.arange(0, 301, 25)
    N_GRID = np.arange(0, N_UNITS_MAX + 1, 10, dtype=int)
    B_GRID = N_GRID * B_UNIT_MWH

    combos  = list(iproduct(S_GRID, zip(N_GRID, B_GRID), G_GRID))
    n_total = len(combos)
    t0      = time.time()
    rows    = []

    for i, (S, (N, B), G) in enumerate(combos):
        results, _ = dispatch_grid(
            float(S), float(B), float(G), ts, gas_marginal
        )

        gas_gen_mwh      = float(results["gas_gen_mw"].sum())
        grid_import_mwh  = float(results["grid_import_mw"].sum())
        grid_import_cost = float(
            (results["grid_import_mw"] * results["grid_price_per_mwh"].clip(lower=0)).sum()
        )
        peak_grid_mw     = float(results["grid_import_mw"].max())

        fin = compute_slcoe_grid(
            iso_cfg, costs,
            float(S), float(B), float(G),
            gas_gen_mwh, grid_import_cost, grid_import_mwh,
            peak_grid_mw, annual_load,
        )

        rows.append({
            "S_mw":                   S,
            "N_units":                int(N),
            "B_mwh":                  B,
            "G_mw":                   G,
            "gas_gen_mwh":            round(gas_gen_mwh, 0),
            "grid_import_mwh_yr":     round(grid_import_mwh, 0),
            "demand_mwh_yr":          round(annual_load, 0),
            "slcoe_per_mwh":          round(fin["slcoe_per_mwh"], 3),
            "total_cost_usd_yr":      round(fin["total_annual_cost"], 0),
            "solar_cost_usd_yr":      round(fin["solar_capex_ann"]  + fin["solar_opex_ann"], 0),
            "bess_cost_usd_yr":       round(fin["bess_capex_ann"]   + fin["bess_opex_ann"],  0),
            "gas_cost_usd_yr":        round(
                fin["gas_capex_ann"]  + fin["gas_opex_ann"] +
                fin["fuel_cost_ann"]  + fin["gas_var_opex_ann"], 0
            ),
            "grid_fixed_cost_usd_yr": round(fin["grid_capex_ann"]  + fin["grid_opex_ann"],  0),
            "grid_energy_cost_usd_yr": round(fin["grid_import_cost_ann"], 0),
        })

        if (i + 1) % max(1, n_total // 5) == 0 or (i + 1) == n_total:
            pct = (i + 1) / n_total * 100
            log(f"  grid slcoe: {pct:.0f}% ({i+1}/{n_total}) "
                f"elapsed {time.time()-t0:.0f}s")

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    log(f"  grid slcoe: saved {len(df)} rows -> {out_path.name}")
    return df


# ---------------------------------------------------------------------------
# Public: run grid pipeline
# ---------------------------------------------------------------------------

def run_grid_pipeline(
    iso_id: str,
    year: int = 2024,
    force: bool = False,
    log: Callable = print,
) -> dict[str, pd.DataFrame]:
    """
    Run (or resume) the grid-connected simulation pipeline for one ISO.

    Automatically runs the base pipeline first if demand/solar/bess outputs
    are missing (they are shared between both models).

    Parameters
    ----------
    iso_id : e.g. "CAISO", "ERCOT"
    year   : simulation year (default 2024)
    force  : re-run all grid steps even if outputs exist
    log    : callable for progress messages

    Returns
    -------
    dict with keys: grid_prices, grid_slcoe
    """
    from models.iso_registry import get_iso
    from models.pipeline     import output_files as base_files, run_pipeline as run_base

    cfg = get_iso(iso_id)

    # Ensure base pipeline outputs (demand, solar, bess) exist
    base = base_files(iso_id, year)
    if any(not base[k].exists() for k in ("demand", "solar", "bess")):
        log("Base pipeline outputs missing — running base pipeline first ...")
        run_base(iso_id, year, log=log)

    if force:
        for p in grid_output_files(iso_id, year).values():
            if p.exists():
                p.unlink()

    log(f"Grid pipeline: {iso_id} — {cfg['city']} — {year}")

    grid_prices = _build_grid_prices(iso_id, year, log)
    grid_slcoe  = _build_grid_slcoe(iso_id, year, cfg, log)

    log(f"Grid pipeline complete for {iso_id}.")
    return dict(grid_prices=grid_prices, grid_slcoe=grid_slcoe)
