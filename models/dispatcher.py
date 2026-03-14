"""
Hourly dispatch simulation for the datacenter microgrid.

Dispatch priority stack
-----------------------
    1. Solar PV       — zero marginal cost, always taken first
    2. BESS discharge — covers residual deficit after solar
    3. Natural gas    — gap-filler of last resort

This is a fully islanded BTM microgrid — no grid connection is modelled.
Any unserved energy represents a reliability failure (datacenter outage).

Each hour t follows this logic:

    solar_gen_t  =  S_mw  ×  solar_cf_t

    If solar_gen_t  ≥  load_t   (surplus):
        BESS charges with the excess up to its power/SoC limits.
        Any remaining excess is curtailed.

    If solar_gen_t  <  load_t   (deficit):
        BESS discharges up to its power/SoC limits to cover the gap.
        Gas fills whatever the BESS could not deliver.
        Any residual is unserved energy.

BESS state update (from methodology document)
---------------------------------------------
    E_t = E_{t-1} + P_ch_t × (η_ch × η_temp_t)
                  − P_dis_t / (η_temp_t × η_dis)

    Constraints:  0.20 B  ≤  E_t  ≤  B
                  P_ch_t, P_dis_t  ≤  B / 4   (4-hour battery)

Inputs
------
    S_mw    : float   Installed solar DC capacity [MW]     ← optimizer var
    B_mwh   : float   Battery energy capacity [MWh]        ← optimizer var
    G_mw    : float   Installed gas capacity [MW]          ← optimizer var

    timeseries : pd.DataFrame (8784 rows)
        load_mw     — hourly facility total load [MW]
        solar_cf    — hourly solar capacity factor [–]
        eta_temp    — hourly BESS PHVAC modifier [–]

Note on timestamp alignment
---------------------------
The demand/BESS data uses a tz-naive CST index; solar uses UTC.
Both cover all 8784 hours of 2024 in order. The dispatcher uses
positional (.values) alignment — row 0 is the first hour of 2024,
row 8783 is the last. Since solar is TMY (not actual 2024 data), exact
UTC↔CST alignment would be a false precision; positional is appropriate.
"""

import numpy as np
import pandas as pd

from models.bess_model  import ETA_CH, ETA_DIS, SOC_MIN, SOC_MAX, DURATION_H
from models.gas_model   import dispatch_gas, fuel_consumption, fuel_cost, co2_emissions


# ---------------------------------------------------------------------------
# Core dispatch loop
# ---------------------------------------------------------------------------

def dispatch(
    S_mw:       float,
    B_mwh:      float,
    G_mw:       float,
    timeseries: pd.DataFrame,
    E0:         float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Run the hourly dispatch simulation for one (S, B, G) configuration.

    Parameters
    ----------
    S_mw, B_mwh, G_mw : float  — installed capacities (optimizer decision vars)
    timeseries : pd.DataFrame  — must have columns: load_mw, solar_cf, eta_temp
    E0 : float | None          — initial BESS SoC [MWh]; default = 50% of B

    Returns
    -------
    results : pd.DataFrame (same index as timeseries)
        load_mw          MW   Facility total load
        solar_gen_mw     MW   Solar output (= S × CF, before curtailment)
        bess_charge_mw   MW   Power flowing into battery
        bess_discharge_mw MW  Power flowing out of battery
        gas_gen_mw       MW   Gas turbine / RICE output
        soc_mwh          MWh  Battery state-of-charge (end of hour)
        curtailed_mw     MW   Excess solar that couldn't be stored
        unserved_mw      MW   Load not met by any source (should be 0 with large G)
        net_load_mw      MW   load − solar (positive = deficit)

    summary : dict  — annual aggregates and key KPIs
    """
    _validate(timeseries)

    load     = timeseries["load_mw"].to_numpy(dtype=float)
    cf       = timeseries["solar_cf"].to_numpy(dtype=float)
    eta_t    = timeseries["eta_temp"].to_numpy(dtype=float)
    n        = len(load)

    # BESS parameters
    P_bess_max = B_mwh / DURATION_H if B_mwh > 0 else 0.0
    E_min      = SOC_MIN * B_mwh
    E_max      = SOC_MAX * B_mwh
    E          = E0 if E0 is not None else 0.5 * B_mwh

    # Pre-compute solar generation (before curtailment)
    solar_potential = S_mw * cf

    # Output arrays
    p_ch    = np.zeros(n)
    p_dis   = np.zeros(n)
    p_gas   = np.zeros(n)
    p_curt  = np.zeros(n)
    p_unmet = np.zeros(n)
    soc     = np.zeros(n)

    # --------------- hourly loop ----------------------------------------
    for t in range(n):
        et      = max(eta_t[t], 1e-6)    # guard against zero
        solar_t = solar_potential[t]
        net     = load[t] - solar_t       # positive = deficit, negative = surplus

        if net > 0.0:
            # --- Deficit: discharge BESS, then gas ----------------------
            # Max BESS discharge this hour (power AND energy limits)
            max_dis = min(
                P_bess_max,
                max((E - E_min) * et * ETA_DIS, 0.0),
            )
            p_dis_t = min(net, max_dis)
            E      -= p_dis_t / (et * ETA_DIS)

            remaining  = net - p_dis_t
            p_gas_t    = min(G_mw, remaining) if G_mw > 0 else 0.0
            unserved_t = max(remaining - p_gas_t, 0.0)

            p_dis[t]   = p_dis_t
            p_gas[t]   = p_gas_t
            p_unmet[t] = unserved_t

        else:
            # --- Surplus: charge BESS, curtail remainder ----------------
            excess = -net
            # Max BESS charge this hour (power AND headroom limits)
            max_ch = min(
                P_bess_max,
                max((E_max - E) / (ETA_CH * et), 0.0),
            ) if B_mwh > 0 else 0.0
            p_ch_t  = min(excess, max_ch)
            E      += p_ch_t * ETA_CH * et

            p_ch[t]   = p_ch_t
            p_curt[t] = excess - p_ch_t

        # Safety clamp (floating-point drift)
        E = max(E_min, min(E_max, E))
        soc[t] = E
    # --------------- end loop -------------------------------------------

    # Build results DataFrame
    results = pd.DataFrame({
        "load_mw":            load,
        "solar_gen_mw":       solar_potential,
        "bess_charge_mw":     p_ch,
        "bess_discharge_mw":  p_dis,
        "gas_gen_mw":         p_gas,
        "soc_mwh":            soc,
        "curtailed_mw":       p_curt,
        "unserved_mw":        p_unmet,
        "net_load_mw":        load - solar_potential,
    }, index=timeseries.index)

    summary = _summarise(results, S_mw, B_mwh, G_mw)
    return results, summary


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def _summarise(r: pd.DataFrame, S_mw: float, B_mwh: float, G_mw: float) -> dict:
    """Compute annual KPIs from the hourly results DataFrame."""
    from models.gas_model import (
        fuel_consumption, fuel_cost, co2_emissions,
        GAS_PRICE_PER_MMBTU, HEAT_RATE_MMBTU_PER_MWH,
    )

    total_demand   = r["load_mw"].sum()
    solar_gen      = r["solar_gen_mw"].sum()
    solar_potential= solar_gen          # before curtailment check
    curtailed      = r["curtailed_mw"].sum()
    solar_used     = solar_gen - curtailed
    bess_dis       = r["bess_discharge_mw"].sum()
    bess_ch        = r["bess_charge_mw"].sum()
    gas_gen        = r["gas_gen_mw"].sum()
    unserved       = r["unserved_mw"].sum()

    supplied       = solar_used + bess_dis + gas_gen
    fuel           = fuel_consumption(r["gas_gen_mw"].to_numpy()).sum()
    cost           = fuel_cost(r["gas_gen_mw"].to_numpy()).sum()
    co2            = co2_emissions(r["gas_gen_mw"].to_numpy()).sum()

    bess_cycles    = bess_dis / B_mwh if B_mwh > 0 else 0.0
    solar_curt_pct = curtailed / solar_gen * 100 if solar_gen > 0 else 0.0
    gas_cf_pct     = r["gas_gen_mw"].mean() / G_mw * 100 if G_mw > 0 else 0.0

    soc_vals = r["soc_mwh"]
    soc_pct  = soc_vals / B_mwh * 100 if B_mwh > 0 else soc_vals * 0

    return {
        # Configuration
        "S_mw":                    S_mw,
        "B_mwh":                   B_mwh,
        "G_mw":                    G_mw,
        "P_bess_max_mw":           B_mwh / DURATION_H,
        # Annual energy [MWh]
        "total_demand_mwh":        total_demand,
        "solar_gen_mwh":           solar_gen,
        "solar_used_mwh":          solar_used,
        "curtailed_mwh":           curtailed,
        "bess_discharge_mwh":      bess_dis,
        "bess_charge_mwh":         bess_ch,
        "gas_gen_mwh":             gas_gen,
        "unserved_mwh":            unserved,
        # Energy mix [%]
        "solar_share_pct":         solar_used    / total_demand * 100,
        "bess_share_pct":          bess_dis      / total_demand * 100,
        "gas_share_pct":           gas_gen       / total_demand * 100,
        "unserved_pct":            unserved      / total_demand * 100,
        # Solar performance
        "solar_curtailment_pct":   solar_curt_pct,
        "solar_cf_pct":            r["solar_gen_mw"].mean() / S_mw * 100 if S_mw > 0 else 0,
        # BESS performance
        "bess_cycles_per_year":    bess_cycles,
        "bess_soc_mean_pct":       soc_pct.mean(),
        "bess_soc_min_pct":        soc_pct.min(),
        # Gas performance
        "gas_cf_pct":              gas_cf_pct,
        "annual_fuel_mmbtu":       fuel,
        "annual_fuel_cost_usd":    cost,
        "annual_co2_t":            co2,
        "gas_variable_cost_mwh":   cost / gas_gen if gas_gen > 0 else 0,
        # Renewable self-sufficiency
        "renewable_share_pct":     (solar_used + bess_dis) / total_demand * 100,
    }


# ---------------------------------------------------------------------------
# Timeseries loader (convenience)
# ---------------------------------------------------------------------------

def load_timeseries(processed_dir, year: int = 2024,
                    iso_id: str = "ercot") -> pd.DataFrame:
    """
    Load and merge demand, solar, and BESS parameter timeseries into a
    single DataFrame ready for the dispatcher.

    Uses positional alignment (all three are 8784-row 2024 annual series).
    The demand index (tz-naive CST) is used as the reference index.

    Parameters
    ----------
    iso_id : ISO prefix for filenames (e.g. "ercot", "pjm", "caiso").
             Case-insensitive. Defaults to "ercot" for backward compatibility.
    """
    import pandas as pd
    from pathlib import Path
    d   = Path(processed_dir)
    pfx = iso_id.lower()

    demand    = pd.read_csv(d / f"{pfx}_demand_{year}.csv",
                            index_col="datetime", parse_dates=True)
    solar     = pd.read_csv(d / f"{pfx}_solar_{year}.csv",
                            index_col="datetime", parse_dates=True)
    bess_par  = pd.read_csv(d / f"{pfx}_bess_params_{year}.csv",
                            index_col="datetime", parse_dates=True)

    # Validate lengths
    assert len(demand) == len(solar) == len(bess_par), (
        f"Timeseries length mismatch: demand={len(demand)}, "
        f"solar={len(solar)}, bess={len(bess_par)}"
    )

    ts = pd.DataFrame({
        "load_mw":   demand["total_load_mw"].values,
        "solar_cf":  solar["solar_cf"].values,
        "eta_temp":  bess_par["eta_temp"].values,
        "temp_c":    demand["temp_c"].values,
    }, index=demand.index)

    return ts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate(ts: pd.DataFrame) -> None:
    required = {"load_mw", "solar_cf", "eta_temp"}
    missing = required - set(ts.columns)
    if missing:
        raise ValueError(f"timeseries missing columns: {missing}")
    if ts.isna().any().any():
        raise ValueError("timeseries contains NaN values — interpolate before dispatch.")
