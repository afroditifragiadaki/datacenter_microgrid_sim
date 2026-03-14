"""
Natural gas generation model for a collocated datacenter microgrid.

Role
----
Gas is the gap-filler of last resort, dispatched after solar and BESS have
been applied:

    P_gas,t  =  min(G,  unmet_demand_t)

Where unmet_demand_t is whatever load remains after solar generation and
BESS discharge. G (installed gas capacity in MW) is the optimizer's decision
variable.

Fuel consumption (from methodology document):

    Fuel_t  [MMBtu]  =  P_gas,t [MW]  ×  Heat_Rate [MMBtu/MWh]

Technology assumptions — Natural Gas Reciprocating Engine (RICE)
-----------------------------------------------------------------
Chosen over simple-cycle gas turbine because:
  - Better part-load efficiency (important for a gap-filler that runs variably)
  - Lower minimum stable generation (~30% vs ~50% for GT)
  - Lower capital cost at datacenter-relevant scales (< 100 MW)
  - Fast start capability (warm start ~5–10 min)

Heat rate: 9.0 MMBtu/MWh  (full load, HHV basis; industry range 8.5–10.0)
Efficiency: 37.9%          (= 3.412 / 9.0;  LHV ≈ 41.5%)

Gas price: 2.50 $/MMBtu    (EIA 2024 Henry Hub avg ~$2.20 + Texas basis ~$0.30)
CO₂ factor: 0.0531 tCO₂/MMBtu  (EPA AP-42, pipeline-quality natural gas)
  → 0.478 tCO₂/MWh at full load

Minimum stable generation: 30% of G
  Below this, the engine either shuts down (P_gas = 0) or is held at minimum.
  In the dispatcher we use a binary on/off: gas either runs at ≥ P_min or is off.

LCOE note
---------
Gas LCOE for the sLCOE formula is computed externally using Lazard 18.0 +
EIA regional fuel prices. This module handles the physical dispatch only.
The annual fuel cost can be derived directly from this model's output:

    Annual_Fuel_Cost  =  sum(Fuel_t)  ×  GAS_PRICE_PER_MMBTU
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

HEAT_RATE_MMBTU_PER_MWH: float = 9.0       # HHV basis, full load
GAS_PRICE_PER_MMBTU:     float = 2.50       # $/MMBtu  (2024 Texas estimate)
CO2_FACTOR_T_PER_MMBTU:  float = 0.0531     # metric tonnes CO₂/MMBtu (EPA)
MIN_LOAD_FRACTION:       float = 0.30       # minimum stable generation (30% of G)

# Derived convenience constants
CO2_FACTOR_T_PER_MWH: float = CO2_FACTOR_T_PER_MMBTU * HEAT_RATE_MMBTU_PER_MWH
# = 0.4779 tCO₂/MWh at full load


# ---------------------------------------------------------------------------
# Core dispatch function
# ---------------------------------------------------------------------------

def dispatch_gas(
    G_mw:           float,
    unmet_demand:   np.ndarray,
    allow_partial:  bool = True,
) -> np.ndarray:
    """
    Dispatch gas generation to cover unmet demand.

        P_gas,t  =  min(G,  unmet_demand_t)   if unmet_demand_t >= P_min
                 =  0                          otherwise (below minimum stable load)

    Parameters
    ----------
    G_mw          : float      Installed gas capacity [MW]  (optimizer var)
    unmet_demand  : ndarray    Remaining load after solar + BESS [MW], shape (n,)
    allow_partial : bool       If True, gas can run below P_min (ignores minimum
                               stable load constraint). Set False for realistic
                               on/off dispatch.

    Returns
    -------
    p_gas : ndarray [MW]  Actual gas output, same shape as unmet_demand.
    """
    unmet = np.asarray(unmet_demand, dtype=float)
    p_gas = np.minimum(np.maximum(unmet, 0.0), G_mw)

    if not allow_partial:
        p_min = MIN_LOAD_FRACTION * G_mw
        # Below minimum: shut down (not worth running at very low output)
        p_gas = np.where(p_gas < p_min, 0.0, p_gas)

    return p_gas


# ---------------------------------------------------------------------------
# Fuel and emissions
# ---------------------------------------------------------------------------

def fuel_consumption(p_gas: np.ndarray) -> np.ndarray:
    """
    Hourly fuel consumption [MMBtu].

        Fuel_t  =  P_gas,t  ×  Heat_Rate

    Parameters
    ----------
    p_gas : ndarray [MW]

    Returns
    -------
    fuel : ndarray [MMBtu/h]
    """
    return np.asarray(p_gas, dtype=float) * HEAT_RATE_MMBTU_PER_MWH


def co2_emissions(p_gas: np.ndarray) -> np.ndarray:
    """
    Hourly CO₂ emissions [metric tonnes].

        CO2_t  =  P_gas,t  ×  CO2_FACTOR_T_PER_MWH
    """
    return np.asarray(p_gas, dtype=float) * CO2_FACTOR_T_PER_MWH


def fuel_cost(p_gas: np.ndarray, gas_price: float = GAS_PRICE_PER_MMBTU) -> np.ndarray:
    """Hourly fuel cost [$]."""
    return fuel_consumption(p_gas) * gas_price


# ---------------------------------------------------------------------------
# Full simulation
# ---------------------------------------------------------------------------

def simulate_gas(
    G_mw:          float,
    unmet_demand:  np.ndarray,
    gas_price:     float = GAS_PRICE_PER_MMBTU,
    allow_partial: bool  = True,
) -> pd.DataFrame | dict:
    """
    Simulate gas dispatch over a full timeseries.

    Parameters
    ----------
    G_mw         : float      Installed capacity [MW]
    unmet_demand : ndarray    Remaining load after solar + BESS [MW]
    gas_price    : float      Fuel price [$/MMBtu]
    allow_partial: bool       Allow sub-minimum generation

    Returns
    -------
    dict with keys:
        p_gas_mw       ndarray [MW]     Hourly gas output
        fuel_mmbtu     ndarray [MMBtu]  Hourly fuel consumption
        cost_usd       ndarray [$]      Hourly fuel cost
        co2_t          ndarray [tCO2]   Hourly CO₂ emissions
        unmet_mw       ndarray [MW]     Unserved demand (= unmet - p_gas, should be 0 if G sized correctly)
        summary        dict             Annual totals and averages
    """
    p_gas    = dispatch_gas(G_mw, unmet_demand, allow_partial)
    fuel     = fuel_consumption(p_gas)
    cost     = fuel_cost(p_gas, gas_price)
    co2      = co2_emissions(p_gas)
    unserved = np.maximum(np.asarray(unmet_demand, dtype=float) - p_gas, 0.0)

    hours_online    = int((p_gas > 0).sum())
    hours_at_min    = int((p_gas > 0).sum() - (p_gas >= 0.99 * G_mw).sum())

    summary = {
        "G_mw":                 G_mw,
        "annual_gen_mwh":       float(p_gas.sum()),
        "annual_fuel_mmbtu":    float(fuel.sum()),
        "annual_fuel_cost_usd": float(cost.sum()),
        "annual_co2_t":         float(co2.sum()),
        "hours_online":         hours_online,
        "hours_at_capacity":    int((p_gas >= 0.99 * G_mw).sum()),
        "capacity_factor_pct":  float(p_gas.mean() / G_mw * 100) if G_mw > 0 else 0,
        "lcoe_fuel_usd_mwh":    float(cost.sum() / p_gas.sum()) if p_gas.sum() > 0 else 0,
        "unserved_energy_mwh":  float(unserved.sum()),
        "peak_unserved_mw":     float(unserved.max()),
    }

    return {
        "p_gas_mw":   p_gas,
        "fuel_mmbtu": fuel,
        "cost_usd":   cost,
        "co2_t":      co2,
        "unmet_mw":   unserved,
        "summary":    summary,
    }


# ---------------------------------------------------------------------------
# Timeseries builder  (pre-computes load-only baseline for sizing analysis)
# ---------------------------------------------------------------------------

def build_gas_sizing_profile(demand_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a worst-case gas sizing profile: assumes no solar, no BESS.
    This equals the full load profile and shows the theoretical maximum
    gas demand at each hour — useful for sizing G in the optimizer.

    Parameters
    ----------
    demand_df : pd.DataFrame  Must have column ``total_load_mw``.

    Returns
    -------
    pd.DataFrame with columns:
        total_load_mw      MW   Facility load
        max_gas_need_mw    MW   Gas need if solar=0, BESS=0 (= total_load_mw)
        load_percentile    %    Percentile rank of each hour's load
    """
    df = demand_df[["total_load_mw"]].copy()
    df["max_gas_need_mw"] = df["total_load_mw"]
    df["load_percentile"] = df["total_load_mw"].rank(pct=True) * 100
    return df


def gas_summary(G_mw: float, demand_df: pd.DataFrame) -> dict:
    """
    Compute sizing statistics for a given G capacity against the full load.
    Assumes no solar/BESS (worst case) to show the maximum role gas could play.
    """
    load = demand_df["total_load_mw"].to_numpy()
    result = simulate_gas(G_mw, load)
    s = result["summary"]

    # Add coverage metrics
    s["load_fully_covered_pct"] = float(
        ((load - result["p_gas_mw"]) <= 0.01).mean() * 100
    )
    s["gas_price_per_mmbtu"] = GAS_PRICE_PER_MMBTU
    s["heat_rate_mmbtu_per_mwh"] = HEAT_RATE_MMBTU_PER_MWH
    return s
