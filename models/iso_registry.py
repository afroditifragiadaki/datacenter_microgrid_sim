"""
ISO/RTO registry — lookup table for indicative datacenter locations and
regional cost parameters.

Usage
-----
    from models.iso_registry import get_iso, get_costs, list_isos, get_all_isos

    iso = get_iso("CAISO")
    costs = get_costs()
    annualized = annualized_costs(iso, costs, S_mw=150, B_mwh=100, G_mw=68)
"""

import json
from pathlib import Path
from functools import lru_cache

_CONFIG_DIR = Path(__file__).parent.parent / "config"
_REGISTRY_PATH = _CONFIG_DIR / "iso_registry.json"
_COSTS_PATH = _CONFIG_DIR / "technology_costs.json"


# ---------------------------------------------------------------------------
# Loaders (cached — files read once per process)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_registry() -> dict:
    with open(_REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_costs() -> dict:
    with open(_COSTS_PATH, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_isos() -> list[str]:
    """Return sorted list of ISO keys, e.g. ['CAISO', 'ERCOT', ...]."""
    return sorted(_load_registry()["isos"].keys())


def get_iso(iso_id: str) -> dict:
    """
    Return config dict for one ISO.

    Keys: name, city, state, lat, lon, timezone, pv_tilt_deg, pv_azimuth_deg,
          gas_price_per_mmbtu, capex_multiplier, notes
    """
    reg = _load_registry()
    iso_id = iso_id.upper()
    if iso_id not in reg["isos"]:
        raise KeyError(f"Unknown ISO '{iso_id}'. Available: {list(reg['isos'].keys())}")
    return reg["isos"][iso_id]


def get_all_isos() -> dict:
    """Return full {iso_id: config_dict} mapping."""
    return _load_registry()["isos"]


def get_costs() -> dict:
    """Return technology cost baselines (solar_pv, bess, gas_rice, common)."""
    return _load_costs()


# ---------------------------------------------------------------------------
# Financial helpers
# ---------------------------------------------------------------------------

def _crf(rate: float, n_years: int) -> float:
    """Capital Recovery Factor — converts capex to equal annual payment."""
    if rate == 0:
        return 1.0 / n_years
    return rate * (1 + rate) ** n_years / ((1 + rate) ** n_years - 1)


def annualized_costs(iso: dict, costs: dict, S_mw: float, B_mwh: float, G_mw: float) -> dict:
    """
    Compute annualized CAPEX + fixed OPEX for each technology at a given
    (S, B, G) system size.

    Parameters
    ----------
    iso    : ISO config dict from get_iso()
    costs  : Technology costs dict from get_costs()
    S_mw   : Solar DC nameplate (MW)
    B_mwh  : Battery energy capacity (MWh)
    G_mw   : Gas generator nameplate (MW)

    Returns
    -------
    dict with keys:
        solar_capex_ann, solar_opex_ann,
        bess_capex_ann,  bess_opex_ann,
        gas_capex_ann,   gas_opex_ann,
        total_fixed_ann
    """
    mult = iso["capex_multiplier"]
    r = costs["common"]["discount_rate"]
    life = costs["common"]["project_life_yr"]
    crf_project = _crf(r, life)

    # --- Solar ---
    sol = costs["solar_pv"]
    sol_capex_kw = sol["capex_per_kw_dc"] * mult
    sol_life = sol["lifetime_yr"]
    # If asset life < project life, add replacement at year sol_life
    sol_crf = _crf(r, sol_life)
    solar_capex_ann = sol_capex_kw * S_mw * 1000 * sol_crf      # $/yr
    solar_opex_ann  = sol["opex_per_kw_yr"] * S_mw * 1000        # $/yr

    # --- BESS ---
    bess = costs["bess"]
    bess_capex_kwh = bess["capex_per_kwh"] * mult
    bess_life = bess["lifetime_yr"]
    bess_crf = _crf(r, bess_life)
    bess_capex_ann = bess_capex_kwh * B_mwh * 1000 * bess_crf    # $/yr
    # Augmentation (cell replacement) as % of capex per year
    bess_aug_ann   = bess_capex_kwh * B_mwh * 1000 * (bess["augmentation_pct_yr"] / 100)
    bess_opex_ann  = bess["opex_per_kw_yr"] * (B_mwh / bess["unit_size_mwh"]) * bess["unit_power_mw"] * 1000 + bess_aug_ann

    # --- Gas RICE ---
    gas = costs["gas_rice"]
    gas_capex_kw = gas["capex_per_kw"] * mult
    gas_life = gas["lifetime_yr"]
    gas_crf = _crf(r, gas_life)
    gas_capex_ann  = gas_capex_kw * G_mw * 1000 * gas_crf        # $/yr
    gas_opex_ann   = gas["opex_fixed_per_kw_yr"] * G_mw * 1000   # $/yr

    total_fixed_ann = (
        solar_capex_ann + solar_opex_ann +
        bess_capex_ann  + bess_opex_ann  +
        gas_capex_ann   + gas_opex_ann
    )

    return {
        "solar_capex_ann": solar_capex_ann,
        "solar_opex_ann":  solar_opex_ann,
        "bess_capex_ann":  bess_capex_ann,
        "bess_opex_ann":   bess_opex_ann,
        "gas_capex_ann":   gas_capex_ann,
        "gas_opex_ann":    gas_opex_ann,
        "total_fixed_ann": total_fixed_ann,
    }


def fuel_cost_annual(iso: dict, costs: dict, gas_gen_mwh_yr: float) -> float:
    """Annual gas fuel cost in $ given ISO gas price and total gas generation."""
    heat_rate = costs["gas_rice"]["heat_rate_mmbtu_per_mwh"]
    gas_price = iso["gas_price_per_mmbtu"]
    return gas_gen_mwh_yr * heat_rate * gas_price


def compute_slcoe(
    iso: dict,
    costs: dict,
    S_mw: float,
    B_mwh: float,
    G_mw: float,
    gas_gen_mwh_yr: float,
    annual_load_mwh: float,
    gas_var_opex_mwh_yr: float = 0.0,
) -> dict:
    """
    Compute system LCOE ($/MWh) for a given (S, B, G, gas_gen) configuration.

    Parameters
    ----------
    gas_gen_mwh_yr       : total gas energy generated in the year (MWh)
    annual_load_mwh      : total datacenter energy consumed (MWh) — denominator
    gas_var_opex_mwh_yr  : variable O&M on gas generation ($/MWh × MWh/yr = $/yr)

    Returns
    -------
    dict: slcoe_per_mwh, fuel_cost_ann, var_opex_ann, total_annual_cost, breakdown
    """
    fixed = annualized_costs(iso, costs, S_mw, B_mwh, G_mw)
    fuel  = fuel_cost_annual(iso, costs, gas_gen_mwh_yr)

    gas_var = (
        gas_var_opex_mwh_yr
        if gas_var_opex_mwh_yr
        else costs["gas_rice"]["opex_variable_per_mwh"] * gas_gen_mwh_yr
    )

    total = fixed["total_fixed_ann"] + fuel + gas_var

    slcoe = total / annual_load_mwh if annual_load_mwh > 0 else float("nan")

    return {
        "slcoe_per_mwh":    slcoe,
        "total_annual_cost": total,
        "fuel_cost_ann":    fuel,
        "gas_var_opex_ann": gas_var,
        **fixed,
    }


def compute_slcoe_grid(
    iso: dict,
    costs: dict,
    S_mw: float,
    B_mwh: float,
    G_mw: float,
    gas_gen_mwh_yr: float,
    grid_import_cost_yr: float,
    grid_import_mwh_yr: float,
    peak_grid_import_mw: float,
    annual_load_mwh: float,
) -> dict:
    """
    Compute grid-connected system LCOE ($/MWh).

    Total cost = solar (capex + opex)
               + bess  (capex + opex)
               + gas   (capex + opex + fuel + var_opex)
               + grid interconnect (capex + opex)  ← sized to peak_grid_import_mw
               + grid import energy cost            ← Σ max(0, price[t]) × import[t]

    Parameters
    ----------
    grid_import_cost_yr   : annual grid energy purchase cost ($), using
                            clipped-positive prices (no revenue from negative prices)
    grid_import_mwh_yr    : annual grid energy imported (MWh) — reported metric
    peak_grid_import_mw   : peak hourly grid import from dispatch (MW)
                            used to size the interconnection infrastructure
    """
    fixed   = annualized_costs(iso, costs, S_mw, B_mwh, G_mw)
    fuel    = fuel_cost_annual(iso, costs, gas_gen_mwh_yr)
    gas_var = costs["gas_rice"]["opex_variable_per_mwh"] * gas_gen_mwh_yr

    grid_cfg       = costs["grid_interconnect"]
    mult           = iso["capex_multiplier"]
    grid_cap_kw    = peak_grid_import_mw * 1000
    grid_capex     = grid_cfg["capex_per_kw"] * grid_cap_kw * mult
    grid_crf       = _crf(costs["common"]["discount_rate"], grid_cfg["lifetime_yr"])
    grid_capex_ann = grid_capex * grid_crf
    grid_opex_ann  = grid_cfg["opex_per_kw_yr"] * grid_cap_kw

    total = (
        fixed["total_fixed_ann"]
        + fuel
        + gas_var
        + grid_capex_ann
        + grid_opex_ann
        + grid_import_cost_yr
    )

    slcoe = total / annual_load_mwh if annual_load_mwh > 0 else float("nan")

    return {
        "slcoe_per_mwh":          slcoe,
        "total_annual_cost":      total,
        "fuel_cost_ann":          fuel,
        "gas_var_opex_ann":       gas_var,
        "grid_capex_ann":         grid_capex_ann,
        "grid_opex_ann":          grid_opex_ann,
        "grid_import_cost_ann":   grid_import_cost_yr,
        "grid_import_mwh_yr":     grid_import_mwh_yr,
        **fixed,
    }


# ---------------------------------------------------------------------------
# Quick summary table (useful for Streamlit display)
# ---------------------------------------------------------------------------

def registry_dataframe():
    """Return a pandas DataFrame of all ISOs for display in dashboards."""
    import pandas as pd
    isos = get_all_isos()
    rows = []
    for iso_id, cfg in isos.items():
        rows.append({
            "ISO":              iso_id,
            "Name":             cfg["name"],
            "City":             cfg["city"],
            "Lat":              cfg["lat"],
            "Lon":              cfg["lon"],
            "Gas ($/MMBtu)":    cfg["gas_price_per_mmbtu"],
            "CAPEX Mult.":      cfg["capex_multiplier"],
            "Notes":            cfg["notes"],
        })
    return pd.DataFrame(rows).set_index("ISO")
