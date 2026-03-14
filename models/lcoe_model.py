"""
Financial model for the datacenter microgrid sLCOE calculation.

Sources
-------
- Capital costs    : NREL ATB 2025 (moderate scenario, Texas region)
- Standalone LCOEs : Lazard LCOE+ 18.0 (June 2025), unsubsidized
- Fuel prices      : EIA 2026 STEO regional forecasts (Texas/ERCOT)
- Discount rate    : 7% nominal WACC (utility-scale project finance)

sLCOE formula (from methodology document)
-----------------------------------------
    sLCOE = (LCOE_solar * E_solar  +  LCOE_bess * E_bess  +  LCOE_gas * E_gas)
            / Total_demand_MWh

Implemented here as first-principles annualized costs (equivalent to Lazard
standalone LCOE approach when capacity factors match the simulated dispatch):

    Total_annual_cost = C_solar + C_bess + C_gas
    sLCOE             = Total_annual_cost / Annual_demand_MWh

Where each component's annual cost = annualized_capex + fixed_opex + variable_opex.

ITC note: All costs are UNSUBSIDIZED (pre-incentive), consistent with Lazard
18.0 baseline. The 30% IRA 2022 Investment Tax Credit would reduce effective
solar + BESS capex by ~30% for a tax-equity eligible owner; this is tracked
separately in the sensitivity analysis.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Project-level financial parameters
# ---------------------------------------------------------------------------

DISCOUNT_RATE:    float = 0.070   # 7%  nominal WACC (utility-scale, unlevered)
PROJECT_LIFE_YR:  int   = 25      # years

# ---------------------------------------------------------------------------
# Solar PV  —  Utility-scale, fixed-tilt, Texas
# NREL ATB 2025 moderate scenario; Texas region ~5% below national average
# Lazard LCOE+ 18.0 confirms $24-96/MWh range (unsubsidized)
# ---------------------------------------------------------------------------

SOLAR_CAPEX_PER_KW:    float = 950.0    # $/kW DC  (NREL ATB 2025 moderate, Texas)
SOLAR_OPEX_FIXED_PER_KW_YR: float = 16.0 # $/kW-DC/yr  fixed O&M (NREL ATB 2025)
SOLAR_LIFE_YR:         int   = 30       # exceeds project life; no replacement needed

# ---------------------------------------------------------------------------
# BESS  —  4-hour Li-ion, utility-scale
# NREL ATB 2025 moderate scenario; Lazard LCOS 10.0 confirms range
# ---------------------------------------------------------------------------

BESS_CAPEX_PER_KWH:    float = 300.0   # $/kWh energy capacity  (NREL ATB 2025 moderate)
BESS_OPEX_FIXED_PER_KW_YR: float = 8.0 # $/kW-power/yr  fixed O&M  (NREL ATB 2025)
BESS_OPEX_VAR_PER_MWH: float = 0.50    # $/MWh discharged  variable O&M
BESS_LIFE_YR:          int   = 15      # one replacement at year 15 within 25-yr project
# BESS replacement cost factor at year 15 (technology learning curve)
BESS_REPLACE_COST_FACTOR: float = 0.85  # 15% cheaper by 2040 (NREL ATB mid learning)

# ---------------------------------------------------------------------------
# Gas  —  Natural Gas Reciprocating Engine (RICE)
# Lazard LCOE+ 18.0 peaker range; RICE at datacenter scale (<150 MW)
# ---------------------------------------------------------------------------

GAS_CAPEX_PER_KW:      float = 1_200.0  # $/kW  (Lazard 18.0 peaker; RICE mid-range)
GAS_OPEX_FIXED_PER_KW_YR: float = 20.0 # $/kW/yr  fixed O&M  (Lazard 18.0 peaker)
GAS_OPEX_VAR_PER_MWH:  float = 7.00    # $/MWh   variable O&M (Lazard 18.0 peaker)
GAS_LIFE_YR:           int   = 25      # matches project life; no replacement

# ITC / IRA 2022 Investment Tax Credit (30% for solar + storage)
ITC_RATE: float = 0.30   # applied to eligible capex for sensitivity analysis


# ---------------------------------------------------------------------------
# Core financial functions
# ---------------------------------------------------------------------------

def crf(n_years: int, rate: float) -> float:
    """Capital Recovery Factor: annualizes a lump-sum NPV over n_years at rate."""
    if rate == 0:
        return 1.0 / n_years
    return rate * (1 + rate) ** n_years / ((1 + rate) ** n_years - 1)


def pv_factor(year: int, rate: float) -> float:
    """Present-value discount factor for a cash flow at `year`."""
    return 1.0 / (1 + rate) ** year


# ---------------------------------------------------------------------------
# Annual cost per technology
# ---------------------------------------------------------------------------

def annual_solar_cost(S_mw: float, rate: float = DISCOUNT_RATE) -> float:
    """
    Total annual cost of the solar plant [$].

    Parameters
    ----------
    S_mw : Solar DC capacity [MW]

    Returns
    -------
    Annual cost [$]  (capex annuity + fixed O&M)
    """
    if S_mw <= 0:
        return 0.0

    capex        = SOLAR_CAPEX_PER_KW * S_mw * 1_000          # $ (kW -> MW)
    capex_annual = capex * crf(PROJECT_LIFE_YR, rate)
    opex_fixed   = SOLAR_OPEX_FIXED_PER_KW_YR * S_mw * 1_000  # $/yr

    return capex_annual + opex_fixed


def annual_bess_cost(
    B_mwh: float,
    bess_discharge_mwh: float = 0.0,
    rate: float = DISCOUNT_RATE,
) -> float:
    """
    Total annual cost of the BESS [$].

    Accounts for one mid-life replacement at BESS_LIFE_YR (year 15).

    Parameters
    ----------
    B_mwh              : Battery energy capacity [MWh]
    bess_discharge_mwh : Annual MWh discharged (for variable O&M)

    Returns
    -------
    Annual cost [$]
    """
    if B_mwh <= 0:
        return 0.0

    P_mw         = B_mwh / 4.0                                 # 4-hour battery → power [MW]
    capex_0      = BESS_CAPEX_PER_KWH * B_mwh * 1_000          # $ initial
    capex_15     = capex_0 * BESS_REPLACE_COST_FACTOR           # $ replacement at year 15
    npv_capex    = capex_0 + capex_15 * pv_factor(BESS_LIFE_YR, rate)
    capex_annual = npv_capex * crf(PROJECT_LIFE_YR, rate)

    opex_fixed   = BESS_OPEX_FIXED_PER_KW_YR * P_mw * 1_000   # $/yr
    opex_var     = BESS_OPEX_VAR_PER_MWH * bess_discharge_mwh  # $/yr

    return capex_annual + opex_fixed + opex_var


def annual_gas_cost(
    G_mw: float,
    gas_gen_mwh: float = 0.0,
    rate: float = DISCOUNT_RATE,
) -> float:
    """
    Total annual cost of the gas plant [$].

    Parameters
    ----------
    G_mw        : Installed gas capacity [MW]
    gas_gen_mwh : Annual MWh generated (for variable O&M + fuel)

    Returns
    -------
    Annual cost [$]
    """
    if G_mw <= 0:
        return 0.0

    from models.gas_model import HEAT_RATE_MMBTU_PER_MWH, GAS_PRICE_PER_MMBTU

    capex        = GAS_CAPEX_PER_KW * G_mw * 1_000             # $
    capex_annual = capex * crf(PROJECT_LIFE_YR, rate)
    opex_fixed   = GAS_OPEX_FIXED_PER_KW_YR * G_mw * 1_000    # $/yr
    fuel_per_mwh = HEAT_RATE_MMBTU_PER_MWH * GAS_PRICE_PER_MMBTU
    opex_var     = (GAS_OPEX_VAR_PER_MWH + fuel_per_mwh) * gas_gen_mwh  # $/yr

    return capex_annual + opex_fixed + opex_var


def system_slcoe(
    S_mw: float,
    B_mwh: float,
    G_mw: float,
    summary: dict,
    rate: float = DISCOUNT_RATE,
) -> dict:
    """
    Compute system LCOE for one (S, B, G) configuration.

    Parameters
    ----------
    S_mw, B_mwh, G_mw : Installed capacities
    summary            : Annual KPI dict returned by dispatcher._summarise()

    Returns
    -------
    dict with sLCOE breakdown:
        slcoe_per_mwh       $/MWh  total system LCOE
        solar_cost_usd_yr   $/yr   annualized solar cost
        bess_cost_usd_yr    $/yr   annualized BESS cost
        gas_cost_usd_yr     $/yr   annualized gas cost
        total_cost_usd_yr   $/yr   total annual cost
        demand_mwh_yr       MWh/yr annual demand
        solar_lcoe_contrib  $/MWh  solar contribution to sLCOE
        bess_lcoe_contrib   $/MWh  BESS contribution to sLCOE
        gas_lcoe_contrib    $/MWh  gas contribution to sLCOE
    """
    demand    = summary["total_demand_mwh"]
    bess_dis  = summary["bess_discharge_mwh"]
    gas_gen   = summary["gas_gen_mwh"]

    c_solar = annual_solar_cost(S_mw, rate)
    c_bess  = annual_bess_cost(B_mwh, bess_dis, rate)
    c_gas   = annual_gas_cost(G_mw, gas_gen, rate)
    c_total = c_solar + c_bess + c_gas

    slcoe   = c_total / demand if demand > 0 else 0.0

    return {
        "slcoe_per_mwh":      round(slcoe, 4),
        "solar_cost_usd_yr":  round(c_solar, 0),
        "bess_cost_usd_yr":   round(c_bess, 0),
        "gas_cost_usd_yr":    round(c_gas, 0),
        "total_cost_usd_yr":  round(c_total, 0),
        "demand_mwh_yr":      round(demand, 0),
        "solar_lcoe_contrib": round(c_solar / demand, 4),
        "bess_lcoe_contrib":  round(c_bess  / demand, 4),
        "gas_lcoe_contrib":   round(c_gas   / demand, 4),
    }


def print_financial_assumptions() -> None:
    """Print all financial parameters in a formatted table."""
    r = DISCOUNT_RATE
    n = PROJECT_LIFE_YR
    print(f"  Discount rate (WACC) : {r*100:.1f}%")
    print(f"  Project lifetime     : {n} years")
    print(f"  CRF ({n}yr, {r*100:.0f}%)     : {crf(n, r):.5f}  ({crf(n,r)*100:.3f}%/yr)")
    print()
    print(f"  Solar PV  (fixed-tilt, Texas — NREL ATB 2025 moderate)")
    print(f"    CAPEX            : ${SOLAR_CAPEX_PER_KW:>8,.0f} /kW DC")
    print(f"    Fixed O&M        : ${SOLAR_OPEX_FIXED_PER_KW_YR:>8,.0f} /kW/yr")
    print(f"    Life             : {SOLAR_LIFE_YR} yr  (no replacement in 25-yr project)")
    print()
    print(f"  BESS  (4-hour Li-ion, utility — NREL ATB 2025 moderate)")
    print(f"    CAPEX            : ${BESS_CAPEX_PER_KWH:>8,.0f} /kWh")
    print(f"    Fixed O&M        : ${BESS_OPEX_FIXED_PER_KW_YR:>8,.0f} /kW-power/yr")
    print(f"    Variable O&M     : ${BESS_OPEX_VAR_PER_MWH:>8,.2f} /MWh discharged")
    print(f"    Life             : {BESS_LIFE_YR} yr  (one replacement at yr {BESS_LIFE_YR}, "
          f"{BESS_REPLACE_COST_FACTOR*100:.0f}% of initial capex)")
    print()
    print(f"  Gas RICE  (Lazard LCOE+ 18.0 peaker, unsubsidized)")
    print(f"    CAPEX            : ${GAS_CAPEX_PER_KW:>8,.0f} /kW")
    print(f"    Fixed O&M        : ${GAS_OPEX_FIXED_PER_KW_YR:>8,.0f} /kW/yr")
    print(f"    Variable O&M     : ${GAS_OPEX_VAR_PER_MWH:>8,.2f} /MWh")
    print(f"    Fuel (9.0 MMBtu/MWh x $2.50/MMBtu): ${9.0*2.50:>5.2f} /MWh")
    print(f"    Total var cost   : ${GAS_OPEX_VAR_PER_MWH + 9.0*2.50:>8.2f} /MWh")
    print(f"    Life             : {GAS_LIFE_YR} yr")
    print()
    print(f"  ITC/IRA 2022 (noted but NOT applied to baseline):")
    print(f"    ITC rate         : {ITC_RATE*100:.0f}%  (applies to solar + BESS capex)")
