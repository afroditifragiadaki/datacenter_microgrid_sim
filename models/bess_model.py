"""
Battery Energy Storage System (BESS) model for a collocated datacenter microgrid.

State update equation (from methodology document)
--------------------------------------------------
    E_t = E_{t-1} + [P_charge,t  × (η_ch  × η_temp,t)]
                  − [P_discharge,t / (η_temp,t × η_dis)]

    Subject to:  0.20 · B  ≤  E_t  ≤  B       (DoD / energy limits)
                 0          ≤  P_charge,t  ≤  P_max
                 0          ≤  P_discharge,t ≤  P_max

Where:
    B           : Battery energy capacity [MWh]         — optimizer decision variable
    P_max       : Maximum charge/discharge power [MW]   — derived as B / DURATION_H
    η_ch        : One-way charge efficiency              — 0.96
    η_dis       : One-way discharge efficiency           — 0.96
    η_temp,t    : Temperature-dependent PHVAC modifier  — see below

Temperature efficiency model (η_temp)
--------------------------------------
Utility-scale Li-ion BESS includes an HVAC system that keeps cells within their
optimal temperature window (≈ 15–35 °C). Outside this window, the HVAC draws
a parasitic load that reduces the effective net efficiency of every charge and
discharge cycle. The impact is symmetric but asymmetric in magnitude:

    Cold side: resistive heating needed — larger efficiency hit per degree
    Hot side:  compressor cooling needed — smaller but still meaningful hit

Control points (ambient °C → η_temp):
    ≤ −10 °C   →  0.94   (heavy heating load, ~6 % parasitic)
      0 °C     →  0.97   (moderate heating, ~3 %)
     15 °C     →  1.00   (lower edge of comfort zone, HVAC off)
     35 °C     →  1.00   (upper edge of comfort zone, HVAC off)
     45 °C     →  0.96   (cooling compressor active, ~4 %)
    ≥ 60 °C    →  0.92   (extreme cooling, ~8 % — unlikely for DFW)

For ERCOT (DFW): the dominant concern is summer heat (> 35 °C ≈ 3 % of hours).
Cold below 0 °C occurs only in extreme winter events (< 1 % of hours).
Mean ambient (20 °C) sits comfortably in the optimal zone → η_temp ≈ 1.00 for
most of the year, with small seasonal corrections.

Round-trip efficiency at optimal temperature:
    η_rt = η_ch × η_dis = 0.96 × 0.96 = 0.9216   (92.2 %)

Round-trip at ERCOT summer peak (40 °C, η_temp ≈ 0.98):
    Effective charge  : 0.96 × 0.98 = 0.940
    Effective discharge: 0.98 × 0.96 = 0.940
    η_rt_effective    ≈ 0.940 × 0.940 = 0.884   (88.4 %)
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Base efficiency constants
# ---------------------------------------------------------------------------

ETA_CH:  float = 0.96   # one-way charge efficiency   (Li-ion NMC/LFP, utility scale)
ETA_DIS: float = 0.96   # one-way discharge efficiency

SOC_MIN: float = 0.20   # minimum state-of-charge (80 % depth of discharge)
SOC_MAX: float = 1.00   # maximum state-of-charge

DURATION_H: float = 4.0  # default energy-to-power ratio [hours]
                          # P_max [MW] = B [MWh] / DURATION_H
                          # (4-hour battery is the current utility-scale standard)

# ---------------------------------------------------------------------------
# Standard commercial BESS unit (container) specification
# ---------------------------------------------------------------------------
# Real installations are built from discrete containerised units rather than
# one monolithic bank.  Defining a standard unit size:
#   • allows the grid search to use integer unit counts (N_units)
#   • gives a realistic upper bound (a single 1,000 MWh bank does not exist)
#   • enables multi-unit dispatch: different units can be at different SoC
#     levels, so one group charges from solar surplus while another holds
#     charge ready for the next deficit hour.
#
# Reference: ~4 MWh per container is consistent with utility-scale products
# such as the Tesla Megapack 2 (3.9 MWh) and CATL EnerC+ (3.44 MWh).
# We round to 4 MWh for clean arithmetic.
#
B_UNIT_MWH:  float = 4.0    # Energy per BESS container unit [MWh]
P_UNIT_MW:   float = B_UNIT_MWH / DURATION_H   # Power per unit = 1.0 MW
N_UNITS_MAX: int   = 100    # Maximum realistic number of units for this site

# ---------------------------------------------------------------------------
# Temperature efficiency model — piecewise-linear PHVAC correction
# ---------------------------------------------------------------------------

# Ambient temperature knots [°C]
_T_KNOTS   = np.array([-10.0,  0.0, 15.0, 35.0, 45.0, 60.0], dtype=float)

# Corresponding η_temp values (PHVAC parasitic load subtracted from 1.0)
_ETA_KNOTS = np.array([  0.94, 0.97, 1.00, 1.00, 0.96, 0.92], dtype=float)


def eta_temp_from_tamb(temp_c: "np.ndarray | float") -> "np.ndarray | float":
    """
    Compute the PHVAC temperature-efficiency modifier from ambient temperature.

    Returns η_temp ∈ [0.92, 1.00], same shape as input.
    Values below −10 °C clamp to 0.94; values above 60 °C clamp to 0.92.

    Usage in state equation:
        Energy into battery  per MW charged   = η_ch  × η_temp
        Energy from battery per MW discharged = η_dis × η_temp
    """
    return np.interp(temp_c, _T_KNOTS, _ETA_KNOTS)


# ---------------------------------------------------------------------------
# BESS simulation (called by the dispatcher)
# ---------------------------------------------------------------------------

def simulate_bess(
    B_mwh:       float,
    p_charge:    np.ndarray,
    p_discharge: np.ndarray,
    eta_temp:    np.ndarray,
    E0:          float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate BESS operation over a timeseries using the methodology equation:

        E_t = E_{t-1} + P_charge_t × (η_ch × η_temp_t)
                      − P_discharge_t / (η_temp_t × η_dis)

    Power inputs are clipped to hardware limits and SoC constraints
    each timestep, returning the actually-delivered charge/discharge.

    Parameters
    ----------
    B_mwh       : float          Battery energy capacity [MWh]  (optimizer var)
    p_charge    : ndarray [MW]   Requested charge power  (non-negative)
    p_discharge : ndarray [MW]   Requested discharge power (non-negative)
    eta_temp    : ndarray [–]    Hourly temperature modifier (from eta_temp_from_tamb)
    E0          : float | None   Initial SoC [MWh].  Default: 50 % of B.

    Returns
    -------
    soc         : ndarray [MWh]  State-of-charge at end of each hour  (n,)
    p_ch_actual : ndarray [MW]   Actual charge power delivered         (n,)
    p_dis_actual: ndarray [MW]   Actual discharge power delivered      (n,)
    """
    n     = len(p_charge)
    E_min = SOC_MIN * B_mwh
    E_max = SOC_MAX * B_mwh
    P_max = B_mwh / DURATION_H          # MW limit from energy-to-power ratio

    soc          = np.empty(n, dtype=float)
    p_ch_actual  = np.zeros(n, dtype=float)
    p_dis_actual = np.zeros(n, dtype=float)

    E = E0 if E0 is not None else 0.50 * B_mwh

    for t in range(n):
        et  = eta_temp[t]

        # Clamp requested powers to hardware limit
        pc  = min(float(p_charge[t]),    P_max)
        pd  = min(float(p_discharge[t]), P_max)

        # Energy balance
        delta_charge    =  pc * ETA_CH  * et            # energy into battery
        delta_discharge =  pd / (et * ETA_DIS)          # energy drawn from battery

        E_new = E + delta_charge - delta_discharge

        # Apply SoC constraints: scale back the offending flow
        if E_new > E_max:
            # Headroom for charging
            headroom = E_max - E + delta_discharge
            pc = max(headroom / (ETA_CH * et), 0.0)
            delta_charge = pc * ETA_CH * et
            E_new = E_max

        elif E_new < E_min:
            # Available energy for discharging
            available = E - E_min + delta_charge
            pd = max(available * et * ETA_DIS, 0.0)
            delta_discharge = pd / (et * ETA_DIS)
            E_new = E_min

        E = E_new
        soc[t]          = E
        p_ch_actual[t]  = pc
        p_dis_actual[t] = pd

    return soc, p_ch_actual, p_dis_actual


# ---------------------------------------------------------------------------
# Timeseries builder (called from scripts/03_build_bess.py)
# ---------------------------------------------------------------------------

def build_bess_params_timeseries(temp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the hourly BESS parameter timeseries from ambient temperature.

    Parameters
    ----------
    temp_df : pd.DataFrame
        Must have a DatetimeIndex and a column ``temp_c`` (°C).

    Returns
    -------
    pd.DataFrame with columns:
        temp_c      °C      Ambient temperature
        eta_temp    –       PHVAC temperature modifier ∈ [0.92, 1.00]
        eta_ch_eff  –       Effective charge efficiency  = η_ch  × η_temp
        eta_dis_eff –       Effective discharge efficiency = η_dis × η_temp
        eta_rt_eff  –       Effective round-trip efficiency = η_ch_eff × η_dis_eff
        phvac_loss  %       PHVAC parasitic load as percentage  = (1 − η_temp) × 100
    """
    if "temp_c" not in temp_df.columns:
        raise ValueError("temp_df must contain a 'temp_c' column.")

    df = temp_df[["temp_c"]].copy()
    df["eta_temp"]    = eta_temp_from_tamb(df["temp_c"].to_numpy())
    df["eta_ch_eff"]  = ETA_CH  * df["eta_temp"]
    df["eta_dis_eff"] = ETA_DIS * df["eta_temp"]
    df["eta_rt_eff"]  = df["eta_ch_eff"] * df["eta_dis_eff"]
    df["phvac_loss"]  = (1.0 - df["eta_temp"]) * 100.0
    return df


def bess_params_summary(bess_df: pd.DataFrame) -> dict:
    """Summary statistics for the BESS parameter timeseries."""
    return {
        "hours":               len(bess_df),
        "mean_eta_temp":       float(bess_df["eta_temp"].mean()),
        "min_eta_temp":        float(bess_df["eta_temp"].min()),
        "hours_hvac_cold":     int((bess_df["temp_c"] < 15).sum()),
        "hours_hvac_hot":      int((bess_df["temp_c"] > 35).sum()),
        "hours_optimal":       int(((bess_df["temp_c"] >= 15) & (bess_df["temp_c"] <= 35)).sum()),
        "mean_eta_rt":         float(bess_df["eta_rt_eff"].mean()),
        "min_eta_rt":          float(bess_df["eta_rt_eff"].min()),
        "base_eta_rt":         ETA_CH * ETA_DIS,
        "mean_phvac_loss_pct": float(bess_df["phvac_loss"].mean()),
    }
