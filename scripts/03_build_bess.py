"""
Build the 2024 hourly BESS parameter timeseries for ERCOT.

Since B (battery capacity) is an optimizer decision variable, this script
does NOT simulate actual BESS operation. Instead it builds the temperature-
dependent efficiency profile (eta_temp) that the dispatcher/optimizer will
consume at runtime.

Temperature source
------------------
Actual 2024 Open-Meteo temperatures from the demand timeseries (same basis
year as the demand model). This is preferred over the PVWatts TMY tamb so
that the BESS thermal losses are consistent with the 2024 demand simulation.

Outputs
-------
    data/processed/ercot_bess_params_2024.csv   hourly eta_temp + derived efficiencies
    data/processed/ercot_bess_params_2024.png   diagnostic plots
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.bess_model import (
    build_bess_params_timeseries,
    bess_params_summary,
    eta_temp_from_tamb,
    ETA_CH, ETA_DIS, SOC_MIN, SOC_MAX, DURATION_H,
    _T_KNOTS, _ETA_KNOTS,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TARGET_YEAR   = 2024
LOCATION_NAME = "Dallas-Fort Worth, TX"
OUT_DIR = PROJECT_ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> pd.DataFrame:
    print("=" * 60)
    print("ERCOT BESS Parameter Timeseries Builder")
    print("=" * 60)

    # 1. Load temperature from demand timeseries (actual 2024)
    demand_path = OUT_DIR / f"ercot_demand_{TARGET_YEAR}.csv"
    if not demand_path.exists():
        raise FileNotFoundError(
            f"{demand_path} not found. Run scripts/01_build_demand.py first."
        )

    print(f"\n[1/4] Loading temperature from demand timeseries ...")
    demand = pd.read_csv(demand_path, index_col="datetime", parse_dates=True)
    temp_df = demand[["temp_c"]].copy()
    print(f"  Rows       : {len(temp_df):,}  ({len(temp_df)/24:.0f} days)")
    print(f"  Temp range : {temp_df['temp_c'].min():.1f} to {temp_df['temp_c'].max():.1f} degC")
    print(f"  Temp mean  : {temp_df['temp_c'].mean():.1f} degC")

    # 2. Build BESS parameter timeseries
    print(f"\n[2/4] Computing eta_temp timeseries ...")
    print(f"  Base efficiencies : eta_ch={ETA_CH}, eta_dis={ETA_DIS}")
    print(f"  Round-trip (base) : {ETA_CH*ETA_DIS:.4f}  ({ETA_CH*ETA_DIS*100:.1f}%)")
    print(f"  SoC limits        : [{SOC_MIN*100:.0f}%, {SOC_MAX*100:.0f}%]  (80% DoD)")
    print(f"  Power rating      : B_MWh / {DURATION_H:.0f}h  (4-hour battery)")
    print(f"  eta_temp model    : PHVAC parasitic load, piecewise-linear")
    print(f"    Control points:")
    for t, e in zip(_T_KNOTS, _ETA_KNOTS):
        phvac = (1 - e) * 100
        label = "cold HVAC" if t < 15 else ("optimal" if t <= 35 else "cooling HVAC")
        print(f"      T={t:5.0f} degC  ->  eta_temp={e:.2f}  (PHVAC={phvac:.0f}%)  [{label}]")

    bess = build_bess_params_timeseries(temp_df)

    # 3. Save
    out_path = OUT_DIR / f"ercot_bess_params_{TARGET_YEAR}.csv"
    bess.to_csv(out_path)
    print(f"\n[3/4] Saved -> {out_path}")

    # 4. Summary
    print(f"\n[4/4] Summary statistics")
    print("-" * 50)
    stats = bess_params_summary(bess)

    print(f"  Hours simulated   : {stats['hours']:,}")
    print(f"  Mean eta_temp     : {stats['mean_eta_temp']:.5f}")
    print(f"  Min  eta_temp     : {stats['min_eta_temp']:.5f}  "
          f"(at T={bess['temp_c'].loc[bess['eta_temp'].idxmin()]:.1f} degC)")
    print(f"  Base RT eff.      : {stats['base_eta_rt']:.4f}  ({stats['base_eta_rt']*100:.1f}%)")
    print(f"  Mean RT eff.      : {stats['mean_eta_rt']:.4f}  ({stats['mean_eta_rt']*100:.1f}%)")
    print(f"  Min  RT eff.      : {stats['min_eta_rt']:.4f}  ({stats['min_eta_rt']*100:.1f}%)")
    print(f"  Mean PHVAC loss   : {stats['mean_phvac_loss_pct']:.3f}%  per cycle")

    total_hrs = stats["hours"]
    print(f"\n  Thermal regime distribution:")
    cold_pct    = stats["hours_hvac_cold"]  / total_hrs * 100
    optimal_pct = stats["hours_optimal"]    / total_hrs * 100
    hot_pct     = stats["hours_hvac_hot"]   / total_hrs * 100

    for label, hrs, pct in [
        ("Cold  (T < 15C, heating HVAC)",  stats["hours_hvac_cold"],    cold_pct),
        ("Optimal (15-35C, HVAC off)",     stats["hours_optimal"],       optimal_pct),
        ("Hot   (T > 35C, cooling HVAC)",  stats["hours_hvac_hot"],      hot_pct),
    ]:
        bar = "#" * int(pct / 2)
        print(f"    {label:<38} {hrs:5d} hrs  ({pct:5.1f}%)  {bar}")

    # Monthly averages
    print(f"\n  Monthly mean eta_temp:")
    monthly = bess["eta_temp"].resample("ME").mean()
    for ts, v in monthly.items():
        diff = (v - 1.0) * 100
        bar = "#" * int((1 - v) * 2000)  # scale small differences
        sign = "-" if diff < 0 else "+"
        print(f"    {ts.strftime('%b'):3s}  {v:.5f}  ({sign}{abs(diff):.3f}% from ideal)  {bar or '(optimal)'}")

    # 5. Plot
    print("\n  Generating plots ...")
    plot_bess(bess, TARGET_YEAR)

    print("\nDone.")
    return bess


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_bess(bess: pd.DataFrame, year: int) -> Path:
    fig = plt.figure(figsize=(15, 11))
    fig.suptitle(
        f"ERCOT BESS Thermal Efficiency Model  |  {LOCATION_NAME}  |  {year}\n"
        f"eta_ch={ETA_CH}, eta_dis={ETA_DIS}, RT(base)={ETA_CH*ETA_DIS:.4f}  |  "
        f"DoD={int((1-SOC_MIN)*100)}%  |  {DURATION_H:.0f}-hour battery",
        fontsize=11, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.35)

    ax_temp   = fig.add_subplot(gs[0, :])    # full-year temp
    ax_eta    = fig.add_subplot(gs[1, :])    # full-year eta_temp
    ax_curve  = fig.add_subplot(gs[2, 0])   # eta_temp(T) model curve
    ax_rt     = fig.add_subplot(gs[2, 1])   # monthly RT efficiency

    MFMT = mdates.DateFormatter("%b")
    MLOC = mdates.MonthLocator()

    C_COLD    = "#61afef"
    C_OPTIMAL = "#98c379"
    C_HOT     = "#e06c75"
    C_ETA     = "#c678dd"
    C_RT      = "#e5c07b"

    # ---- Panel 1: Temperature with regime shading ----------------------
    ax_temp.plot(bess.index, bess["temp_c"], color="#abb2bf", lw=0.5, alpha=0.8)
    ax_temp.fill_between(bess.index, bess["temp_c"],
                         where=bess["temp_c"] < 15,
                         color=C_COLD, alpha=0.25, label="Cold (<15 C)")
    ax_temp.fill_between(bess.index, bess["temp_c"],
                         where=(bess["temp_c"] >= 15) & (bess["temp_c"] <= 35),
                         color=C_OPTIMAL, alpha=0.20, label="Optimal (15-35 C)")
    ax_temp.fill_between(bess.index, bess["temp_c"],
                         where=bess["temp_c"] > 35,
                         color=C_HOT, alpha=0.35, label="Hot (>35 C)")
    ax_temp.axhline(35, ls="--", color=C_HOT,     lw=0.8, alpha=0.7)
    ax_temp.axhline(15, ls="--", color=C_COLD,    lw=0.8, alpha=0.7)
    ax_temp.set_ylabel("Temp (degC)")
    ax_temp.set_title("Ambient Temperature — BESS Thermal Regimes")
    ax_temp.legend(fontsize=8, loc="upper right", ncol=3)
    ax_temp.xaxis.set_major_formatter(MFMT); ax_temp.xaxis.set_major_locator(MLOC)
    ax_temp.grid(True, alpha=0.25)

    # ---- Panel 2: eta_temp timeseries ----------------------------------
    ax_eta.plot(bess.index, bess["eta_temp"], color=C_ETA, lw=0.5)
    ax_eta.axhline(1.00, ls="--", color="#abb2bf", lw=0.8, label="Ideal (no PHVAC)")
    ax_eta.fill_between(bess.index, bess["eta_temp"], 1.0,
                        where=bess["eta_temp"] < 1.0,
                        color=C_ETA, alpha=0.25, label="PHVAC loss")
    ax_eta.set_ylim(0.90, 1.02)
    ax_eta.set_ylabel("eta_temp (--)")
    ax_eta.set_title("Temperature Efficiency Modifier  |  "
                     "Charge: P_ch x (eta_ch x eta_temp)  |  "
                     "Discharge: P_dis / (eta_temp x eta_dis)")
    ax_eta.legend(fontsize=8)
    ax_eta.xaxis.set_major_formatter(MFMT); ax_eta.xaxis.set_major_locator(MLOC)
    ax_eta.grid(True, alpha=0.25)

    # ---- Panel 3: eta_temp(T) model curve ------------------------------
    t_range  = np.linspace(-15, 55, 500)
    eta_vals = eta_temp_from_tamb(t_range)
    rt_vals  = ETA_CH * eta_vals * ETA_DIS * eta_vals   # full RT with temp

    ax_curve.plot(t_range, eta_vals,  color=C_ETA, lw=2,   label="eta_temp (PHVAC model)")
    ax_curve.plot(t_range, rt_vals,   color=C_RT,  lw=1.5, ls="--",
                  label=f"RT eff. = eta_ch x eta_temp^2 x eta_dis")
    ax_curve.axhline(ETA_CH * ETA_DIS, color="#abb2bf", lw=0.8, ls=":",
                     label=f"RT base = {ETA_CH*ETA_DIS:.4f}")

    # Mark knot points
    ax_curve.scatter(_T_KNOTS, _ETA_KNOTS, color="#d19a66", zorder=5, s=50,
                     label="Model knots")

    # Shade regimes
    ax_curve.axvspan(-15, 15, alpha=0.07, color=C_COLD)
    ax_curve.axvspan(15,  35, alpha=0.07, color=C_OPTIMAL)
    ax_curve.axvspan(35,  55, alpha=0.07, color=C_HOT)
    for x in [15, 35]:
        ax_curve.axvline(x, ls=":", color="#abb2bf", lw=0.8)

    ax_curve.set_xlabel("Ambient Temperature (degC)")
    ax_curve.set_ylabel("Efficiency (--)")
    ax_curve.set_title("PHVAC Model: eta_temp and RT Efficiency vs Temperature")
    ax_curve.legend(fontsize=7); ax_curve.grid(True, alpha=0.25)
    ax_curve.set_ylim(0.84, 1.02)

    # ---- Panel 4: Monthly mean RT efficiency ---------------------------
    monthly_rt = bess["eta_rt_eff"].resample("ME").mean()
    months = [t.strftime("%b") for t in monthly_rt.index]
    base_rt = ETA_CH * ETA_DIS
    colours = [C_COLD if v < base_rt - 0.001 else C_OPTIMAL for v in monthly_rt.values]
    bars = ax_rt.bar(months, monthly_rt.values * 100, color=colours, alpha=0.75, edgecolor="white")
    ax_rt.axhline(base_rt * 100, ls="--", color="#abb2bf", lw=0.9,
                  label=f"Base RT = {base_rt*100:.1f}%")
    for bar, val in zip(bars, monthly_rt.values):
        ax_rt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                   f"{val*100:.2f}%", ha="center", va="bottom", fontsize=7)
    ax_rt.set_ylabel("Round-trip Efficiency (%)")
    ax_rt.set_title("Monthly Mean Round-Trip Efficiency (eta_ch x eta_temp^2 x eta_dis)")
    ax_rt.legend(fontsize=8)
    ax_rt.grid(True, axis="y", alpha=0.3)
    ax_rt.set_ylim(89, 92.5)

    plot_path = OUT_DIR / f"ercot_bess_params_{year}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved -> {plot_path}")
    plt.show()
    return plot_path


if __name__ == "__main__":
    main()
