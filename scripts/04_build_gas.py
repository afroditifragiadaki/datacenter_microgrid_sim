"""
Gas model characterization and sizing analysis for ERCOT 2024.

Since G (installed gas capacity) is an optimizer decision variable, we cannot
pre-compute a gas dispatch timeseries without knowing what solar and BESS do.
Instead this script:

    1. Documents the gas model parameters
    2. Runs a G-sensitivity analysis (worst-case: no solar, no BESS) to show
       how gas coverage scales with installed capacity
    3. Profiles the load duration curve — the key input for gas sizing
    4. Computes illustrative annual fuel costs at candidate G values
    5. Exports a model-parameters JSON for the dispatcher to consume

Outputs
-------
    data/processed/ercot_gas_sizing_2024.csv   load duration + sizing profile
    data/processed/ercot_gas_params.json       model constants for dispatcher
    data/processed/ercot_gas_sizing_2024.png   diagnostic plots
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.gas_model import (
    build_gas_sizing_profile,
    simulate_gas,
    HEAT_RATE_MMBTU_PER_MWH,
    GAS_PRICE_PER_MMBTU,
    CO2_FACTOR_T_PER_MMBTU,
    CO2_FACTOR_T_PER_MWH,
    MIN_LOAD_FRACTION,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TARGET_YEAR   = 2024
LOCATION_NAME = "Dallas-Fort Worth, TX (ERCOT)"
OUT_DIR = PROJECT_ROOT / "data" / "processed"

# G candidates for sensitivity analysis [MW]
G_CANDIDATES = [50, 60, 68, 75, 100]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("ERCOT Gas Model — Parameters & Sizing Analysis")
    print("=" * 60)

    # 1. Load demand timeseries
    demand_path = OUT_DIR / f"ercot_demand_{TARGET_YEAR}.csv"
    if not demand_path.exists():
        raise FileNotFoundError(f"{demand_path} not found. Run 01_build_demand.py first.")

    print(f"\n[1/4] Loading demand timeseries ...")
    demand = pd.read_csv(demand_path, index_col="datetime", parse_dates=True)
    load   = demand["total_load_mw"].to_numpy()
    print(f"  Hours        : {len(load):,}")
    print(f"  Load range   : {load.min():.1f} – {load.max():.1f} MW")
    print(f"  Annual demand: {load.sum():,.0f} MWh")

    # 2. Model parameters
    eff = 3.412 / HEAT_RATE_MMBTU_PER_MWH  # HHV thermal efficiency
    print(f"\n[2/4] Gas model parameters")
    print(f"  Technology           : Natural Gas Reciprocating Engine (RICE)")
    print(f"  Role                 : Gap-filler  [P_gas = min(G, unmet_demand)]")
    print(f"  Installed capacity G : optimizer decision variable [MW]")
    print(f"  Heat rate            : {HEAT_RATE_MMBTU_PER_MWH:.1f} MMBtu/MWh  (HHV, full load)")
    print(f"  Thermal efficiency   : {eff*100:.1f}%")
    print(f"  Gas price            : ${GAS_PRICE_PER_MMBTU:.2f}/MMBtu  (EIA 2024 Texas)")
    print(f"  Fuel cost per MWh    : ${HEAT_RATE_MMBTU_PER_MWH * GAS_PRICE_PER_MMBTU:.2f}/MWh  (variable cost only)")
    print(f"  CO2 factor           : {CO2_FACTOR_T_PER_MMBTU:.4f} tCO2/MMBtu")
    print(f"                       = {CO2_FACTOR_T_PER_MWH:.4f} tCO2/MWh at full load")
    print(f"  Min stable load      : {MIN_LOAD_FRACTION*100:.0f}% of G")

    # 3. Build sizing profile (worst-case: gas covers full load)
    print(f"\n[3/4] Sizing sensitivity analysis (no solar, no BESS — worst case)")
    print(f"  {'G (MW)':>8}  {'Ann. Gen (MWh)':>16}  {'Capacity Factor':>16}"
          f"  {'Ann. Fuel Cost ($M)':>20}  {'Ann. CO2 (kt)':>14}  {'Unserved (MWh)':>15}")
    print(f"  {'-'*8}  {'-'*16}  {'-'*16}  {'-'*20}  {'-'*14}  {'-'*15}")

    sensitivity_rows = []
    for G in G_CANDIDATES:
        res = simulate_gas(G, load)
        s   = res["summary"]
        row = {
            "G_mw":                 G,
            "annual_gen_mwh":       s["annual_gen_mwh"],
            "capacity_factor_pct":  s["capacity_factor_pct"],
            "annual_fuel_cost_usd": s["annual_fuel_cost_usd"],
            "annual_co2_t":         s["annual_co2_t"],
            "unserved_energy_mwh":  s["unserved_energy_mwh"],
            "hours_online":         s["hours_online"],
            "lcoe_fuel_usd_mwh":    s["lcoe_fuel_usd_mwh"],
        }
        sensitivity_rows.append(row)
        print(f"  {G:>8.0f}  {s['annual_gen_mwh']:>16,.0f}  {s['capacity_factor_pct']:>15.1f}%"
              f"  ${s['annual_fuel_cost_usd']/1e6:>18.2f}M"
              f"  {s['annual_co2_t']/1000:>13.1f}k"
              f"  {s['unserved_energy_mwh']:>15,.0f}")

    sensitivity_df = pd.DataFrame(sensitivity_rows)

    # 4. Build load duration curve profile
    sizing_profile = build_gas_sizing_profile(demand)
    load_sorted = np.sort(load)[::-1]

    # 5. Save outputs
    out_csv  = OUT_DIR / f"ercot_gas_sizing_{TARGET_YEAR}.csv"
    out_json = OUT_DIR / "ercot_gas_params.json"

    sizing_profile.to_csv(out_csv)

    gas_params = {
        "technology":             "Natural Gas Reciprocating Engine (RICE)",
        "heat_rate_mmbtu_per_mwh": HEAT_RATE_MMBTU_PER_MWH,
        "thermal_efficiency_pct":  round(eff * 100, 2),
        "gas_price_per_mmbtu":     GAS_PRICE_PER_MMBTU,
        "fuel_cost_per_mwh":       round(HEAT_RATE_MMBTU_PER_MWH * GAS_PRICE_PER_MMBTU, 2),
        "co2_factor_t_per_mmbtu":  CO2_FACTOR_T_PER_MMBTU,
        "co2_factor_t_per_mwh":    round(CO2_FACTOR_T_PER_MWH, 4),
        "min_load_fraction":       MIN_LOAD_FRACTION,
        "dispatch_rule":           "P_gas_t = min(G_mw, unmet_demand_t)",
        "fuel_rule":               "Fuel_t [MMBtu] = P_gas_t [MW] * heat_rate",
        "ercot_peak_load_mw":      float(round(load.max(), 2)),
        "ercot_min_load_mw":       float(round(load.min(), 2)),
        "ercot_mean_load_mw":      float(round(load.mean(), 2)),
        "ercot_annual_demand_mwh": float(round(load.sum(), 0)),
    }
    with open(out_json, "w") as f:
        json.dump(gas_params, f, indent=2)

    print(f"\n  Saved sizing profile -> {out_csv.name}")
    print(f"  Saved model params   -> {out_json.name}")

    # 6. Additional metrics
    print(f"\n  Load Duration Curve key percentiles:")
    for pct in [100, 99, 95, 90, 75, 50, 25]:
        val = np.percentile(load, pct)
        print(f"    P{pct:3d}: {val:.2f} MW")

    print(f"\n  Fuel cost sensitivity ($ per MWh generated, variable cost only):")
    for price in [2.00, 2.50, 3.00, 3.50, 4.00]:
        cost_mwh = HEAT_RATE_MMBTU_PER_MWH * price
        print(f"    @ ${price:.2f}/MMBtu  ->  ${cost_mwh:.2f}/MWh")

    # 7. Plot
    print(f"\n  Generating plots ...")
    plot_gas(load, load_sorted, sensitivity_df, TARGET_YEAR)

    print("\nDone.")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_gas(load, load_sorted, sensitivity_df, year):
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        f"ERCOT Gas Model — Sizing Analysis  |  {LOCATION_NAME}  |  {year}\n"
        f"Technology: RICE  |  Heat rate: {HEAT_RATE_MMBTU_PER_MWH} MMBtu/MWh  |  "
        f"Gas price: ${GAS_PRICE_PER_MMBTU}/MMBtu  |  "
        f"Fuel cost: ${HEAT_RATE_MMBTU_PER_MWH*GAS_PRICE_PER_MMBTU:.2f}/MWh",
        fontsize=10, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.33)

    ax_ldc   = fig.add_subplot(gs[0, :])   # load duration curve (wide)
    ax_gen   = fig.add_subplot(gs[1, 0])   # annual generation vs G
    ax_cost  = fig.add_subplot(gs[1, 1])   # annual fuel cost vs G

    C_LOAD  = "#98c379"
    C_GAS   = "#e06c75"
    C_COST  = "#e5c07b"
    C_GRID  = "#abb2bf"

    n_hours = len(load_sorted)
    x_hours = np.arange(1, n_hours + 1)
    x_pct   = x_hours / n_hours * 100

    # ---- Panel 1: Load Duration Curve with G capacity lines -------------
    ax_ldc.fill_between(x_pct, load_sorted, alpha=0.25, color=C_LOAD)
    ax_ldc.plot(x_pct, load_sorted, color=C_LOAD, lw=1.5, label="Facility load (sorted)")

    styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    for G, sty in zip(G_CANDIDATES, styles):
        covered  = np.minimum(load_sorted, G)
        ax_ldc.axhline(G, ls=sty, lw=1.2, color=C_GAS, alpha=0.8)
        ax_ldc.text(101, G, f"G={G} MW", va="center", fontsize=7, color=C_GAS)

    ax_ldc.set_xlabel("Duration (%  of hours)")
    ax_ldc.set_ylabel("Load / Gas Capacity (MW)")
    ax_ldc.set_title("Load Duration Curve  |  Gas capacity lines show coverage threshold")
    ax_ldc.set_xlim(0, 105)
    ax_ldc.legend(fontsize=8)
    ax_ldc.grid(True, alpha=0.25)

    # ---- Panel 2: Annual generation vs G --------------------------------
    Gs   = sensitivity_df["G_mw"].values
    gen  = sensitivity_df["annual_gen_mwh"].values / 1e3   # GWh

    ax_gen.bar(Gs, gen, width=6, color=C_LOAD, alpha=0.75, edgecolor="white")
    ax_gen.set_xlabel("Installed Capacity G (MW)")
    ax_gen.set_ylabel("Annual Generation (GWh)")
    ax_gen.set_title("Annual Gas Generation vs G\n(worst case: no solar, no BESS)")
    for x, y in zip(Gs, gen):
        ax_gen.text(x, y + 0.5, f"{y:.0f}", ha="center", va="bottom", fontsize=8)
    ax_gen.grid(True, axis="y", alpha=0.3)

    # ---- Panel 3: Annual fuel cost vs G ---------------------------------
    costs = sensitivity_df["annual_fuel_cost_usd"].values / 1e6   # $M
    co2s  = sensitivity_df["annual_co2_t"].values / 1e3           # kt

    ax_cost.bar(Gs, costs, width=6, color=C_COST, alpha=0.75, edgecolor="white",
                label="Fuel cost ($M)")
    ax_cost2 = ax_cost.twinx()
    ax_cost2.plot(Gs, co2s, "o--", color=C_GAS, lw=1.5, label="CO2 (kt)")
    ax_cost2.set_ylabel("CO2 Emissions (kt/year)", color=C_GAS)
    ax_cost2.tick_params(axis="y", labelcolor=C_GAS)

    ax_cost.set_xlabel("Installed Capacity G (MW)")
    ax_cost.set_ylabel("Annual Fuel Cost ($M)")
    ax_cost.set_title("Annual Fuel Cost & CO2  vs G\n(worst case: no solar, no BESS)")
    for x, y in zip(Gs, costs):
        ax_cost.text(x, y + 0.5, f"${y:.1f}M", ha="center", va="bottom", fontsize=8)
    ax_cost.grid(True, axis="y", alpha=0.3)

    lines1, labels1 = ax_cost.get_legend_handles_labels()
    lines2, labels2 = ax_cost2.get_legend_handles_labels()
    ax_cost.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    plot_path = OUT_DIR / f"ercot_gas_sizing_{year}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved -> {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
