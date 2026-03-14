"""
System LCOE optimizer for the ERCOT BTM microgrid.

Objective
---------
Find the (S, B) combination that minimises total system LCOE, subject to the
reliability constraint G = G_min(S, B) from the reliability surface.

sLCOE formula (methodology document)
--------------------------------------
    sLCOE = (C_solar + C_bess + C_gas) / Annual_demand_MWh

Where each component's annual cost = annualized_capex + fixed_opex + variable_opex.
All costs are UNSUBSIDIZED (pre-incentive) consistent with Lazard 18.0 baseline.

Method
------
1. Load the reliability surface  →  G_min(S, B)  from step 06
2. For each (S, B) in the grid:
   a. Look up G_min from the reliability surface
   b. Run dispatch(S, B, G_min) to get the actual energy mix
   c. Compute annual costs via lcoe_model
   d. Compute sLCOE
3. Find the minimum sLCOE point and characterise the optimal configuration
4. Sensitivity analysis (discount rate, gas price, capex +-20%)

Outputs
-------
    data/processed/ercot_slcoe_surface_2024.csv    sLCOE surface (long-form)
    data/processed/ercot_slcoe_surface_2024.png    heatmap + breakdown plots
    data/processed/ercot_slcoe_sensitivity_2024.png tornado chart
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.dispatcher import dispatch, load_timeseries
from models.lcoe_model  import (
    system_slcoe, print_financial_assumptions,
    annual_solar_cost, annual_bess_cost, annual_gas_cost,
    DISCOUNT_RATE, PROJECT_LIFE_YR, ITC_RATE,
    SOLAR_CAPEX_PER_KW, BESS_CAPEX_PER_KWH, GAS_CAPEX_PER_KW,
    GAS_OPEX_VAR_PER_MWH,
    crf,
)
from models.gas_model import HEAT_RATE_MMBTU_PER_MWH, GAS_PRICE_PER_MMBTU

OUT_DIR = PROJECT_ROOT / "data" / "processed"
YEAR    = 2024


# ---------------------------------------------------------------------------
# Grid (must match reliability solver grid exactly)
# ---------------------------------------------------------------------------
S_GRID = np.arange(0,   301, 25)   # MW DC
B_GRID = np.arange(0,  1001, 100)  # MWh


# ---------------------------------------------------------------------------
# Core optimisation loop
# ---------------------------------------------------------------------------

def run_slcoe_surface(
    ts:      pd.DataFrame,
    surface: pd.DataFrame,
) -> pd.DataFrame:
    """
    For every (S, B) grid point, run dispatch with G = G_min and compute sLCOE.

    Returns long-form DataFrame with columns:
        S_mw, B_mwh, G_min_mw, gas_hours_yr,
        solar_used_mwh, bess_discharge_mwh, gas_gen_mwh,
        solar_cf_pct, bess_cycles_per_year, gas_cf_pct,
        solar_cost_usd_yr, bess_cost_usd_yr, gas_cost_usd_yr,
        total_cost_usd_yr, demand_mwh_yr, slcoe_per_mwh,
        solar_lcoe_contrib, bess_lcoe_contrib, gas_lcoe_contrib
    """
    # Build lookup dict for G_min
    gmin_map = {
        (row.S_mw, row.B_mwh): row.G_min_mw
        for _, row in surface.iterrows()
    }

    rows   = []
    n_total = len(S_GRID) * len(B_GRID)
    t0     = time.time()

    for i, S in enumerate(S_GRID):
        for j, B in enumerate(B_GRID):
            G_min = gmin_map.get((float(S), float(B)))
            if G_min is None:
                continue

            results, summary = dispatch(float(S), float(B), float(G_min), ts)
            costs = system_slcoe(float(S), float(B), float(G_min), summary)

            row = {
                "S_mw":             S,
                "B_mwh":            B,
                "G_min_mw":         round(G_min, 2),
                # Energy
                "solar_used_mwh":      round(summary["solar_used_mwh"], 0),
                "bess_discharge_mwh":  round(summary["bess_discharge_mwh"], 0),
                "gas_gen_mwh":         round(summary["gas_gen_mwh"], 0),
                "curtailed_mwh":       round(summary["curtailed_mwh"], 0),
                "solar_share_pct":     round(summary["solar_share_pct"], 2),
                "bess_share_pct":      round(summary["bess_share_pct"], 2),
                "gas_share_pct":       round(summary["gas_share_pct"], 2),
                "renewable_share_pct": round(summary["renewable_share_pct"], 2),
                "gas_hours_yr":        round((results["gas_gen_mw"] > 0.1).sum(), 0),
                "bess_cycles_per_year":round(summary["bess_cycles_per_year"], 1),
                "annual_co2_t":        round(summary["annual_co2_t"], 0),
                # Costs
                **costs,
            }
            rows.append(row)

        # Progress every row of S
        elapsed = time.time() - t0
        done = (i + 1) * len(B_GRID)
        print(f"  S={S:>3.0f} MW done  [{done:>3}/{n_total}]  elapsed {elapsed:.1f}s")

    df = pd.DataFrame(rows)
    print(f"\n  sLCOE surface complete in {time.time()-t0:.1f}s")
    return df


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def sensitivity_analysis(opt_S: float, opt_B: float, opt_G: float,
                          ts: pd.DataFrame) -> pd.DataFrame:
    """
    Tornado-chart sensitivity: vary one parameter at a time +-20% from base.
    Returns DataFrame with columns: parameter, low_slcoe, base_slcoe, high_slcoe.
    """
    import importlib
    import models.lcoe_model as lm
    import models.gas_model  as gm

    _, base_summary = dispatch(opt_S, opt_B, opt_G, ts)
    base_costs      = system_slcoe(opt_S, opt_B, opt_G, base_summary)
    base_slcoe      = base_costs["slcoe_per_mwh"]

    scenarios = [
        ("Solar capex ($/kW)",        "SOLAR_CAPEX_PER_KW",      lm, None),
        ("BESS capex ($/kWh)",         "BESS_CAPEX_PER_KWH",      lm, None),
        ("Gas capex ($/kW)",           "GAS_CAPEX_PER_KW",        lm, None),
        ("Gas price ($/MMBtu)",        "GAS_PRICE_PER_MMBTU",     gm, None),
        ("Discount rate",              "DISCOUNT_RATE",            lm, None),
        ("Gas var O&M ($/MWh)",        "GAS_OPEX_VAR_PER_MWH",    lm, None),
    ]

    rows = []
    for label, attr, module, _ in scenarios:
        base_val = getattr(module, attr)
        results  = {}
        for factor, key in [(0.80, "low"), (1.00, "base"), (1.20, "high")]:
            setattr(module, attr, base_val * factor)
            _, s = dispatch(opt_S, opt_B, opt_G, ts)
            c = system_slcoe(opt_S, opt_B, opt_G, s)
            results[key] = c["slcoe_per_mwh"]
        setattr(module, attr, base_val)   # restore

        rows.append({
            "parameter":  label,
            "base_val":   base_val,
            "low_slcoe":  results["low"],
            "base_slcoe": results["base"],
            "high_slcoe": results["high"],
            "swing":      abs(results["high"] - results["low"]),
        })

    return pd.DataFrame(rows).sort_values("swing", ascending=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("ERCOT Microgrid  —  System LCOE Optimiser")
    print("=" * 70)

    # 1. Load reliability surface
    surface_path = OUT_DIR / f"ercot_reliability_surface_{YEAR}.csv"
    if not surface_path.exists():
        raise FileNotFoundError(
            f"{surface_path.name} not found. Run 06_solve_reliability.py first."
        )
    surface = pd.read_csv(surface_path)
    print(f"\n[1/5] Reliability surface loaded: {len(surface)} grid points")
    print(f"  G_min range: {surface['G_min_mw'].min():.1f} — {surface['G_min_mw'].max():.1f} MW")

    # 2. Financial assumptions
    print(f"\n[2/5] Financial assumptions")
    print("-" * 55)
    print_financial_assumptions()

    # 3. sLCOE grid search
    print(f"\n[3/5] Running sLCOE grid search ...")
    ts = load_timeseries(OUT_DIR, YEAR)
    slcoe_df = run_slcoe_surface(ts, surface)

    # Save
    out_csv = OUT_DIR / f"ercot_slcoe_surface_{YEAR}.csv"
    slcoe_df.to_csv(out_csv, index=False)
    print(f"\n  Saved -> {out_csv.name}")

    # 4. Analysis
    print(f"\n[4/5] sLCOE surface analysis")
    print("-" * 70)

    demand_mwh = slcoe_df["demand_mwh_yr"].iloc[0]
    print(f"\n  Annual demand: {demand_mwh:,.0f} MWh/yr")

    # Baseline: gas-only
    base_row = slcoe_df.loc[(slcoe_df.S_mw == 0) & (slcoe_df.B_mwh == 0)].iloc[0]
    print(f"\n  Baseline (S=0, B=0, G={base_row['G_min_mw']:.1f} MW):")
    print(f"    sLCOE            = ${base_row['slcoe_per_mwh']:.2f}/MWh")
    print(f"    Annual cost      = ${base_row['total_cost_usd_yr']/1e6:.2f}M/yr")
    print(f"      Solar          = ${base_row['solar_cost_usd_yr']/1e6:.2f}M")
    print(f"      BESS           = ${base_row['bess_cost_usd_yr']/1e6:.2f}M")
    print(f"      Gas            = ${base_row['gas_cost_usd_yr']/1e6:.2f}M")
    print(f"    Gas share        = {base_row['gas_share_pct']:.1f}%  |  "
          f"Renewable = {base_row['renewable_share_pct']:.1f}%")

    # Optimum
    idx_opt = slcoe_df["slcoe_per_mwh"].idxmin()
    opt     = slcoe_df.loc[idx_opt]
    savings_pct = (base_row["slcoe_per_mwh"] - opt["slcoe_per_mwh"]) / base_row["slcoe_per_mwh"] * 100
    print(f"\n  OPTIMUM (min sLCOE):")
    print(f"    S = {opt['S_mw']:.0f} MW DC  |  B = {opt['B_mwh']:.0f} MWh  |  "
          f"G = {opt['G_min_mw']:.1f} MW")
    print(f"    sLCOE            = ${opt['slcoe_per_mwh']:.2f}/MWh  "
          f"({savings_pct:.1f}% below gas-only)")
    print(f"    Annual cost      = ${opt['total_cost_usd_yr']/1e6:.2f}M/yr")
    print(f"      Solar          = ${opt['solar_cost_usd_yr']/1e6:.2f}M  "
          f"({opt['solar_lcoe_contrib']:.2f}/MWh)")
    print(f"      BESS           = ${opt['bess_cost_usd_yr']/1e6:.2f}M  "
          f"({opt['bess_lcoe_contrib']:.2f}/MWh)")
    print(f"      Gas            = ${opt['gas_cost_usd_yr']/1e6:.2f}M  "
          f"({opt['gas_lcoe_contrib']:.2f}/MWh)")
    print(f"    Energy mix:")
    print(f"      Solar          = {opt['solar_share_pct']:.1f}%  "
          f"({opt['solar_used_mwh']/1e3:.1f} GWh/yr)")
    print(f"      BESS           = {opt['bess_share_pct']:.1f}%  "
          f"({opt['bess_discharge_mwh']/1e3:.1f} GWh/yr, "
          f"{opt['bess_cycles_per_year']:.0f} cycles/yr)")
    print(f"      Gas            = {opt['gas_share_pct']:.1f}%  "
          f"({opt['gas_gen_mwh']/1e3:.1f} GWh/yr, "
          f"{opt['gas_hours_yr']:.0f} hrs/yr)")
    print(f"    Renewable share  = {opt['renewable_share_pct']:.1f}%")
    print(f"    Annual CO2       = {opt['annual_co2_t']/1e3:.1f} kt  "
          f"(vs {base_row['annual_co2_t']/1e3:.1f} kt baseline)")

    # ITC impact on optimum
    itc_eligible_capex = (SOLAR_CAPEX_PER_KW * opt["S_mw"] * 1000 +
                          BESS_CAPEX_PER_KWH  * opt["B_mwh"] * 1000)
    itc_saving_yr = itc_eligible_capex * ITC_RATE * crf(PROJECT_LIFE_YR, DISCOUNT_RATE)
    slcoe_with_itc = opt["slcoe_per_mwh"] - itc_saving_yr / demand_mwh
    print(f"\n  ITC/IRA 2022 impact on optimum (30% on solar + BESS capex):")
    print(f"    ITC eligible capex: ${itc_eligible_capex/1e6:.1f}M")
    print(f"    Annual ITC saving : ${itc_saving_yr/1e6:.2f}M/yr")
    print(f"    sLCOE with ITC    = ${slcoe_with_itc:.2f}/MWh")

    # sLCOE table: S rows, B columns (pivot)
    print(f"\n  sLCOE surface ($/MWh) — rows=B [MWh], cols=S [MW]:")
    pivot = slcoe_df.pivot(index="B_mwh", columns="S_mw", values="slcoe_per_mwh")
    print(pivot.to_string(float_format=lambda x: f"{x:.2f}"))

    # Top 10 configurations
    print(f"\n  Top 10 lowest-sLCOE configurations:")
    top10 = slcoe_df.nsmallest(10, "slcoe_per_mwh")[
        ["S_mw","B_mwh","G_min_mw","slcoe_per_mwh",
         "solar_share_pct","bess_share_pct","gas_share_pct",
         "renewable_share_pct","gas_hours_yr","annual_co2_t"]
    ]
    print(top10.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # 5. Sensitivity analysis
    print(f"\n[5/5] Sensitivity analysis ...")
    sens = sensitivity_analysis(
        float(opt["S_mw"]), float(opt["B_mwh"]), float(opt["G_min_mw"]), ts
    )
    print(f"\n  Tornado chart (+-20% on each parameter):")
    print(f"  {'Parameter':<30}  {'Base':>8}  {'Low (-20%)':>12}  {'High (+20%)':>12}  {'Swing':>8}")
    print("  " + "-" * 76)
    for _, r in sens.iterrows():
        print(f"  {r['parameter']:<30}  {r['base_slcoe']:>8.2f}  "
              f"{r['low_slcoe']:>12.2f}  {r['high_slcoe']:>12.2f}  "
              f"{r['swing']:>8.3f}")

    # 6. Plots
    print(f"\n  Generating plots ...")
    plot_slcoe(slcoe_df, pivot, opt, base_row, ts, YEAR)
    plot_sensitivity(sens, opt, YEAR)

    print("\nDone.")
    return slcoe_df, opt


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_slcoe(slcoe_df, pivot, opt, base_row, ts, year):
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"ERCOT Microgrid  —  System LCOE Optimisation  |  Dallas-Fort Worth, TX  |  {year}\n"
        f"Unsubsidized sLCOE ($/MWh)  |  BTM Islanded Microgrid  |  50 MW IT Load\n"
        f"Sources: NREL ATB 2025 (capex)  |  Lazard LCOE+ 18.0 (benchmarks)  |  EIA 2026 STEO (fuel)",
        fontsize=10, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    ax_heat   = fig.add_subplot(gs[0, :2])   # sLCOE heatmap (wide)
    ax_pie    = fig.add_subplot(gs[0,  2])   # cost breakdown at optimum
    ax_vs_s   = fig.add_subplot(gs[1,  0])   # sLCOE vs S (for key B)
    ax_vs_b   = fig.add_subplot(gs[1,  1])   # sLCOE vs B (for key S)
    ax_stack  = fig.add_subplot(gs[1,  2])   # cost stacked bar for key configs

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "slcoe", ["#98c379", "#e5c07b", "#e06c75"], N=256
    )

    # ---- Panel 1: sLCOE heatmap ------------------------------------------
    vmin = slcoe_df["slcoe_per_mwh"].min()
    vmax = slcoe_df["slcoe_per_mwh"].max()

    im = ax_heat.imshow(
        pivot.values,
        aspect="auto", origin="lower", cmap=cmap,
        vmin=vmin, vmax=vmax,
        extent=[
            S_GRID.min() - 12.5, S_GRID.max() + 12.5,
            B_GRID.min() - 50,   B_GRID.max() + 50,
        ],
    )
    cbar = plt.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.01)
    cbar.set_label("sLCOE ($/MWh)", fontsize=9)

    # Contour lines
    Z  = pivot.values
    cs = ax_heat.contour(S_GRID, B_GRID, Z,
                         levels=np.arange(np.floor(vmin), np.ceil(vmax) + 1, 1.0),
                         colors="white", linewidths=0.8, alpha=0.7)
    ax_heat.clabel(cs, fmt="$%.0f", fontsize=8, inline=True)

    # Mark optimum
    ax_heat.plot(opt["S_mw"], opt["B_mwh"], "w*", ms=18, zorder=5,
                 label=f"Optimum: S={opt['S_mw']:.0f} MW, B={opt['B_mwh']:.0f} MWh")
    ax_heat.legend(fontsize=9, loc="upper right")
    ax_heat.set_xlabel("Solar Capacity S (MW DC)")
    ax_heat.set_ylabel("Battery Capacity B (MWh)")
    ax_heat.set_title("System LCOE Surface  ($/MWh, unsubsidized)\nWhite star = optimum")

    # ---- Panel 2: Cost breakdown pie at optimum --------------------------
    labels_pie = ["Solar", "BESS", "Gas\n(capex+O&M)", "Gas\n(fuel)"]
    # Split gas into capex/opex vs fuel
    G_mw    = opt["G_min_mw"]
    gas_gen = opt["gas_gen_mwh"]
    gas_fuel_cost  = (GAS_OPEX_VAR_PER_MWH + HEAT_RATE_MMBTU_PER_MWH * GAS_PRICE_PER_MMBTU) * gas_gen
    gas_capex_cost = opt["gas_cost_usd_yr"] - gas_fuel_cost
    sizes = [
        opt["solar_cost_usd_yr"],
        opt["bess_cost_usd_yr"],
        gas_capex_cost,
        gas_fuel_cost,
    ]
    # Guard for zero-size slices
    colors_pie = ["#e5c07b", "#61afef", "#c678dd", "#e06c75"]
    non_zero   = [(s, l, c) for s, l, c in zip(sizes, labels_pie, colors_pie) if s > 0]
    if non_zero:
        s_nz, l_nz, c_nz = zip(*non_zero)
        wedges, texts, autotexts = ax_pie.pie(
            s_nz, labels=l_nz, colors=c_nz,
            autopct="%1.1f%%", startangle=90,
            textprops={"fontsize": 8},
        )
    ax_pie.set_title(
        f"Annual Cost Breakdown\nOptimum: S={opt['S_mw']:.0f} MW, "
        f"B={opt['B_mwh']:.0f} MWh\n"
        f"Total = ${opt['total_cost_usd_yr']/1e6:.1f}M/yr  |  "
        f"sLCOE = ${opt['slcoe_per_mwh']:.2f}/MWh",
        fontsize=8,
    )

    # ---- Panel 3: sLCOE vs S (for key B values) --------------------------
    colours_B = ["#abb2bf", "#61afef", "#98c379", "#e5c07b", "#c678dd"]
    B_show    = [0, 200, 400, 600, 800]
    for B_val, col in zip(B_show, colours_B):
        sub = slcoe_df[slcoe_df.B_mwh == B_val].sort_values("S_mw")
        ax_vs_s.plot(sub["S_mw"], sub["slcoe_per_mwh"], "o-", color=col,
                     lw=1.5, ms=4, label=f"B={B_val} MWh")
    ax_vs_s.axhline(base_row["slcoe_per_mwh"], ls="--", color="#e06c75", lw=0.9,
                    label=f"Gas-only (${base_row['slcoe_per_mwh']:.2f})")
    ax_vs_s.axvline(opt["S_mw"], ls=":", color="white", lw=0.8, alpha=0.6)
    ax_vs_s.set_xlabel("Solar Capacity S (MW DC)")
    ax_vs_s.set_ylabel("sLCOE ($/MWh)")
    ax_vs_s.set_title("sLCOE vs Solar — by Battery Size")
    ax_vs_s.legend(fontsize=7); ax_vs_s.grid(True, alpha=0.25)

    # ---- Panel 4: sLCOE vs B (for key S values) --------------------------
    colours_S = ["#abb2bf", "#e5c07b", "#98c379", "#61afef", "#c678dd"]
    S_show    = [0, 75, 150, 200, 250]
    for S_val, col in zip(S_show, colours_S):
        sub = slcoe_df[slcoe_df.S_mw == S_val].sort_values("B_mwh")
        ax_vs_b.plot(sub["B_mwh"], sub["slcoe_per_mwh"], "o-", color=col,
                     lw=1.5, ms=4, label=f"S={S_val} MW")
    ax_vs_b.axhline(base_row["slcoe_per_mwh"], ls="--", color="#e06c75", lw=0.9,
                    label=f"Gas-only (${base_row['slcoe_per_mwh']:.2f})")
    ax_vs_b.axvline(opt["B_mwh"], ls=":", color="white", lw=0.8, alpha=0.6)
    ax_vs_b.set_xlabel("Battery Capacity B (MWh)")
    ax_vs_b.set_ylabel("sLCOE ($/MWh)")
    ax_vs_b.set_title("sLCOE vs Battery — by Solar Capacity")
    ax_vs_b.legend(fontsize=7); ax_vs_b.grid(True, alpha=0.25)

    # ---- Panel 5: Stacked cost bar for key configs -----------------------
    configs_bar = [
        (0,   0,   "Gas-only"),
        (75,  0,   "S=75, B=0"),
        (150, 0,   "S=150, B=0"),
        (int(opt["S_mw"]), int(opt["B_mwh"]), "Optimum"),
        (250, 400, "S=250, B=400"),
        (300, 1000,"S=300, B=1000"),
    ]
    bar_labels, solar_c, bess_c, gas_c = [], [], [], []
    for Sv, Bv, lbl in configs_bar:
        # Match to nearest grid point
        match = slcoe_df.loc[
            (slcoe_df.S_mw == Sv) & (slcoe_df.B_mwh == Bv)
        ]
        if match.empty:
            continue
        r = match.iloc[0]
        bar_labels.append(lbl)
        solar_c.append(r["solar_lcoe_contrib"])
        bess_c.append(r["bess_lcoe_contrib"])
        gas_c.append(r["gas_lcoe_contrib"])

    x = np.arange(len(bar_labels))
    w = 0.6
    b1 = ax_stack.bar(x, solar_c, w, label="Solar",     color="#e5c07b", alpha=0.85)
    b2 = ax_stack.bar(x, bess_c,  w, label="BESS",      color="#61afef", alpha=0.85,
                      bottom=solar_c)
    b3 = ax_stack.bar(x, gas_c,   w, label="Gas",       color="#e06c75", alpha=0.85,
                      bottom=[s+b for s,b in zip(solar_c, bess_c)])

    # Total labels
    totals = [s+b+g for s,b,g in zip(solar_c, bess_c, gas_c)]
    for xi, tot in zip(x, totals):
        ax_stack.text(xi, tot + 0.15, f"${tot:.1f}", ha="center", fontsize=7)

    ax_stack.set_xticks(x); ax_stack.set_xticklabels(bar_labels, rotation=20, ha="right", fontsize=7)
    ax_stack.set_ylabel("sLCOE contribution ($/MWh)")
    ax_stack.set_title("sLCOE Cost Stack by Configuration")
    ax_stack.legend(fontsize=7); ax_stack.grid(True, axis="y", alpha=0.25)

    plt.savefig(OUT_DIR / f"ercot_slcoe_surface_{year}.png", dpi=150, bbox_inches="tight")
    print(f"  Plot saved -> ercot_slcoe_surface_{year}.png")
    plt.show()


def plot_sensitivity(sens: pd.DataFrame, opt, year: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        f"Sensitivity Analysis — sLCOE at Optimum  |  "
        f"S={opt['S_mw']:.0f} MW, B={opt['B_mwh']:.0f} MWh, "
        f"G={opt['G_min_mw']:.1f} MW\n"
        f"Each parameter varied +-20% independently",
        fontsize=10, fontweight="bold",
    )

    base_slcoe = sens["base_slcoe"].iloc[0]
    y_pos      = np.arange(len(sens))

    for i, (_, row) in enumerate(sens.iterrows()):
        lo = min(row["low_slcoe"],  row["high_slcoe"]) - base_slcoe
        hi = max(row["low_slcoe"],  row["high_slcoe"]) - base_slcoe
        color_lo = "#98c379" if lo < 0 else "#e06c75"
        color_hi = "#e06c75" if hi > 0 else "#98c379"
        ax.barh(i, lo, left=0, color=color_lo, alpha=0.8, height=0.6)
        ax.barh(i, hi, left=0, color=color_hi, alpha=0.8, height=0.6)
        ax.text(hi + 0.02, i, f"${row['high_slcoe']:.2f}", va="center", fontsize=8)
        ax.text(lo - 0.02, i, f"${row['low_slcoe']:.2f}", va="center",
                ha="right", fontsize=8)

    ax.set_yticks(y_pos); ax.set_yticklabels(sens["parameter"], fontsize=9)
    ax.axvline(0, color="white", lw=1.0)
    ax.set_xlabel("Change in sLCOE vs base ($/MWh)")
    ax.set_title(f"Base sLCOE = ${base_slcoe:.2f}/MWh")
    ax.grid(True, axis="x", alpha=0.25)

    plt.tight_layout()
    plt.savefig(OUT_DIR / f"ercot_slcoe_sensitivity_{year}.png", dpi=150, bbox_inches="tight")
    print(f"  Plot saved -> ercot_slcoe_sensitivity_{year}.png")
    plt.show()


if __name__ == "__main__":
    main()
