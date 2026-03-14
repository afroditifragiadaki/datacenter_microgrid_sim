"""
Run the hourly dispatch simulation for ERCOT 2024 across several
(S, B, G) configurations to validate the dispatcher and explore the
solution space before optimization.

Usage
-----
    python scripts/05_run_dispatch.py

Outputs
-------
    data/processed/ercot_dispatch_<tag>_2024.csv    hourly results per config
    data/processed/ercot_dispatch_summary_2024.csv  KPI comparison table
    data/processed/ercot_dispatch_2024.png          multi-config comparison plot
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.dispatcher import dispatch, load_timeseries

OUT_DIR = PROJECT_ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Configurations to explore  (S_mw, B_mwh, G_mw, label)
# ---------------------------------------------------------------------------
CONFIGS = [
    # S_mw   B_mwh   G_mw   label
    (   0,      0,    68,  "Gas-only baseline"),
    ( 100,    200,    60,  "S=100 B=200 G=60"),
    ( 150,    400,    50,  "S=150 B=400 G=50"),
    ( 200,    600,    40,  "S=200 B=600 G=40"),
    ( 250,    800,    30,  "S=250 B=800 G=30"),
]

# Colours for each config in the comparison plots
CONFIG_COLOURS = ["#abb2bf", "#e5c07b", "#98c379", "#61afef", "#c678dd"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("ERCOT Microgrid Dispatcher  —  Multi-Config Run")
    print("=" * 65)

    # Load unified timeseries
    print(f"\n[1/3] Loading timeseries ...")
    ts = load_timeseries(OUT_DIR)
    print(f"  Rows : {len(ts):,}  |  Columns : {list(ts.columns)}")
    print(f"  Load : {ts['load_mw'].min():.1f} – {ts['load_mw'].max():.1f} MW  "
          f"(mean {ts['load_mw'].mean():.1f} MW)")
    print(f"  Solar CF : 0 – {ts['solar_cf'].max():.3f}  "
          f"(annual mean {ts['solar_cf'].mean():.4f})")

    # Run dispatch for each config
    print(f"\n[2/3] Running dispatch ...")
    all_summaries = []
    all_results   = {}

    for (S, B, G, label) in CONFIGS:
        results, summary = dispatch(S, B, G, ts)
        all_summaries.append({"config": label, **summary})
        all_results[label] = results

        # Save hourly results
        tag = label.replace(" ", "_").replace("=", "").replace(",", "")
        results.to_csv(OUT_DIR / f"ercot_dispatch_{tag}_2024.csv")

        # Print one-line summary
        print(
            f"  {label:<28}  "
            f"Solar {summary['solar_share_pct']:5.1f}%  "
            f"BESS {summary['bess_share_pct']:5.1f}%  "
            f"Gas {summary['gas_share_pct']:5.1f}%  "
            f"Unserved {summary['unserved_pct']:.3f}%  "
            f"Fuel ${summary['annual_fuel_cost_usd']/1e6:.2f}M  "
            f"CO2 {summary['annual_co2_t']/1e3:.1f}kt"
        )

    # Summary table
    summary_df = pd.DataFrame(all_summaries).set_index("config")
    summary_df.to_csv(OUT_DIR / "ercot_dispatch_summary_2024.csv")

    # Print detailed table
    print(f"\n[3/3] Detailed KPI comparison")
    print_summary_table(summary_df)

    # Plot
    print("\n  Generating plots ...")
    plot_dispatch(all_results, all_summaries, ts)

    print("\nDone.")
    return summary_df


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_summary_table(df: pd.DataFrame) -> None:
    kpis = [
        ("S_mw",                  "Solar capacity (MW DC)"),
        ("B_mwh",                 "Battery capacity (MWh)"),
        ("G_mw",                  "Gas capacity (MW)"),
        ("P_bess_max_mw",         "BESS power rating (MW)"),
        ("total_demand_mwh",      "Annual demand (MWh)"),
        ("solar_used_mwh",        "Solar used (MWh)"),
        ("curtailed_mwh",         "Solar curtailed (MWh)"),
        ("solar_curtailment_pct", "Solar curtailment (%)"),
        ("bess_discharge_mwh",    "BESS discharge (MWh)"),
        ("bess_cycles_per_year",  "BESS cycles/year"),
        ("gas_gen_mwh",           "Gas generation (MWh)"),
        ("gas_cf_pct",            "Gas capacity factor (%)"),
        ("annual_fuel_cost_usd",  "Annual fuel cost ($)"),
        ("annual_co2_t",          "Annual CO2 (t)"),
        ("solar_share_pct",       "Solar share (%)"),
        ("bess_share_pct",        "BESS share (%)"),
        ("gas_share_pct",         "Gas share (%)"),
        ("renewable_share_pct",   "Renewable share (%)"),
        ("unserved_mwh",          "Unserved energy (MWh)"),
    ]
    configs = list(df.index)
    col_w   = max(len(c) for c in configs) + 2

    # Header
    header = f"  {'KPI':<35}" + "".join(f"{c:>{col_w}}" for c in configs)
    print(header)
    print("  " + "-" * (len(header) - 2))

    fmt_map = {
        "annual_fuel_cost_usd": "${:>12,.0f}",
        "annual_co2_t":         "{:>12,.0f}",
        "total_demand_mwh":     "{:>12,.0f}",
        "solar_used_mwh":       "{:>12,.0f}",
        "curtailed_mwh":        "{:>12,.0f}",
        "bess_discharge_mwh":   "{:>12,.0f}",
        "gas_gen_mwh":          "{:>12,.0f}",
        "unserved_mwh":         "{:>12,.1f}",
    }

    for col, label in kpis:
        if col not in df.columns:
            continue
        row = f"  {label:<35}"
        for c in configs:
            val = df.loc[c, col]
            if col in fmt_map:
                cell = fmt_map[col].format(val)
            elif isinstance(val, float):
                cell = f"{val:>12.2f}"
            else:
                cell = f"{val:>12}"
            row += f"{cell:>{col_w}}"
        print(row)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_dispatch(
    all_results: dict,
    all_summaries: list,
    ts: pd.DataFrame,
) -> None:
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        "ERCOT Datacenter Microgrid — Dispatch Comparison  |  "
        "Dallas-Fort Worth, TX  |  2024\n"
        "Priority: Solar -> BESS -> Gas  (grid imports not modelled)",
        fontsize=11, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax_mix    = fig.add_subplot(gs[0, :])   # stacked energy mix bar chart
    ax_week_s = fig.add_subplot(gs[1, 0])   # sample summer week
    ax_week_w = fig.add_subplot(gs[1, 1])   # sample winter week
    ax_soc    = fig.add_subplot(gs[2, 0])   # BESS SoC for one config
    ax_gas_ldc= fig.add_subplot(gs[2, 1])   # gas generation duration curve

    MFMT = mdates.DateFormatter("%d %b\n%Hh")
    DLOC = mdates.DayLocator()

    configs  = [s["config"]  for s in all_summaries]
    colours  = CONFIG_COLOURS

    # ---- Panel 1: Annual energy mix stacked bar -------------------------
    solar_vals = [s["solar_used_mwh"]     / 1e3 for s in all_summaries]
    bess_vals  = [s["bess_discharge_mwh"] / 1e3 for s in all_summaries]
    gas_vals   = [s["gas_gen_mwh"]        / 1e3 for s in all_summaries]
    curt_vals  = [s["curtailed_mwh"]      / 1e3 for s in all_summaries]

    x = np.arange(len(configs))
    w = 0.55
    b1 = ax_mix.bar(x, solar_vals, w, label="Solar used",      color="#e5c07b", alpha=0.85)
    b2 = ax_mix.bar(x, bess_vals,  w, label="BESS discharge",  color="#61afef", alpha=0.85,
                    bottom=solar_vals)
    b3 = ax_mix.bar(x, gas_vals,   w, label="Gas",             color="#e06c75", alpha=0.85,
                    bottom=[s+b for s, b in zip(solar_vals, bess_vals)])
    # Curtailment as hatched overlay (negative indicator)
    ax_mix.bar(x, curt_vals, w, label="Solar curtailed", color="#abb2bf",
               alpha=0.5, hatch="//",
               bottom=[s+b+g for s, b, g in zip(solar_vals, bess_vals, gas_vals)])

    ax_mix.axhline(ts["load_mw"].sum() / 1e3, ls="--", color="#abb2bf", lw=0.9,
                   label=f"Annual demand ({ts['load_mw'].sum()/1e3:.0f} GWh)")
    ax_mix.set_xticks(x); ax_mix.set_xticklabels(configs, rotation=15, ha="right")
    ax_mix.set_ylabel("Annual Energy (GWh)")
    ax_mix.set_title("Annual Energy Mix by Configuration")
    ax_mix.legend(fontsize=8, ncol=5, loc="upper left")
    ax_mix.grid(True, axis="y", alpha=0.25)

    # ---- Panels 2 & 3: Sample weeks (pick config 3 for detail) ----------
    detail_label = configs[3] if len(configs) > 3 else configs[-1]
    r = all_results[detail_label]

    # Summer peak week (hottest week ~ late July)
    summer_start = pd.Timestamp("2024-07-15")
    summer_end   = summer_start + pd.Timedelta(days=7)
    sw = r.loc[summer_start:summer_end]

    ax_week_s.stackplot(
        sw.index,
        sw["solar_gen_mw"] - sw["curtailed_mw"],
        sw["bess_discharge_mw"],
        sw["gas_gen_mw"],
        labels=["Solar", "BESS dis.", "Gas"],
        colors=["#e5c07b", "#61afef", "#e06c75"],
        alpha=0.8,
    )
    ax_week_s.plot(sw.index, sw["load_mw"], "k-", lw=1.2, label="Load")
    ax_week_s.plot(sw.index, -sw["bess_charge_mw"], color="#61afef",
                   lw=0.8, ls="--", alpha=0.6, label="BESS charge (neg.)")
    ax_week_s.set_title(f"Summer Week (Jul 15-22)  |  {detail_label}")
    ax_week_s.set_ylabel("Power (MW)")
    ax_week_s.legend(fontsize=7, loc="upper right")
    ax_week_s.xaxis.set_major_formatter(MFMT); ax_week_s.xaxis.set_major_locator(DLOC)
    ax_week_s.grid(True, alpha=0.25)

    # Winter week (coldest ~ Jan)
    winter_start = pd.Timestamp("2024-01-15")
    winter_end   = winter_start + pd.Timedelta(days=7)
    ww = r.loc[winter_start:winter_end]

    ax_week_w.stackplot(
        ww.index,
        ww["solar_gen_mw"] - ww["curtailed_mw"],
        ww["bess_discharge_mw"],
        ww["gas_gen_mw"],
        labels=["Solar", "BESS dis.", "Gas"],
        colors=["#e5c07b", "#61afef", "#e06c75"],
        alpha=0.8,
    )
    ax_week_w.plot(ww.index, ww["load_mw"], "k-", lw=1.2, label="Load")
    ax_week_w.plot(ww.index, -ww["bess_charge_mw"], color="#61afef",
                   lw=0.8, ls="--", alpha=0.6, label="BESS charge (neg.)")
    ax_week_w.set_title(f"Winter Week (Jan 15-22)  |  {detail_label}")
    ax_week_w.set_ylabel("Power (MW)")
    ax_week_w.legend(fontsize=7, loc="upper right")
    ax_week_w.xaxis.set_major_formatter(MFMT); ax_week_w.xaxis.set_major_locator(DLOC)
    ax_week_w.grid(True, alpha=0.25)

    # ---- Panel 4: BESS SoC for detail config ----------------------------
    ax_soc.plot(r.index, r["soc_mwh"], color="#61afef", lw=0.5, alpha=0.8)
    s_info = next(s for s in all_summaries if s["config"] == detail_label)
    ax_soc.axhline(s_info["B_mwh"] * 0.20, ls="--", color="#e06c75",
                   lw=0.8, label=f"SoC min (20%  = {s_info['B_mwh']*0.20:.0f} MWh)")
    ax_soc.axhline(s_info["B_mwh"],        ls="--", color="#98c379",
                   lw=0.8, label=f"SoC max (100% = {s_info['B_mwh']:.0f} MWh)")
    ax_soc.set_ylabel("SoC (MWh)")
    ax_soc.set_title(f"BESS State-of-Charge  |  {detail_label}")
    ax_soc.legend(fontsize=7)
    ax_soc.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax_soc.xaxis.set_major_locator(mdates.MonthLocator())
    ax_soc.grid(True, alpha=0.25)

    # ---- Panel 5: Gas generation duration curve -------------------------
    for (S, B, G, label), col in zip(CONFIGS, colours):
        g_sorted = np.sort(all_results[label]["gas_gen_mw"].to_numpy())[::-1]
        hours_nonzero = (g_sorted > 0.01).sum()
        pct = np.arange(1, len(g_sorted) + 1) / len(g_sorted) * 100
        ax_gas_ldc.plot(pct, g_sorted, lw=1.2, color=col,
                        label=f"{label} ({hours_nonzero}h online)")
    ax_gas_ldc.set_xlabel("Duration (% of hours)")
    ax_gas_ldc.set_ylabel("Gas Output (MW)")
    ax_gas_ldc.set_title("Gas Generation Duration Curve")
    ax_gas_ldc.legend(fontsize=7)
    ax_gas_ldc.grid(True, alpha=0.25)

    plot_path = OUT_DIR / "ercot_dispatch_2024.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved -> {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
