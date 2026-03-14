"""
Reliability solver for the ERCOT BTM microgrid.

Objective
---------
Find the minimum installed gas capacity G that guarantees zero unserved
energy at every hour, for all combinations of solar (S) and battery (B).

This is a PURE RELIABILITY problem — no economics yet.
The sLCOE optimization comes in the next step.

Method
------
For a fixed (S, B), the minimum G is determined analytically by running
the dispatch with G = infinity (unconstrained gas):

    G_min(S, B)  =  max_t [ gas_gen_t ]  when G = inf

Because gas is the gap-filler: P_gas,t = min(G, unmet_t).
When G is unconstrained, gas absorbs ALL residual demand not covered by
solar + BESS. Therefore the peak gas output equals the worst-case
hourly deficit the battery cannot cover — which is exactly the minimum
G required for zero outages.

The solver sweeps a grid of (S, B) values and returns G_min for each,
producing a reliability surface and the Pareto frontier of viable
(S, B, G) triples.

Outputs
-------
    data/processed/ercot_reliability_surface_2024.csv   grid of G_min values
    data/processed/ercot_reliability_surface_2024.png   heatmap + frontier plots
"""

import sys
import time
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.dispatcher import dispatch, load_timeseries

OUT_DIR = PROJECT_ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Search grid
# ---------------------------------------------------------------------------
from models.bess_model import B_UNIT_MWH, N_UNITS_MAX

S_GRID  = np.arange(0, 301, 25)          # Solar DC capacity [MW]: 0–300 step 25
N_GRID  = np.arange(0, N_UNITS_MAX + 1,  # Number of BESS units: 0–100 step 10
                    10, dtype=int)
B_GRID  = N_GRID * B_UNIT_MWH            # Battery MWh equivalent (0–400 MWh)

# G=1e6 effectively means "unconstrained gas" for finding G_min
G_INFINITE = 1_000_000.0

YEAR = 2024


# ---------------------------------------------------------------------------
# Core: find minimum G for a single (S, B)
# ---------------------------------------------------------------------------
def find_min_G(S_mw: float, B_mwh: float, ts: pd.DataFrame) -> tuple[float, float]:
    """
    Find minimum gas capacity for zero outages given (S, B).

    Returns
    -------
    G_min_mw : float   Minimum G for 100% reliability [MW]
    gas_hours: float   Hours gas runs at > 0.1 MW output
    """
    results, _ = dispatch(S_mw, B_mwh, G_INFINITE, ts)
    G_min     = float(results["gas_gen_mw"].max())
    gas_hours = float((results["gas_gen_mw"] > 0.1).sum())
    return G_min, gas_hours


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------
def run_grid_search(ts: pd.DataFrame) -> pd.DataFrame:
    """
    Sweep the (S, B) grid and compute G_min for each combination.
    Returns a long-form DataFrame with columns: S_mw, B_mwh, G_min_mw, gas_hours_yr.
    """
    n_total = len(S_GRID) * len(B_GRID)
    rows = []
    t0 = time.time()

    print(f"  Grid: S in {S_GRID.tolist()} MW")
    print(f"        N in {N_GRID.tolist()} units  ({B_UNIT_MWH:.0f} MWh/unit)")
    print(f"        B in {B_GRID.tolist()} MWh")
    print(f"  Total combinations: {n_total}")
    print()

    for i, (S, (N, B)) in enumerate(product(S_GRID, zip(N_GRID, B_GRID))):
        G_min, gas_hrs = find_min_G(float(S), float(B), ts)
        rows.append({
            "S_mw":         S,
            "N_units":      int(N),
            "B_mwh":        B,
            "G_min_mw":     round(G_min, 2),
            "gas_hours_yr": round(gas_hrs, 0),
        })

        # Progress every 10%
        if (i + 1) % max(1, n_total // 10) == 0 or (i + 1) == n_total:
            elapsed = time.time() - t0
            pct = (i + 1) / n_total * 100
            print(f"  [{pct:5.1f}%]  {i+1:>4}/{n_total}  "
                  f"elapsed {elapsed:.1f}s  "
                  f"last: S={S} B={B} G_min={G_min:.1f} MW")

    df = pd.DataFrame(rows)
    print(f"\n  Grid search complete in {time.time()-t0:.1f}s")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("ERCOT Reliability Solver  —  Minimum G Surface")
    print("=" * 65)
    print(f"\nObjective: find G_min(S, B) such that unserved energy = 0 at all hours")
    print(f"Method   : dispatch(S, B, G=inf) -> G_min = max(gas_gen_t)")

    # 1. Load timeseries
    print(f"\n[1/4] Loading timeseries ...")
    ts = load_timeseries(OUT_DIR, YEAR)
    peak_load = ts["load_mw"].max()
    print(f"  Peak load: {peak_load:.2f} MW  (= G_min when S=0, B=0)")

    # 2. Grid search
    print(f"\n[2/4] Running grid search ...")
    surface = run_grid_search(ts)

    # 3. Save
    out_path = OUT_DIR / f"ercot_reliability_surface_{YEAR}.csv"
    surface.to_csv(out_path, index=False)
    print(f"\n[3/4] Saved -> {out_path}")

    # 4. Analysis
    print(f"\n[4/4] Reliability surface analysis")
    print("-" * 55)

    pivot = surface.pivot(index="B_mwh", columns="S_mw", values="G_min_mw")

    # Key reference points
    base_G = surface.loc[(surface.S_mw == 0) & (surface.B_mwh == 0), "G_min_mw"].values[0]
    print(f"\n  Baseline (S=0, B=0):    G_min = {base_G:.1f} MW  (= peak load)")

    # How much does solar reduce G_min (with no battery)?
    print(f"\n  Effect of solar on G_min (B=0 MWh):")
    b0 = surface[surface.B_mwh == 0].sort_values("S_mw")
    for _, row in b0.iterrows():
        reduction = base_G - row["G_min_mw"]
        pct = reduction / base_G * 100
        print(f"    S={row['S_mw']:>3.0f} MW  ->  G_min={row['G_min_mw']:>5.1f} MW  "
              f"(saves {reduction:>4.1f} MW, {pct:>4.1f}%)")

    # How much does battery reduce G_min (with S=150)?
    print(f"\n  Effect of battery on G_min (S=150 MW):")
    s150 = surface[surface.S_mw == 150].sort_values("B_mwh")
    for _, row in s150.iterrows():
        g_b0 = s150[s150.B_mwh == 0]["G_min_mw"].values[0]
        reduction = g_b0 - row["G_min_mw"]
        print(f"    N={row['N_units']:>3.0f} units ({row['B_mwh']:>4.0f} MWh)  ->  "
              f"G_min={row['G_min_mw']:>5.1f} MW  "
              f"(saves {reduction:>4.1f} MW vs N=0)")

    # Minimum G achievable (with large S and B)
    min_G = surface["G_min_mw"].min()
    min_G_row = surface.loc[surface["G_min_mw"].idxmin()]
    print(f"\n  Minimum achievable G_min: {min_G:.1f} MW")
    print(f"    at S={min_G_row['S_mw']:.0f} MW, B={min_G_row['B_mwh']:.0f} MWh")

    # Iso-G lines: what (S, N) combinations keep G_min <= threshold?
    print(f"\n  Pareto frontier — min S+N to keep G_min <= threshold:")
    for G_thresh in [60, 50, 40, 30, 20, 10]:
        feasible = surface[surface["G_min_mw"] <= G_thresh]
        if feasible.empty:
            print(f"    G <= {G_thresh:>2.0f} MW:  not achievable in search space")
            continue
        best = feasible.sort_values("S_mw").iloc[0]
        print(f"    G <= {G_thresh:>2.0f} MW:  e.g. S={best['S_mw']:>3.0f} MW, "
              f"N={best['N_units']:>3.0f} units ({best['B_mwh']:>4.0f} MWh)  "
              f"(gas runs {best['gas_hours_yr']:.0f} hrs/yr)")

    # 5. Plot
    print(f"\n  Generating plots ...")
    plot_reliability(surface, pivot, ts, YEAR)

    print("\nDone.")
    return surface


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_reliability(surface: pd.DataFrame, pivot: pd.DataFrame,
                     ts: pd.DataFrame, year: int) -> None:
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"ERCOT Microgrid Reliability Solver  |  Dallas-Fort Worth, TX  |  {year}\n"
        f"G_min(S, B) = minimum gas capacity for zero outages at all 8,784 hours",
        fontsize=11, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    ax_heat  = fig.add_subplot(gs[0, :])   # heatmap (wide)
    ax_solar = fig.add_subplot(gs[1, 0])   # G_min vs S for different B
    ax_batt  = fig.add_subplot(gs[1, 1])   # G_min vs B for different S

    peak_load = ts["load_mw"].max()

    # Custom colormap: green (low G) -> yellow -> red (high G = gas-heavy)
    cmap = LinearSegmentedColormap.from_list(
        "reliability", ["#98c379", "#e5c07b", "#e06c75"], N=256
    )

    # ---- Panel 1: Heatmap G_min(S, B) -----------------------------------
    im = ax_heat.imshow(
        pivot.values,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=0,
        vmax=peak_load,
        extent=[
            S_GRID.min() - 12.5, S_GRID.max() + 12.5,
            B_GRID.min() - 50,   B_GRID.max() + 50,
        ],
    )
    cbar = plt.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.01)
    cbar.set_label("G_min (MW)", fontsize=9)

    # Contour lines for iso-G thresholds
    S_fine = S_GRID
    B_fine = B_GRID
    Z = pivot.values  # shape: (n_B, n_S)
    contour_levels = [10, 20, 30, 40, 50, 60]
    cs = ax_heat.contour(S_fine, B_fine, Z, levels=contour_levels,
                         colors="white", linewidths=0.9, alpha=0.8)
    ax_heat.clabel(cs, fmt="%g MW", fontsize=8, inline=True)

    ax_heat.set_xlabel("Solar Capacity S (MW DC)")
    ax_heat.set_ylabel("Battery Capacity B (MWh)")
    ax_heat.set_title("Minimum Gas Capacity G_min Required for 100% Reliability\n"
                      "White contours = iso-G lines (MW)")

    # ---- Panel 2: G_min vs S (for key B values) -------------------------
    colours_B = ["#abb2bf", "#61afef", "#98c379", "#e5c07b", "#c678dd"]
    B_show = [0, 200, 400, 600, 800]
    for B_val, col in zip(B_show, colours_B):
        sub = surface[surface.B_mwh == B_val].sort_values("S_mw")
        ax_solar.plot(sub["S_mw"], sub["G_min_mw"], "o-", color=col,
                      lw=1.5, ms=4, label=f"B={B_val} MWh")
    ax_solar.axhline(peak_load, ls="--", color="#e06c75", lw=0.8,
                     label=f"Peak load ({peak_load:.1f} MW)")
    ax_solar.set_xlabel("Solar Capacity S (MW DC)")
    ax_solar.set_ylabel("G_min (MW)")
    ax_solar.set_title("G_min vs Solar — by Battery Size")
    ax_solar.legend(fontsize=8); ax_solar.grid(True, alpha=0.25)
    ax_solar.set_ylim(0, peak_load * 1.05)

    # ---- Panel 3: G_min vs B (for key S values) -------------------------
    colours_S = ["#abb2bf", "#e5c07b", "#98c379", "#61afef", "#c678dd"]
    S_show = [0, 75, 150, 200, 250]
    for S_val, col in zip(S_show, colours_S):
        sub = surface[surface.S_mw == S_val].sort_values("B_mwh")
        ax_batt.plot(sub["B_mwh"], sub["G_min_mw"], "o-", color=col,
                     lw=1.5, ms=4, label=f"S={S_val} MW")
    ax_batt.axhline(peak_load, ls="--", color="#e06c75", lw=0.8,
                    label=f"Peak load ({peak_load:.1f} MW)")
    ax_batt.set_xlabel("Battery Capacity B (MWh)")
    ax_batt.set_ylabel("G_min (MW)")
    ax_batt.set_title("G_min vs Battery — by Solar Capacity")
    ax_batt.legend(fontsize=8); ax_batt.grid(True, alpha=0.25)
    ax_batt.set_ylim(0, peak_load * 1.05)

    plot_path = OUT_DIR / f"ercot_reliability_surface_{year}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved -> {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
