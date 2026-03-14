"""
Build the 2024 hourly demand-side timeseries for the ERCOT datacenter model.

Weather source
--------------
Open-Meteo Historical Weather API (https://open-meteo.com/)
    - Free, no API key required
    - 2-m dry-bulb temperature at hourly resolution
    - Data accuracy: ERA5 reanalysis + station corrections (~0.5 °C RMSE)

ERCOT representative location
------------------------------
Dallas / Fort Worth, TX  (32.78 °N, 96.80 °W)
    - Largest datacenter cluster inside the ERCOT footprint
    - Captures full ERCOT summer extremes (routinely > 38 °C in July/Aug)
    - Central Time (America/Chicago)

Outputs
-------
    data/raw/ercot_weather_2024.csv          – raw hourly temperature
    data/processed/ercot_demand_2024.csv     – demand timeseries
    data/processed/ercot_demand_2024.png     – diagnostic plots
"""

import sys
from pathlib import Path

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Path setup – allow running from any working directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.demand_model import (
    build_demand_timeseries,
    demand_summary,
    pue_from_temp,
    _MODE_THRESHOLDS,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LAT           = 32.78
LON           = -96.80
YEAR          = 2024
LOCATION_NAME = "Dallas-Fort Worth, TX"
TIMEZONE      = "America/Chicago"

RAW_DIR  = PROJECT_ROOT / "data" / "raw"
OUT_DIR  = PROJECT_ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Weather fetch
# ---------------------------------------------------------------------------
def fetch_weather(lat: float, lon: float, year: int) -> pd.DataFrame:
    """Download hourly 2-m temperature from Open-Meteo for the given year."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":         lat,
        "longitude":        lon,
        "start_date":       f"{year}-01-01",
        "end_date":         f"{year}-12-31",
        "hourly":           "temperature_2m",
        "timezone":         TIMEZONE,
        "temperature_unit": "celsius",
        "wind_speed_unit":  "ms",
        "precipitation_unit": "mm",
    }
    print(f"  Fetching Open-Meteo archive for {LOCATION_NAME} ({lat}°N, {abs(lon):.2f}°W) ...")
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    times = pd.to_datetime(data["hourly"]["time"])
    temps = data["hourly"]["temperature_2m"]

    df = pd.DataFrame({"temp_c": temps}, index=times)
    df.index.name = "datetime"
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
MONTH_FMT  = mdates.DateFormatter("%b")
MONTH_LOC  = mdates.MonthLocator()

# Colour palette (neutral dark theme–friendly)
C_TEMP   = "#e06c75"
C_PUE    = "#c678dd"
C_LOAD   = "#98c379"
C_GRID   = "#abb2bf"
C_THRESH = "#d19a66"

# Cooling-mode band colours for the temperature plot
MODE_COLOURS = {
    "Free Cooling (Economizer)":   "#61afef",
    "Standard Liquid Cooling":     "#98c379",
    "Hybrid / Assisted Cooling":   "#e5c07b",
    "Max Mechanical Chilling":     "#e06c75",
}


def plot_demand(demand: pd.DataFrame, year: int) -> Path:
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(
        f"ERCOT Datacenter Demand Model  |  {LOCATION_NAME}  |  {year}",
        fontsize=13, fontweight="bold", y=0.98,
    )
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35)

    ax_temp  = fig.add_subplot(gs[0, :])   # full-year temp (wide)
    ax_pue   = fig.add_subplot(gs[1, :])   # full-year PUE  (wide)
    ax_load  = fig.add_subplot(gs[2, 0])   # full-year load
    ax_curve = fig.add_subplot(gs[2, 1])   # PUE(T) curve

    # ---- Panel 1: Temperature -------------------------------------------
    ax_temp.plot(demand.index, demand["temp_c"], color=C_TEMP, lw=0.5, alpha=0.85)
    ax_temp.axhline(35, ls="--", color=C_THRESH, lw=0.9,
                    label="Max Chilling threshold (35 °C)")
    ax_temp.axhline(25, ls="--", color="#e5c07b", lw=0.9,
                    label="Hybrid threshold (25 °C)")
    ax_temp.axhline(10, ls="--", color="#61afef", lw=0.9,
                    label="Free Cooling threshold (10 °C)")
    ax_temp.set_ylabel("Temp (°C)")
    ax_temp.set_title("Hourly Ambient Temperature — 2-m Dry Bulb")
    ax_temp.legend(fontsize=7, loc="upper right", ncol=3)
    ax_temp.xaxis.set_major_formatter(MONTH_FMT)
    ax_temp.xaxis.set_major_locator(MONTH_LOC)
    ax_temp.grid(True, alpha=0.25)

    # ---- Panel 2: PUE ---------------------------------------------------
    # Shade cooling regime bands
    boundaries = [(-50, 10), (10, 25), (25, 35), (35, 60)]
    mode_labels = [m[1] for m in _MODE_THRESHOLDS]
    y_lo, y_hi = 1.08, 1.50
    for (t_lo, t_hi), label in zip(boundaries, mode_labels):
        # find time windows where temp is in this range
        mask = (demand["temp_c"] >= t_lo) & (demand["temp_c"] < t_hi)
        ax_pue.fill_between(
            demand.index, 1.08, demand["pue"],
            where=mask,
            color=MODE_COLOURS[label], alpha=0.25, lw=0,
        )
    ax_pue.plot(demand.index, demand["pue"], color=C_PUE, lw=0.5)
    ax_pue.set_ylim(1.08, 1.50)
    ax_pue.set_ylabel("PUE")
    ax_pue.set_title("Power Usage Effectiveness (piecewise-linear, temperature-driven)")
    patches = [mpatches.Patch(color=v, alpha=0.5, label=k)
               for k, v in MODE_COLOURS.items()]
    ax_pue.legend(handles=patches, fontsize=7, loc="upper right", ncol=2)
    ax_pue.xaxis.set_major_formatter(MONTH_FMT)
    ax_pue.xaxis.set_major_locator(MONTH_LOC)
    ax_pue.grid(True, alpha=0.25)

    # ---- Panel 3: Total Load --------------------------------------------
    ax_load.fill_between(demand.index, demand["total_load_mw"],
                         alpha=0.35, color=C_LOAD)
    ax_load.plot(demand.index, demand["total_load_mw"], color=C_LOAD, lw=0.45)
    ax_load.axhline(50, ls="--", color=C_GRID, lw=0.9,
                    label="IT baseline (50 MW)")
    ax_load.set_ylabel("Total Load (MW)")
    ax_load.set_title("Total Facility Load = IT Load × PUE")
    ax_load.legend(fontsize=7)
    ax_load.xaxis.set_major_formatter(MONTH_FMT)
    ax_load.xaxis.set_major_locator(MONTH_LOC)
    ax_load.grid(True, alpha=0.25)

    # ---- Panel 4: PUE(T) curve (methodology illustration) ---------------
    t_range = np.linspace(-5, 45, 500)
    pue_curve = pue_from_temp(t_range)

    ax_curve.plot(t_range, pue_curve, color=C_PUE, lw=2, label="PUE(T) — piecewise linear")

    # Show original step-function for comparison
    def step_pue(t):
        if t < 10:   return 1.12
        elif t < 25: return 1.20
        elif t < 35: return 1.32
        else:        return 1.45

    step_vals = [step_pue(t) for t in t_range]
    ax_curve.step(t_range, step_vals, color=C_GRID, lw=1.2, alpha=0.6,
                  linestyle="--", label="Step function (original)")

    # Mark knot points
    knot_temps = [10, 25, 35]
    knot_pues  = [pue_from_temp(t) for t in knot_temps]
    ax_curve.scatter(knot_temps, knot_pues, color=C_THRESH, zorder=5,
                     s=50, label="Regime boundaries")

    ax_curve.set_xlabel("Ambient Temperature (°C)")
    ax_curve.set_ylabel("PUE")
    ax_curve.set_title("PUE Model: Interpolated vs. Step Function")
    ax_curve.legend(fontsize=7)
    ax_curve.grid(True, alpha=0.25)

    # Threshold lines
    for t in [10, 25, 35]:
        ax_curve.axvline(t, ls=":", color=C_GRID, lw=0.8, alpha=0.7)

    plot_path = OUT_DIR / f"ercot_demand_{year}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved -> {plot_path}")
    plt.show()
    return plot_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("ERCOT Demand-Side Timeseries Builder")
    print("=" * 60)

    # 1. Fetch or load cached weather
    raw_path = RAW_DIR / f"ercot_weather_{YEAR}.csv"
    if raw_path.exists():
        print(f"\n[1/4] Loading cached weather -> {raw_path}")
        weather = pd.read_csv(raw_path, index_col="datetime", parse_dates=True)
    else:
        print(f"\n[1/4] Fetching weather data ...")
        weather = fetch_weather(LAT, LON, YEAR)
        weather.to_csv(raw_path)
        print(f"       Cached -> {raw_path}")

    # Sanity check: 2024 is a leap year -> 8784 hours
    expected = 8784 if YEAR % 4 == 0 else 8760
    n = len(weather)
    if n != expected:
        print(f"  WARNING: expected {expected} rows, got {n}. "
              "Check for DST gaps/duplicates.")
    else:
        print(f"  OK: {n} hourly records ({n/24:.0f} days)")

    # 2. Handle any NaN gaps (should be rare for ERA5-based data)
    nan_count = weather["temp_c"].isna().sum()
    if nan_count > 0:
        print(f"  Interpolating {nan_count} missing temperature values ...")
        weather["temp_c"] = weather["temp_c"].interpolate(method="time")

    # 3. Build demand timeseries
    print("\n[2/4] Building demand timeseries ...")
    demand = build_demand_timeseries(weather)

    # 4. Save
    out_path = OUT_DIR / f"ercot_demand_{YEAR}.csv"
    demand.to_csv(out_path)
    print(f"\n[3/4] Saved demand timeseries -> {out_path}")

    # 5. Summary
    print("\n[4/4] Summary statistics")
    print("-" * 45)
    stats = demand_summary(demand)
    print(f"  Hours simulated    : {stats['hours']:,}  "
          f"({'leap year' if YEAR % 4 == 0 else 'standard year'})")
    print(f"  IT Load            : {50.0:.1f} MW  (constant)")
    print(f"  PUE range          : {stats['min_pue']:.3f} – {stats['max_pue']:.3f}")
    print(f"  Average PUE        : {stats['avg_pue']:.4f}")
    print(f"  Total Load range   : {stats['min_load_mw']:.2f} – "
          f"{stats['max_load_mw']:.2f} MW")
    print(f"  Average Total Load : {stats['avg_load_mw']:.3f} MW")
    print(f"  Annual Energy      : {stats['annual_energy_mwh']:,.0f} MWh")
    print(f"\n  Cooling mode distribution:")
    for mode, hrs in sorted(stats["cooling_mode_hours"].items(),
                             key=lambda x: -x[1]):
        pct = hrs / stats["hours"] * 100
        bar = "#" * int(pct / 2)
        print(f"    {mode:<35} {hrs:5d} hrs  ({pct:5.1f}%)  {bar}")

    # Temperature statistics
    print(f"\n  Temperature statistics (Dallas/Fort Worth 2024):")
    t = weather["temp_c"]
    print(f"    Min   : {t.min():.1f} °C")
    print(f"    Mean  : {t.mean():.1f} °C")
    print(f"    Max   : {t.max():.1f} °C")
    print(f"    Hours > 35 °C (max chilling): "
          f"{(t > 35).sum()} ({(t > 35).mean()*100:.1f}%)")
    print(f"    Hours < 10 °C (free cooling): "
          f"{(t < 10).sum()} ({(t < 10).mean()*100:.1f}%)")

    # 6. Plot
    print("\n  Generating plots ...")
    plot_demand(demand, YEAR)

    print("\nDone.")
    return demand


if __name__ == "__main__":
    main()
