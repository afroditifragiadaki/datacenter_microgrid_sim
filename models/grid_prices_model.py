"""
Synthetic hourly wholesale electricity price model.

Generates 8760/8784-hour day-ahead price timeseries per ISO, calibrated
to known 2024 annual hub averages with realistic time-of-day and seasonal
shapes, daily noise, and scarcity spikes.
"""

import numpy as np
import pandas as pd


# 2024 approximate annual average day-ahead hub prices ($/MWh)
# Sources: EIA 2024 electric power annual, ISO published annual reports
_AVG_PRICE: dict[str, float] = {
    "ERCOT": 42.0,
    "PJM":   52.0,
    "MISO":  45.0,
    "CAISO": 55.0,
    "SPP":   38.0,
    "NYISO": 60.0,
    "ISONE": 65.0,
}

# Seasonal peak type determines monthly multipliers
_PEAK_TYPE: dict[str, str] = {
    "ERCOT": "summer",
    "PJM":   "both",
    "MISO":  "summer",
    "CAISO": "summer",
    "SPP":   "summer",
    "NYISO": "winter",
    "ISONE": "winter",
}

# Normalized time-of-day shape (hour 0–23)
_TOD_SHAPE: list[float] = [
    0.65, 0.60, 0.58, 0.57, 0.58, 0.65,  # midnight–5 AM
    0.80, 0.90, 1.00, 1.05, 1.05, 1.02,  # 6–11 AM
    0.98, 0.95, 0.95, 1.00, 1.10, 1.35,  # noon–5 PM
    1.45, 1.40, 1.20, 1.00, 0.85, 0.72,  # 6–11 PM
]

# Monthly multipliers by peak type (index 0 = January)
_MONTHLY_MULT: dict[str, list[float]] = {
    "summer": [0.85, 0.85, 0.88, 0.92, 1.02, 1.12, 1.22, 1.20, 1.08, 0.92, 0.88, 0.90],
    "winter": [1.25, 1.20, 1.00, 0.88, 0.85, 0.90, 0.95, 0.93, 0.88, 0.92, 1.05, 1.22],
    "both":   [1.10, 1.08, 0.95, 0.88, 0.95, 1.08, 1.12, 1.10, 1.00, 0.92, 0.95, 1.08],
}


def build_grid_price_timeseries(
    iso_id: str,
    year: int = 2024,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic hourly grid price timeseries for an ISO.

    Returns a DataFrame (8760 or 8784 rows for leap years) with column:
        grid_price_per_mwh  — hourly day-ahead wholesale price ($/MWh)

    The profile is calibrated so its annual mean matches the known ISO
    average, with time-of-day shape, seasonal shape, daily noise, hourly
    noise, and ~40 scarcity spike hours per year.
    CAISO additionally receives negative-price hours during spring solar
    surplus (March–May midday).
    """
    rng    = np.random.default_rng(seed)
    key    = iso_id.upper()
    avg    = _AVG_PRICE.get(key, 50.0)
    ptype  = _PEAK_TYPE.get(key, "both")
    mmult  = _MONTHLY_MULT[ptype]

    n_hours = 8784 if year % 4 == 0 else 8760
    index   = pd.date_range(f"{year}-01-01", periods=n_hours, freq="h")

    # Deterministic shape vectors
    tod  = np.array([_TOD_SHAPE[ts.hour]        for ts in index])
    seas = np.array([mmult[ts.month - 1]         for ts in index])

    # Day-level noise: one draw per calendar day, applied to all hours in that day
    unique_days    = list(dict.fromkeys(index.date))  # ordered, no duplicates
    day_noise_vals = np.clip(1.0 + rng.normal(0, 0.15, len(unique_days)), 0.2, 3.0)
    day_noise_map  = dict(zip(unique_days, day_noise_vals))
    daily          = np.array([day_noise_map[d] for d in index.date])

    # Hour-level noise
    hourly = np.clip(1.0 + rng.normal(0, 0.08, n_hours), 0.2, 3.0)

    prices = avg * tod * seas * daily * hourly

    # Scarcity spikes (~40 hrs/yr replicates real ISO extreme price events)
    spike_idx = rng.choice(n_hours, size=40, replace=False)
    prices[spike_idx] *= rng.uniform(3.0, 15.0, size=40)

    # CAISO: negative prices during spring solar surplus (Mar–May, 10 AM–2 PM)
    if key == "CAISO":
        neg_mask = np.array([
            ts.month in (3, 4, 5) and 10 <= ts.hour <= 14 for ts in index
        ])
        flip     = rng.random(n_hours) < 0.25
        neg_val  = rng.uniform(5.0, 30.0, n_hours)
        prices   = np.where(neg_mask & flip, -neg_val, prices)

    # Recalibrate mean to target ISO average
    current_mean = prices.mean()
    if abs(current_mean) > 1e-6:
        prices *= avg / current_mean

    df = pd.DataFrame(
        {"grid_price_per_mwh": np.round(prices, 2)},
        index=index,
    )
    df.index.name = "datetime"
    return df
