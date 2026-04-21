"""
Hourly wholesale electricity price fetcher for US ISOs.

Pulls 2024 day-ahead LMP data from public ISO APIs via the gridstatus
library. Fetching is parallelized (monthly chunks via ThreadPoolExecutor)
so the first call per ISO typically completes in 1–5 minutes and is then
cached to CSV by the pipeline.

Falls back to a calibrated synthetic profile if the fetch fails or returns
incomplete data (< 90% of expected hours).

ISO → hub/zone used for representative system-average price
------------------------------------------------------------
ERCOT  : average of all Trading Hubs  (get_dam_spp)
CAISO  : TH_SP15_GEN-APND             (SP15 hub, DA hourly)
PJM    : Western Hub                   (DA hourly hubs)
MISO   : system average of all zones   (DA hourly, daily parallel fetch)
NYISO  : average of all zones          (DA hourly)
ISONE  : average of all locations      (DA hourly)
SPP    : average of all hubs           (DA hourly, HUB type)
"""

import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Synthetic fallback — calibrated to 2024 ISO hub averages
# ---------------------------------------------------------------------------

_AVG_PRICE: dict[str, float] = {
    "ERCOT": 42.0,
    "PJM":   52.0,
    "MISO":  45.0,
    "CAISO": 55.0,
    "SPP":   38.0,
    "NYISO": 60.0,
    "ISONE": 65.0,
}

_PEAK_TYPE: dict[str, str] = {
    "ERCOT": "summer", "PJM": "both",  "MISO": "summer",
    "CAISO": "summer", "SPP": "summer","NYISO": "winter", "ISONE": "winter",
}

_TOD_SHAPE = [
    0.65, 0.60, 0.58, 0.57, 0.58, 0.65,
    0.80, 0.90, 1.00, 1.05, 1.05, 1.02,
    0.98, 0.95, 0.95, 1.00, 1.10, 1.35,
    1.45, 1.40, 1.20, 1.00, 0.85, 0.72,
]

_MONTHLY_MULT = {
    "summer": [0.85,0.85,0.88,0.92,1.02,1.12,1.22,1.20,1.08,0.92,0.88,0.90],
    "winter": [1.25,1.20,1.00,0.88,0.85,0.90,0.95,0.93,0.88,0.92,1.05,1.22],
    "both":   [1.10,1.08,0.95,0.88,0.95,1.08,1.12,1.10,1.00,0.92,0.95,1.08],
}


def _synthetic(iso_id: str, year: int, seed: int = 42) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    key   = iso_id.upper()
    avg   = _AVG_PRICE.get(key, 50.0)
    mmult = _MONTHLY_MULT[_PEAK_TYPE.get(key, "both")]
    n     = 8784 if year % 4 == 0 else 8760
    idx   = pd.date_range(f"{year}-01-01", periods=n, freq="h")

    tod  = np.array([_TOD_SHAPE[ts.hour]       for ts in idx])
    seas = np.array([mmult[ts.month - 1]        for ts in idx])
    udays = list(dict.fromkeys(idx.date))
    dnoise_map = dict(zip(udays, np.clip(1 + rng.normal(0, .15, len(udays)), .2, 3)))
    daily  = np.array([dnoise_map[d] for d in idx.date])
    hourly = np.clip(1 + rng.normal(0, .08, n), .2, 3)
    prices = avg * tod * seas * daily * hourly
    spike  = rng.choice(n, 40, replace=False)
    prices[spike] *= rng.uniform(3, 15, 40)
    if key == "CAISO":
        neg_mask = np.array([ts.month in (3,4,5) and 10 <= ts.hour <= 14 for ts in idx])
        prices = np.where(neg_mask & (rng.random(n) < .25), -rng.uniform(5, 30, n), prices)
    prices *= avg / prices.mean()
    df = pd.DataFrame({"grid_price_per_mwh": np.round(prices, 2)}, index=idx)
    df.index.name = "datetime"
    return df


# ---------------------------------------------------------------------------
# Real-data helpers
# ---------------------------------------------------------------------------

def _to_hourly_avg(
    df: pd.DataFrame,
    time_col: str,
    price_col: str,
    year: int,
) -> pd.Series:
    """Average multi-location LMP DataFrame to a single hourly Series."""
    ts = pd.to_datetime(df[time_col], utc=True).dt.tz_convert(None).dt.floor("h")
    hourly = df.assign(_h=ts).groupby("_h")[price_col].mean()
    n = 8784 if year % 4 == 0 else 8760
    full_idx = pd.date_range(f"{year}-01-01", periods=n, freq="h")
    hourly = hourly.reindex(full_idx).interpolate(method="linear", limit=6).ffill().bfill()
    return hourly


def _fetch_monthly_parallel(
    iso_obj,
    year: int,
    get_lmp_kwargs: dict,
    n_workers: int = 6,
) -> pd.DataFrame | None:
    """Fetch a full year month-by-month in parallel (for ISOs that accept end=)."""
    months = pd.date_range(f"{year}-01-01", f"{year+1}-01-01", freq="MS")

    def fetch(start):
        end = (start + pd.DateOffset(months=1)).strftime("%Y-%m-%d")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return iso_obj.get_lmp(start.strftime("%Y-%m-%d"), end=end, **get_lmp_kwargs)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        chunks = list(pool.map(fetch, months))

    valid = [c for c in chunks if c is not None and len(c) > 0]
    return pd.concat(valid, ignore_index=True) if valid else None


def _fetch_daily_parallel(
    iso_obj,
    year: int,
    get_lmp_kwargs: dict,
    n_workers: int = 20,
) -> pd.DataFrame | None:
    """Fetch a full year day-by-day in parallel (for ISOs without end= support)."""
    days = pd.date_range(f"{year}-01-01", f"{year+1}-01-01", freq="D")[:-1]

    def fetch(d):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return iso_obj.get_lmp(d.strftime("%Y-%m-%d"), **get_lmp_kwargs)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        chunks = list(pool.map(fetch, days))

    valid = [c for c in chunks if c is not None and len(c) > 0]
    return pd.concat(valid, ignore_index=True) if valid else None


def _is_complete(series: pd.Series, year: int, threshold: float = 0.90) -> bool:
    n = 8784 if year % 4 == 0 else 8760
    return series.notna().sum() >= int(n * threshold)


# ---------------------------------------------------------------------------
# Per-ISO real fetchers
# ---------------------------------------------------------------------------

def _fetch_ercot(year: int) -> pd.Series | None:
    import gridstatus
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = gridstatus.Ercot().get_dam_spp(year)
        hourly = (
            df.assign(_h=pd.to_datetime(df["Interval Start"], utc=True)
                         .dt.tz_convert(None).dt.floor("h"))
            .groupby("_h")["SPP"].mean()
        )
        n = 8784 if year % 4 == 0 else 8760
        full = pd.date_range(f"{year}-01-01", periods=n, freq="h")
        return hourly.reindex(full).interpolate(limit=6).ffill().bfill()
    except Exception:
        return None


def _fetch_caiso(year: int) -> pd.Series | None:
    import gridstatus
    try:
        df = _fetch_monthly_parallel(
            gridstatus.CAISO(), year,
            {"market": "DAY_AHEAD_HOURLY", "locations": ["TH_SP15_GEN-APND"]},
        )
        return _to_hourly_avg(df, "Interval Start", "LMP", year) if df is not None else None
    except Exception:
        return None


def _fetch_pjm(year: int) -> pd.Series | None:
    import gridstatus
    try:
        df = _fetch_monthly_parallel(
            gridstatus.PJM(), year,
            {"market": "DAY_AHEAD_HOURLY", "location_type": "hub"},
        )
        return _to_hourly_avg(df, "Interval Start", "LMP", year) if df is not None else None
    except Exception:
        return None


def _fetch_miso(year: int) -> pd.Series | None:
    import gridstatus
    try:
        df = _fetch_daily_parallel(
            gridstatus.MISO(), year,
            {"market": "DAY_AHEAD_HOURLY"},
            n_workers=20,
        )
        return _to_hourly_avg(df, "Interval Start", "LMP", year) if df is not None else None
    except Exception:
        return None


def _fetch_nyiso(year: int) -> pd.Series | None:
    import gridstatus
    try:
        df = _fetch_monthly_parallel(
            gridstatus.NYISO(), year,
            {"market": "DAY_AHEAD_HOURLY", "location_type": "zone"},
        )
        return _to_hourly_avg(df, "Interval Start", "LMP", year) if df is not None else None
    except Exception:
        return None


def _fetch_isone(year: int) -> pd.Series | None:
    import gridstatus
    try:
        df = _fetch_monthly_parallel(
            gridstatus.ISONE(), year,
            {"market": "DAY_AHEAD_HOURLY"},
        )
        return _to_hourly_avg(df, "Interval Start", "LMP", year) if df is not None else None
    except Exception:
        return None


def _fetch_spp(year: int) -> pd.Series | None:
    import gridstatus
    try:
        df = _fetch_monthly_parallel(
            gridstatus.SPP(), year,
            {"market": "DAY_AHEAD_HOURLY", "location_type": "HUB"},
        )
        return _to_hourly_avg(df, "Interval Start", "LMP", year) if df is not None else None
    except Exception:
        return None


_FETCHERS: dict[str, Callable] = {
    "ERCOT": _fetch_ercot,
    "CAISO": _fetch_caiso,
    "PJM":   _fetch_pjm,
    "MISO":  _fetch_miso,
    "NYISO": _fetch_nyiso,
    "ISONE": _fetch_isone,
    "SPP":   _fetch_spp,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_grid_price_timeseries(
    iso_id: str,
    year: int = 2024,
    log: Callable = print,
) -> pd.DataFrame:
    """
    Build an 8760/8784-row hourly grid price timeseries for an ISO.

    Tries to fetch real day-ahead LMP data from the ISO's public API via
    gridstatus. Falls back to a calibrated synthetic profile if the fetch
    fails or returns incomplete data (< 90% of expected hours).

    Returns a DataFrame with column: grid_price_per_mwh ($/MWh)
    """
    key = iso_id.upper()
    n   = 8784 if year % 4 == 0 else 8760
    fetcher = _FETCHERS.get(key)

    if fetcher is not None:
        log(f"  grid prices: fetching real DA LMP for {key} {year} "
            f"(this runs once and is cached) ...")
        try:
            series = fetcher(year)
            if series is not None and _is_complete(series, year):
                log(f"  grid prices: real data OK — "
                    f"mean=${series.mean():.2f}/MWh, "
                    f"filled={series.isna().sum()} gaps")
                df = pd.DataFrame(
                    {"grid_price_per_mwh": np.round(series.values, 2)},
                    index=series.index,
                )
                df.index.name = "datetime"
                return df
            else:
                filled = 0 if series is None else int(series.notna().sum())
                log(f"  grid prices: real fetch incomplete "
                    f"({filled}/{n} hours) — using synthetic fallback")
        except Exception as exc:
            log(f"  grid prices: real fetch failed ({exc}) — "
                f"using synthetic fallback")
    else:
        log(f"  grid prices: no fetcher for {key} — using synthetic")

    df = _synthetic(key, year)
    log(f"  grid prices: synthetic — mean=${df.grid_price_per_mwh.mean():.2f}/MWh")
    return df
