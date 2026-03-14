# 50 MW Datacenter Microgrid Simulator

A Python simulation framework that models the energy reliability and economics of a
hybrid **Solar PV + BESS + Natural Gas** microgrid serving a 50 MW IT load, fully
islanded from the grid (behind-the-meter, no grid imports). An interactive Streamlit
dashboard allows users to select any US ISO/RTO and run the full pipeline on demand.

**Current status:** ERCOT (Dallas-Fort Worth, TX) pilot complete. Multi-ISO expansion
framework built; pipelines for PJM, MISO, CAISO, SPP, NYISO, ISONE pending.

---

## Project Objectives

- **Reliability** — guarantee zero unserved energy at every hour across all 8,784 hours
  of 2024 (leap year); find the minimum gas capacity G required for any (Solar, BESS) pair.
- **Economics** — minimise system Levelised Cost of Energy (sLCOE, $/MWh) across all
  viable (S, B, G) configurations using unsubsidised NREL ATB 2025 / Lazard 18.0 costs.
- **Multi-ISO** — replicate methodology across PJM, MISO, CAISO, SPP, NYISO, ISONE using
  a single ISO registry; each region has its own location, gas price, and CAPEX multiplier.

---

## Architecture

```
50 MW IT Load (constant)
        │
        ▼
[Demand model]  PUE(T_ambient) × 50 MW  →  total_load_mw (56–68 MW, 8,784 hrs)
        │
        ▼
[Dispatcher]  greedy priority stack per hour
   ├── ☀️  Solar PV    →  S_mw × solar_cf_t           (PVWatts V8 TMY)
   ├── 🔋  BESS        →  discharge to cover deficit  (4-hr Li-ion, η_temp-adjusted)
   └── 🔥  Gas RICE    →  P_gas = min(G, unmet)       (gap-filler)
        │
        ▼
[Reliability solver]   G_min(S, B) = max(gas_gen_t) | G=∞   (grid search)
        │
        ▼
[sLCOE optimiser]      min sLCOE = total_annual_cost / demand_MWh
        │
        ▼
[Streamlit dashboard]  ISO selector · live dispatch · reliability surface · sLCOE surface
```

No grid connection is modelled. Unserved energy = reliability failure (datacenter outage).

---

## Repository Structure

```
datacenter_microgrid_sim/
├── models/
│   ├── demand_model.py       PUE(T) piecewise-linear model; load timeseries
│   ├── solar_model.py        pvlib fallback (not used in primary pipeline)
│   ├── bess_model.py         BESS state equation; temperature efficiency; unit sizing
│   ├── gas_model.py          RICE dispatch; fuel cost; CO₂ emissions
│   ├── dispatcher.py         8,784-hour greedy dispatch loop; ISO-aware timeseries loader
│   ├── lcoe_model.py         CRF; annualised cost functions; sLCOE formula
│   ├── iso_registry.py       ISO lookup table API; regional cost helpers
│   └── pipeline.py           On-demand pipeline orchestrator for any ISO
│
├── scripts/
│   ├── 01_build_demand.py    Open-Meteo weather fetch; PUE model; load timeseries
│   ├── 02_build_solar.py     PVWatts V8 API; TMY→2024 expansion; solar_cf timeseries
│   ├── 03_build_bess.py      Temperature-adjusted η_temp timeseries
│   ├── 04_build_gas.py       Gas model characterisation; load duration curve
│   ├── 05_run_dispatch.py    Multi-config dispatch validation; KPI comparison
│   ├── 06_solve_reliability.py  G_min(S, N) surface; Pareto frontier
│   └── 07_optimize_slcoe.py  sLCOE surface; optimum; sensitivity analysis
│
├── config/
│   ├── iso_registry.json     7 ISO/RTOs: location, gas price, CAPEX multiplier
│   ├── technology_costs.json National CAPEX/OPEX baselines (Solar, BESS, Gas RICE)
│   └── secrets.env           NREL API key (gitignored — never commit)
│
├── data/
│   ├── raw/                  Cached API responses (gitignored)
│   └── processed/            Model outputs: {iso}_demand/solar/bess/reliability/slcoe CSVs
│
├── dashboard.py              Streamlit multi-ISO dashboard
├── requirements.txt          pip dependencies for deployment
└── environment.yml           Conda environment (full pinned versions)
```

---

## Methodology

### 1. Demand Timeseries

Temperature from **Open-Meteo Historical API** (ERA5, free, no key). PUE modelled as a
piecewise-linear function of ambient temperature:

| T (°C) | ≤ 10 | 25 | 35 | ≥ 60 |
|--------|------|-----|-----|------|
| PUE    | 1.12 | 1.20 | 1.32 | 1.45 |

```
total_load_t = PUE(T_t) × 50 MW
```

ERCOT 2024: avg PUE 1.190, load 56.0–67.6 MW, annual demand **522,524 MWh**.

### 2. Solar PV (PVWatts V8)

Hourly capacity factors from **NREL PVWatts V8 API** (NSRDB TMY data). Key parameters:
Premium module (γ = −0.35%/°C), fixed-tilt at latitude°, south-facing, 15% losses,
96% inverter efficiency, 1.2 DC:AC ratio. Tilt is ISO-specific (≈ latitude).

TMY 8,760 hours expanded to 8,784 (leap year) by duplicating Feb 28 as Feb 29.

ERCOT 2024: annual CF **17.05%**, peak CF 83.3%.

### 3. BESS Model

4-hour Li-ion battery, modelled as **N discrete standard units** (4 MWh / 1 MW each).

```
E_t = E_{t-1} + P_ch_t × (η_ch × η_temp_t) − P_dis_t / (η_temp_t × η_dis)

η_ch = η_dis = 0.96
SoC: 20% ≤ E_t ≤ B_mwh
Power: P ≤ B_mwh / 4  (4-hour rating)
```

`η_temp` is a piecewise-linear temperature modifier capturing HVAC parasitic loads:
T = [−10, 0, 15, 35, 45, 60°C] → η = [0.94, 0.97, 1.00, 1.00, 0.96, 0.92].

### 4. Gas (RICE)

Reciprocating Internal Combustion Engine. Gap-filler dispatch:

```
P_gas_t = min(G_mw, unmet_t)
```

Heat rate 9.0 MMBtu/MWh. Gas price is ISO-specific ($2.50/MMBtu ERCOT → $6.00 ISONE).

### 5. Reliability Solver

For fixed (S, B), minimum G for zero outages is analytically exact:

```
G_min(S, B) = max_t(gas_gen_t)  when G = ∞
```

Grid search: S ∈ [0–300 MW step 25], N ∈ [0–100 units step 5] → ~273 combinations, ~10s.

**Key ERCOT finding:** G_min ≈ 67 MW regardless of solar/BESS — a 24/7 islanded datacenter
always needs gas sized to peak load. Solar/BESS reduce gas *runtime*, not gas *capacity*.

| Configuration | Gas hrs/yr | Gas fraction |
|---|---|---|
| Gas-only | 8,784 | 100% |
| S=150 MW, N=0 | 6,913 | 79% |
| S=150 MW, N=25 (100 MWh) | 5,870 | 67% |
| S=300 MW, N=100 (400 MWh) | 2,629 | 30% |

### 6. sLCOE Optimiser

```
sLCOE = (CAPEX_ann + OPEX_fixed + OPEX_var + Fuel) / Annual_demand_MWh
```

All costs unsubsidised. 8% WACC, 25-year project life, CRF applied per technology lifetime.
BESS augmentation (cell replacement) modelled at 2.5%/yr of capex.

---

## Key Results (ERCOT 2024)

| Metric | Value |
|---|---|
| Annual demand | 522,524 MWh |
| Peak load | 67.6 MW |
| Solar CF (TMY, DFW) | 17.05% |
| G_min (any S, B) | ≈ 67 MW |
| Gas-only sLCOE | ~$45/MWh |
| Unconstrained optimum | Gas-only (S=0, B=0) — see Challenges |

---

## Challenges & Known Issues

### 1. Gas-only is the unconstrained optimum

Under unsubsidised assumptions, the sLCOE optimiser selects **S=0, B=0** (gas-only).
This is technically correct: solar at 17% CF costs ~$65/MWh LCOE, but only avoids gas
*fuel* (~$22.50/MWh) since gas capacity must remain at ~67 MW regardless. Net solar
penalty: ~$42/MWh over gas dispatch.

**Fix in progress:** add a minimum renewable fraction constraint (≥ 30% of MWh from
solar/BESS) to the optimiser. Running with ITC (30%) and/or higher gas price (CAISO/ISONE)
also makes solar competitive.

### 2. Streamlit Community Cloud deployment pending

Dashboard runs correctly locally. Cloud deployment is blocked by:
- Missing transitive dependencies in `requirements.txt`
- NREL API key not yet injected via Streamlit Cloud Secrets UI
- On-demand pipeline runner uses blocking API calls that may exceed Cloud timeouts

**Workaround:** `pip install -r requirements.txt && streamlit run dashboard.py`

### 3. TMY vs actual 2024 solar

PVWatts returns TMY data (industry standard for sizing); demand uses actual 2024
temperatures. This temporal mismatch is accepted practice but means the solar timeseries
does not reflect 2024-specific cloud events or irradiance anomalies.

---

## ISO Registry

Seven US ISO/RTOs pre-configured. Each has an indicative datacenter location, regional
gas price (EIA 2024 industrial), and CAPEX multiplier (NREL ATB 2025 regional factors):

| ISO | Location | Gas ($/MMBtu) | CAPEX mult. |
|-----|----------|--------------|-------------|
| ERCOT | Dallas-Fort Worth, TX | $2.50 | 1.00 |
| PJM | Pittsburgh, PA | $3.20 | 1.05 |
| MISO | Indianapolis, IN | $3.00 | 1.03 |
| CAISO | Fresno, CA | $5.00 | 1.15 |
| SPP | Oklahoma City, OK | $2.80 | 1.00 |
| NYISO | Albany, NY | $5.50 | 1.18 |
| ISONE | Hartford, CT | $6.00 | 1.20 |

---

## Installation

```bash
# Clone and install
git clone https://github.com/afroditifragiadaki/Microgrid_project.git
cd datacenter_microgrid_sim
pip install -r requirements.txt

# Add your NREL API key
echo "NREL_API_KEY=your_key_here" > config/secrets.env
echo "NREL_EMAIL=you@example.com" >> config/secrets.env

# Run the dashboard
streamlit run dashboard.py
```

Get a free NREL API key at [developer.nrel.gov](https://developer.nrel.gov/signup/).

On Windows, prefix Python commands with `set PYTHONIOENCODING=utf-8 &&`.

### Run scripts manually (ERCOT only)

```bash
set PYTHONIOENCODING=utf-8 && python scripts/01_build_demand.py
set PYTHONIOENCODING=utf-8 && python scripts/02_build_solar.py
set PYTHONIOENCODING=utf-8 && python scripts/03_build_bess.py
set PYTHONIOENCODING=utf-8 && python scripts/04_build_gas.py
set PYTHONIOENCODING=utf-8 && python scripts/05_run_dispatch.py
set PYTHONIOENCODING=utf-8 && python scripts/06_solve_reliability.py
set PYTHONIOENCODING=utf-8 && python scripts/07_optimize_slcoe.py
```

For other ISOs, use the dashboard's **"Run pipeline"** button — it fetches APIs and
runs all steps automatically, caching results to `data/processed/{iso}_*.csv`.

---

## Data Sources

| Data | Source | Notes |
|---|---|---|
| Ambient temperature | Open-Meteo Historical (ERA5) | Free, no key, hourly |
| Solar generation | NREL PVWatts V8 / NSRDB TMY | Free key required |
| Gas prices | EIA 2024 industrial (by state) | ISO-specific in registry |
| Solar CAPEX | NREL ATB 2025 Moderate | $950/kW DC baseline |
| BESS CAPEX | Lazard LCOS V18.0 (2024) | $280/kWh, 4-hr |
| Gas CAPEX | NREL ATB 2025 (RICE proxy) | $1,100/kW installed |
| CO₂ factor | EPA AP-42 | 0.0531 tCO₂/MMBtu |

---

## Planned Extensions

- [ ] Minimum renewable fraction constraint in sLCOE optimiser
- [ ] ITC (30% IRA 2022) and carbon price sensitivity scenarios
- [ ] Single-axis tracking option (CF → ~23%)
- [ ] Run all 7 ISO pipelines and cross-ISO comparison dashboard
- [ ] Hourly LMP integration for value-stacking analysis
- [ ] Fix Streamlit Cloud deployment
- [ ] CSV data export from dashboard
- [ ] Expand IT load configurability (variable load profile, colocation scenarios)
