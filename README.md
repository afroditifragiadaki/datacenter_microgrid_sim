# 50 MW Datacenter Microgrid Simulation

A Python simulation framework that models the energy reliability and economics of a
hybrid Solar + BESS + Gas microgrid serving a 50 MW IT load, fully islanded from the
grid (behind-the-meter, no grid imports). The first implementation targets ERCOT
(Dallas-Fort Worth, TX) with 2024 as the simulation year.

---

## Project Objectives

- **Reliability** — guarantee zero unserved energy at every hour across all 8,784 hours
  of 2024 (leap year); identify the minimum gas capacity G required for any (Solar, BESS)
  combination.
- **Economic optimisation** — minimise system Levelised Cost of Energy (sLCOE) across
  all viable (S, B, G) configurations.
- **Extensibility** — methodology designed to be replicated across PJM, MISO, CAISO,
  SPP, NYISO once the ERCOT baseline is validated.

---

## Architecture Overview

```
50 MW IT Load
    |
    v
[Demand model]  PUE(T) * 50 MW  -->  total_load_mw  (56-68 MW, 8,784 hrs)
    |
    v
[Dispatcher]  priority stack: Solar -> BESS -> Gas
    |                                              |
[Solar PV]  PVWatts V8 TMY CF x S_mw             |
[BESS]      4-hr Li-ion, SoC [20-100%], T-adj    |
[Gas RICE]  gap-filler: P_gas = min(G, unmet)    |
    |
    v
[Reliability solver]  G_min(S, B) = max(gas_gen_t) | G=inf
    |
    v
[sLCOE optimiser]   min sLCOE = total_annual_cost / demand_MWh
```

Dispatch priority (greedy, hourly):
1. Solar PV — zero marginal cost, always taken first
2. BESS discharge — covers residual deficit after solar
3. Natural gas (RICE) — gap-filler of last resort
4. Unserved energy — any remaining deficit (reliability failure)

No grid connection is modelled. This is a fully islanded BTM microgrid.

---

## Repository Structure

```
datacenter_microgrid_sim/
|-- models/
|   |-- demand_model.py       PUE(T) model; load timeseries builder
|   |-- solar_model.py        pvlib fallback (not used in primary pipeline)
|   |-- bess_model.py         BESS state equation; temperature efficiency
|   |-- gas_model.py          RICE dispatch; fuel consumption; CO2 emissions
|   |-- dispatcher.py         Hourly dispatch loop; timeseries loader
|   |-- lcoe_model.py         Financial parameters; annualised cost functions; sLCOE
|
|-- scripts/
|   |-- 01_build_demand.py    Fetch weather (Open-Meteo); build 8,784-hr load timeseries
|   |-- 02_build_solar.py     PVWatts V8 API; TMY->2024 expansion; solar_cf timeseries
|   |-- 03_build_bess.py      Temperature-adjusted BESS efficiency timeseries
|   |-- 04_build_gas.py       Gas model characterisation; load duration curve; sizing
|   |-- 05_run_dispatch.py    Multi-config dispatch validation; KPI comparison
|   |-- 06_solve_reliability.py  G_min(S,B) surface; Pareto frontier
|   |-- 07_optimize_slcoe.py  sLCOE surface; optimum; sensitivity analysis
|
|-- data/
|   |-- raw/                  Cached API responses (gitignored)
|   |-- processed/            All model outputs (CSVs + PNGs)
|
|-- config/
|   |-- secrets.env           API keys (gitignored)
|
|-- notebooks/                Exploratory analysis (Jupyter)
|-- environment.yml           Conda environment
```

---

## Models

### Demand Model (`models/demand_model.py`)

IT load is constant at 50 MW. Total facility load scales with the Power Usage
Effectiveness (PUE), which is a piecewise-linear function of ambient temperature:

| Temperature (°C) | PUE  | Cooling mode              |
|------------------|------|---------------------------|
| <= 10            | 1.12 | Free cooling / economizer |
| 25               | 1.20 | Standard liquid cooling   |
| 35               | 1.32 | Hybrid / assisted         |
| >= 60            | 1.45 | Max mechanical chilling   |

Values between knots are linearly interpolated (`np.interp`).

```
total_load_mw = PUE(T_ambient) * 50 MW
```

2024 results (DFW): avg PUE = 1.190, load range 56.0–67.6 MW,
annual demand = 522,524 MWh.

Temperature data: Open-Meteo historical archive (ERA5), UTC, 2024.

### Solar PV Model (`scripts/02_build_solar.py`)

Uses the **NREL PVWatts V8 API** with NSRDB TMY data for DFW (32.78°N, 96.80°W).

| Parameter       | Value                          |
|-----------------|--------------------------------|
| Module type     | Premium (N-type, γ = -0.35%/°C)|
| Array type      | Fixed open rack                |
| Tilt / Azimuth  | 32.78° / 180° (south-facing)   |
| Losses          | 15%                            |
| Inverter eff.   | 96%                            |
| DC:AC ratio     | 1.2                            |
| NSRDB station   | 702381.csv (2.2 km from site)  |

```
P_solar_t = S_mw * solar_cf_t
```

2024 CF (TMY expanded to 8,784 hrs): annual mean = 17.05%, peak = 83.3%.
PVWatts returns 8,760 TMY hours; Feb 29 is inserted by duplicating Feb 28.

### BESS Model (`models/bess_model.py`)

4-hour Li-ion battery. State equation per hour:

```
E_t = E_{t-1} + P_ch_t * (eta_ch * eta_temp_t)
                - P_dis_t / (eta_temp_t * eta_dis)

Constraints:
  0.20 * B <= E_t <= B          (SoC limits)
  P_ch_t, P_dis_t <= B / 4     (4-hour power rating)
```

- `eta_ch = eta_dis = 0.96`
- `eta_temp` — temperature-adjusted HVAC parasitic modifier:

| T (°C)  | -10  |  0   |  15  |  35  |  45  |  60  |
|---------|------|------|------|------|------|------|
| eta_temp| 0.94 | 0.97 | 1.00 | 1.00 | 0.96 | 0.92 |

2024 results (DFW): mean eta_temp = 0.9961, cold hours = 2,451 (28%), optimal = 6,061 (69%).

### Gas Model (`models/gas_model.py`)

Technology: Natural Gas Reciprocating Engine (RICE).
Chosen for: better part-load efficiency, low minimum stable load (30%), fast warm-start.

```
P_gas_t = min(G_mw, unmet_demand_t)
Fuel_t  = P_gas_t * 9.0  [MMBtu/h]
```

| Parameter         | Value                          |
|-------------------|--------------------------------|
| Heat rate         | 9.0 MMBtu/MWh (HHV, full load) |
| Thermal efficiency| 37.9%                          |
| Gas price         | $2.50/MMBtu (EIA 2024, Texas)  |
| Fuel cost         | $22.50/MWh                     |
| CO2 factor        | 0.0531 tCO2/MMBtu = 0.478 tCO2/MWh |
| Min stable load   | 30% of G                       |

### Dispatcher (`models/dispatcher.py`)

Greedy hourly priority stack. For each hour t:

- **Surplus** (solar > load): charge BESS up to power/SoC limits; curtail remainder.
- **Deficit** (solar < load): discharge BESS up to power/SoC limits; gas fills the rest.

No grid imports. Unserved energy = reliability failure (datacenter outage).

### Financial Model (`models/lcoe_model.py`)

All costs are **unsubsidized** (pre-incentive), consistent with Lazard LCOE+ 18.0.

| Component  | CAPEX           | Fixed O&M       | Variable O&M   | Life  | Source            |
|------------|-----------------|-----------------|----------------|-------|-------------------|
| Solar PV   | $950/kW DC      | $16/kW/yr       | —              | 30 yr | NREL ATB 2025     |
| BESS 4-hr  | $300/kWh        | $8/kW-power/yr  | $0.50/MWh dis. | 15 yr | NREL ATB 2025     |
| Gas RICE   | $1,200/kW       | $20/kW/yr       | $7.00/MWh      | 25 yr | Lazard LCOE+ 18.0 |

Project: 25-year life, 7% nominal WACC, CRF = 8.58%/yr.
BESS replacement at year 15 at 85% of initial capex (technology learning).
ITC (30% IRA 2022) tracked separately; not applied to baseline.

```
sLCOE = (C_solar + C_bess + C_gas) / Annual_demand_MWh
```

---

## Pipeline

Run scripts in order. Each step outputs to `data/processed/`.

```bash
# 0. Install environment
conda env create -f environment.yml
conda activate microgrid_sim

# 1. Demand timeseries (Open-Meteo weather + PUE model)
python scripts/01_build_demand.py

# 2. Solar timeseries (PVWatts V8 API — requires NREL API key)
python scripts/02_build_solar.py

# 3. BESS efficiency timeseries
python scripts/03_build_bess.py

# 4. Gas model characterisation and load duration curve
python scripts/04_build_gas.py

# 5. Dispatch validation across multiple (S, B, G) configurations
python scripts/05_run_dispatch.py

# 6. Reliability surface: G_min(S, B) for zero outages
python scripts/06_solve_reliability.py

# 7. sLCOE optimisation: minimum cost (S, B, G) configuration
python scripts/07_optimize_slcoe.py
```

On Windows, prefix with `set PYTHONIOENCODING=utf-8 &&` to avoid encoding errors.

---

## Key Results (ERCOT 2024)

### Demand
- Annual demand: **522,524 MWh/yr**
- Load range: **56.0 – 67.6 MW** (avg 59.5 MW)
- Avg PUE: **1.190**

### Solar
- Annual capacity factor: **17.05%** (TMY, DFW, fixed-tilt)
- Peak CF: **83.3%**

### Reliability Surface
The minimum gas capacity G required for zero outages is **essentially independent of
solar and battery size** within the modelled range (S ≤ 300 MW, B ≤ 1,000 MWh):

- G_min baseline (S=0, B=0): **67.6 MW** (= peak load)
- G_min at S=300 MW, B=1,000 MWh: **66.5 MW** (only 1.1 MW reduction)

**Why:** The datacenter runs 24/7 at 56–68 MW. At night, solar = 0. Even 1,000 MWh
of battery (usable 800 MWh) barely covers one night at minimum load. Gas must therefore
always be sized close to peak load regardless of solar/BESS scale.

However, solar + BESS dramatically reduce **gas runtime** (and thus fuel cost):

| Configuration       | Gas hrs/yr | Gas fraction |
|---------------------|-----------|--------------|
| S=0, B=0 (gas-only) | 8,784     | 100%         |
| S=150 MW, B=0       | 6,913     | 79%          |
| S=150 MW, B=400 MWh | 5,870     | 67%          |
| S=300 MW, B=1,000 MWh | 2,629   | 30%          |

### sLCOE Optimisation (unsubsidized baseline)
- **Minimum sLCOE: $45.42/MWh — gas-only (S=0, B=0, G=67.6 MW)**
- Adding solar increases sLCOE under current unsubsidized assumptions

**Why solar doesn't pencil out (yet):**
Solar LCOE at 17% CF and $950/kW ≈ **$65/MWh**. Since G_min is fixed, solar only
avoids gas *fuel* (not gas capex/fixed O&M). Gas variable cost = **$29.50/MWh**.
Net cost of solar = $65 − $29.50 = **+$35/MWh penalty**.

**Levers that change the result:**
- **ITC (30% IRA 2022):** solar capex → $665/kW effective → LCOE ~$49/MWh (still above $29.50)
- **Higher gas price ($5/MMBtu):** fuel savings → $52.50/MWh → solar becomes competitive
- **Single-axis tracking:** CF → 23% → solar LCOE → ~$50/MWh
- **Carbon pricing:** adds cost to gas, improves renewable economics

---

## Configuration

### API Keys (`config/secrets.env`)
```
NREL_API_KEY=<your_key>
NREL_EMAIL=<your_email>
```
Obtain a free key at [developer.nrel.gov](https://developer.nrel.gov/signup/).
This file is gitignored; never commit credentials.

### Key Constants

| File                  | Parameter                   | Default        |
|-----------------------|-----------------------------|----------------|
| `models/demand_model.py` | IT load                  | 50 MW          |
| `models/demand_model.py` | PUE knots                | see table above|
| `models/gas_model.py`   | Heat rate                 | 9.0 MMBtu/MWh  |
| `models/gas_model.py`   | Gas price                 | $2.50/MMBtu    |
| `models/lcoe_model.py`  | Discount rate             | 7%             |
| `models/lcoe_model.py`  | Solar CAPEX               | $950/kW DC     |
| `models/lcoe_model.py`  | BESS CAPEX                | $300/kWh       |
| `models/lcoe_model.py`  | Gas CAPEX                 | $1,200/kW      |
| `scripts/06_solve_reliability.py` | Solar grid  | 0–300 MW, step 25 |
| `scripts/06_solve_reliability.py` | BESS grid   | 0–1,000 MWh, step 100 |

---

## Data Sources

| Data               | Source                         | Coverage          |
|--------------------|--------------------------------|-------------------|
| Ambient temperature| Open-Meteo historical (ERA5)   | DFW, 2024, hourly |
| Solar irradiance   | NREL PVWatts V8 / NSRDB TMY   | DFW, TMY          |
| Gas price          | EIA 2024 Texas Henry Hub       | Annual avg        |
| Capital costs      | NREL ATB 2025 (moderate)       | 2025 USD          |
| LCOE benchmarks    | Lazard LCOE+ 18.0 (June 2025)  | Unsubsidized      |
| CO2 factor         | EPA AP-42                      | Pipeline gas      |

---

## Planned Extensions

- [ ] Refine sLCOE assumptions: ITC, carbon price, single-axis tracking scenario
- [ ] Expand to PJM, MISO, CAISO, SPP, NYISO (same methodology, different weather + fuel prices)
- [ ] Add hourly ERCOT LMPs for behind-the-meter value stacking analysis
- [ ] Model multi-day storage to reduce G_min requirement
- [ ] Optimize across finer (S, B) grid near the economic frontier

---

## Installation

```bash
git clone https://github.com/afroditifragiadaki/Microgrid_project.git
cd datacenter_microgrid_sim
conda env create -f environment.yml
conda activate microgrid_sim
```

Dependencies: `numpy`, `pandas`, `matplotlib`, `requests`, `pvlib`, `python-dotenv`.
See `environment.yml` for full pinned versions.
