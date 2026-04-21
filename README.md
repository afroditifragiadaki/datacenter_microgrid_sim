# Datacenter Microgrid Cost Explorer

**ELEN4510 Grid Modernization & Clean Tech — Columbia University**

An interactive optimization tool that models and **compares two energy strategies** for a
behind-the-meter collocated datacenter:

- **Microgrid** — fully islanded Solar PV + BESS + Gas, no grid connection
- **Grid-Connected** — Solar PV + BESS + Gas + utility grid, with hourly price-optimal dispatch

For each of the seven major US electricity markets (ISO/RTOs), the tool finds the
lowest-cost asset configuration for both strategies under a user-defined renewable
energy share target, then shows which option is cheaper and by how much.

---

## Team

| Name | Email |
|------|-------|
| Afroditi Fragkiadaki | af3619@columbia.edu |
| Daniel Holland | doh2105@columbia.edu |
| Raphael Vogeley | rpv2113@columbia.edu |
| Tselmeg Mendsaikhan | tm3516@columbia.edu |

---

## What It Does

Given a datacenter IT load (MW) and a minimum on-site renewable share (%), the tool:

1. Searches across Solar + BESS + Gas size combinations for both models
2. Simulates 8,784 hours of hourly dispatch for each combination
3. Filters out designs below the renewable floor
4. Returns the minimum-sLCOE design per market per model
5. Compares microgrid vs grid-connected sLCOE across all seven markets

---

## Dashboard

Run locally with:

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

Add your NREL API key to `config/secrets.env` (free at [developer.nrel.gov](https://developer.nrel.gov/signup/)):

```
NREL_API_KEY=your_key_here
NREL_EMAIL=you@example.com
```

| Tab | Description |
|-----|-------------|
| **Optimizer** | Configure IT load and renewable floor → compare all markets (microgrid + grid) → deep dive into any market's cost breakdown, energy mix, and grid-connected comparison |
| **Methodology** | Pipeline flowcharts, sLCOE formulas, dispatch logic for both models, data sources |
| **Team** | Project team and contact information |

---

## Methodology

### Two Models

#### 1 — Microgrid (fully islanded)

The site operates with no grid connection. Solar, BESS, and gas must cover 100% of load.
Gas is sized to the minimum capacity needed to guarantee zero unserved energy.

**Dispatch priority each hour:**
1. Solar → serves load directly
2. Surplus solar → charges BESS
3. Deficit after solar → BESS discharges
4. Remaining deficit → gas fills the gap

#### 2 — Grid-Connected

The site has a utility grid connection. Solar and BESS operate the same way, but for any
remaining deficit, the cheaper of gas or grid is used each hour based on the real-time
day-ahead LMP.

**Dispatch priority each hour:**
1. Solar → serves load directly
2. Surplus solar → charges BESS
3. Deficit after solar → BESS discharges
4. Remaining deficit → **compare gas marginal cost vs grid price[t]:**
   - If grid price ≤ gas marginal → buy from grid
   - If gas marginal < grid price → dispatch gas up to capacity, grid covers any remainder

Gas capacity is a free design variable (can be 0 — grid is the infinite backstop).

### System LCOE Formulas

**Microgrid sLCOE:**
```
sLCOE = ( Solar CAPEX×FCR + Solar O&M
        + BESS CAPEX×FCR  + BESS O&M
        + Gas CAPEX×FCR   + Gas O&M  + Gas Fuel + Gas Var O&M
        ) / Annual Demand MWh
```

**Grid-Connected sLCOE:**
```
sLCOE = ( Solar CAPEX×FCR + Solar O&M
        + BESS CAPEX×FCR  + BESS O&M
        + Gas CAPEX×FCR   + Gas O&M  + Gas Fuel + Gas Var O&M
        + Grid Interconnect CAPEX×FCR + Grid Interconnect O&M
        + Σ max(0, grid_price[t]) × grid_import[t]
        ) / Annual Demand MWh
```

FCR = WACC × (1+WACC)ⁿ / ((1+WACC)ⁿ − 1)

### Key Assumptions

| Parameter | Value |
|-----------|-------|
| WACC | 7% |
| Project life | 25 years |
| PUE | 1.35 |
| BESS round-trip efficiency | 85% |
| Solar DC:AC ratio | 1.3, fixed-tilt |
| Gas heat rate | 9.0 MMBtu/MWh (RICE) |
| Grid interconnect | $100/kW capex, $2/kW-yr O&M |
| Costs | Unsubsidised (pre-IRA ITC), real 2024 USD |

---

## Supported Markets

| ISO | Location | Gas ($/MMBtu) | CAPEX Mult. | 2024 DA Price |
|-----|----------|--------------|-------------|---------------|
| ERCOT | Dallas-Fort Worth, TX | $2.50 | 1.00 | $28.90/MWh (real) |
| PJM | Pittsburgh, PA | $3.20 | 1.05 | $27.36/MWh (real) |
| MISO | Indianapolis, IN | $3.00 | 1.03 | $45.00/MWh (synthetic) |
| CAISO | Fresno, CA | $5.00 | 1.15 | $37.94/MWh (real) |
| SPP | Oklahoma City, OK | $2.80 | 1.00 | $38.00/MWh (synthetic) |
| NYISO | Albany, NY | $5.50 | 1.18 | $36.08/MWh (real) |
| ISONE | Hartford, CT | $6.00 | 1.20 | $65.00/MWh (synthetic) |

*Real = fetched from ISO public API via gridstatus. Synthetic = calibrated profile where the API was too slow or incomplete.*

---

## Data Sources

| Data | Source |
|------|--------|
| Solar generation | NREL PVWatts V8 / NSRDB TMY |
| Technology costs | NREL ATB 2025 — Moderate scenario |
| LCOE benchmarks | Lazard LCOE+ 18.0 |
| Regional gas prices | EIA 2026 Short-Term Energy Outlook |
| Hourly temperature | Open-Meteo ERA5 reanalysis, 2024 |
| Day-ahead LMP prices | ISO public APIs via gridstatus 0.21 (ERCOT, CAISO, PJM, NYISO); calibrated synthetic for MISO, ISONE, SPP |

---

## Repository Structure

```
datacenter_microgrid_sim/
├── models/
│   ├── demand_model.py        PUE(T) model; site load timeseries
│   ├── bess_model.py          BESS state equation; unit sizing
│   ├── gas_model.py           RICE dispatch; fuel cost
│   ├── dispatcher.py          8,784-hour dispatch loop (microgrid + grid-connected)
│   ├── grid_prices_model.py   Real DA LMP fetcher (gridstatus) with synthetic fallback
│   ├── iso_registry.py        ISO/RTO lookup; sLCOE + grid sLCOE cost functions
│   ├── pipeline.py            Microgrid pipeline orchestrator per ISO
│   └── pipeline_grid.py       Grid-connected pipeline orchestrator per ISO
│
├── scripts/                   Standalone analysis scripts (exploratory use)
│
├── config/
│   ├── iso_registry.json      ISO/RTO parameters (location, gas price, CAPEX mult)
│   ├── technology_costs.json  CAPEX/OPEX baselines incl. grid interconnect
│   └── secrets.env            NREL API key (gitignored)
│
├── data/
│   ├── raw/                   Cached API responses (gitignored)
│   └── processed/             Per-ISO model outputs (demand, solar, BESS, reliability,
│                              sLCOE surface, grid prices, grid sLCOE surface)
│
├── dashboard.py               Streamlit app (Optimizer · Methodology · Team)
└── requirements.txt           Python dependencies
```
