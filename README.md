# Datacenter Microgrid Cost Explorer

**ELEN4510 Grid Modernization & Clean Tech — Columbia University**

An interactive optimization tool that models the optimal **Solar PV + BESS + Natural Gas**
microgrid for a behind-the-meter collocated datacenter. For each of the major US electricity
markets (ISO/RTOs), it finds the lowest-cost asset configuration that satisfies a
user-defined renewable energy share target.

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

Given a datacenter IT load (MW) and a minimum renewable energy share (%), the tool:

1. Searches across hundreds of Solar + BESS + Gas size combinations
2. Simulates hourly dispatch for each combination across 8,760 hours of the year
3. Filters out designs that don't meet the renewable floor
4. Ranks the feasible designs by System LCOE ($/MWh) and returns the cheapest option

This is repeated for every supported ISO/RTO market, enabling direct cost comparison
across regions with different solar resources, gas prices, and construction costs.

---

## Dashboard

Run locally with:

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

The dashboard has three tabs:

| Tab | Description |
|-----|-------------|
| **Optimizer** | Configure IT load and renewable floor → compare all markets → deep dive into any market's cost breakdown, energy mix, and gas price sensitivity |
| **Methodology** | Step-by-step pipeline flowchart, sLCOE formula, hourly dispatch logic, and linked data sources |
| **Team** | Project team and contact information |

Get a free NREL API key at [developer.nrel.gov](https://developer.nrel.gov/signup/) and add it to `config/secrets.env`:

```
NREL_API_KEY=your_key_here
NREL_EMAIL=you@example.com
```

---

## Methodology

### System LCOE (sLCOE)

All capital costs are annualised using a Fixed Charge Rate (FCR) derived from a
**7% WACC** over a **25-year project life** (FCR ≈ 8.58%). Annual O&M is added,
then gas fuel costs. The total is divided by annual site demand:

```
sLCOE = ( Solar CAPEX×FCR + Solar O&M
        + BESS CAPEX×FCR  + BESS O&M
        + Gas CAPEX×FCR   + Gas O&M
        + Gas Price × Heat Rate × Gas Generation ) / Annual Demand MWh
```

Costs are unsubsidised (pre-IRA Investment Tax Credit). All values in real 2024 USD.

### Hourly Dispatch Logic

Each of the 8,760 hours of the year is simulated with a simple priority stack:

1. **Solar first** — available solar output (PVWatts TMY profile) serves load directly
2. **BESS** — surplus solar charges the battery; deficits draw it down (10% SoC floor)
3. **Gas backup** — any remaining unmet load is covered by the gas generator

At year-end, renewable share = (solar MWh + BESS-served MWh) ÷ total demand MWh.

### Optimization

A grid search sweeps Solar MW, BESS MWh, and Gas MW. For each triplet the dispatch
simulation runs in full. Designs below the renewable floor are discarded; the
minimum-sLCOE survivor is reported as the constrained optimum.

### Key Assumptions

| Parameter | Value |
|-----------|-------|
| WACC | 7% |
| Project life | 25 years |
| PUE | 1.35 |
| BESS round-trip efficiency | 85% |
| Solar DC:AC ratio | 1.3, fixed-tilt |
| Gas heat rate | 9.0 MMBtu/MWh (RICE) |
| Costs | Unsubsidised, real 2024 USD |

---

## Supported Markets

| ISO | Location | Gas ($/MMBtu) | CAPEX Multiplier |
|-----|----------|--------------|-----------------|
| ERCOT | Dallas-Fort Worth, TX | $2.50 | 1.00 |
| PJM | Pittsburgh, PA | $3.20 | 1.05 |
| MISO | Indianapolis, IN | $3.00 | 1.03 |
| CAISO | Fresno, CA | $5.00 | 1.15 |
| SPP | Oklahoma City, OK | $2.80 | 1.00 |
| NYISO | Albany, NY | $5.50 | 1.18 |
| ISONE | Hartford, CT | $6.00 | 1.20 |

---

## Data Sources

| Data | Source | URL |
|------|--------|-----|
| Solar generation | NREL PVWatts V8 / NSRDB TMY | [pvwatts.nrel.gov](https://pvwatts.nrel.gov/) |
| Technology costs (Solar, BESS, Gas) | NREL ATB 2025 — Moderate scenario | [atb.nrel.gov](https://atb.nrel.gov/) |
| LCOE benchmarks | Lazard LCOE+ 18.0 | [lazard.com](https://www.lazard.com/research-insights/levelized-cost-of-energyplus/) |
| Regional gas prices | EIA 2026 Short-Term Energy Outlook | [eia.gov/outlooks/steo](https://www.eia.gov/outlooks/steo/) |
| Hourly temperature | Open-Meteo ERA5 reanalysis, 2024 | [open-meteo.com](https://open-meteo.com/) |

---

## Repository Structure

```
datacenter_microgrid_sim/
├── models/
│   ├── demand_model.py       PUE(T) model; site load timeseries
│   ├── solar_model.py        PVWatts capacity factor profiles
│   ├── bess_model.py         BESS state equation; unit sizing
│   ├── gas_model.py          RICE dispatch; fuel cost
│   ├── dispatcher.py         8,760-hour greedy dispatch loop
│   ├── lcoe_model.py         FCR; annualised cost functions; sLCOE
│   ├── iso_registry.py       ISO/RTO lookup table and regional parameters
│   └── pipeline.py           On-demand pipeline orchestrator per ISO
│
├── scripts/
│   ├── 01_build_demand.py    Weather fetch; PUE model; load timeseries
│   ├── 02_build_solar.py     PVWatts V8 API; solar capacity factor
│   ├── 03_build_bess.py      Temperature-adjusted efficiency timeseries
│   ├── 04_build_gas.py       Gas model characterisation
│   ├── 05_run_dispatch.py    Dispatch validation
│   ├── 06_solve_reliability.py  Minimum gas capacity surface
│   └── 07_optimize_slcoe.py  sLCOE surface and constrained optimum
│
├── config/
│   ├── iso_registry.json     ISO/RTO parameters
│   ├── technology_costs.json CAPEX/OPEX baselines
│   └── secrets.env           NREL API key (gitignored)
│
├── data/
│   ├── raw/                  Cached API responses (gitignored)
│   └── processed/            Model outputs per ISO
│
├── dashboard.py              Streamlit dashboard (Optimizer · Methodology · Team)
├── requirements.txt          Python dependencies
└── environment.yml           Conda environment (pinned)
```
