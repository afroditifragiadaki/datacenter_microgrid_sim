# 50 MW Data Center Microgrid Simulation

A technical simulation framework designed to model the energy reliability and economic 
feasibility of a hybrid microgrid for high-density data centers in 2026.

## Project Objectives
* Reliability: Maintain 99.999% uptime for a baseline 50 MW IT load across 8,760 hours.
* Economic Optimization: Minimize the System Levelized Cost of Energy (sLCOE).
* Regional Analysis: Compare performance across multiple ISOs including PJM, ERCOT, and MISO.

## Technical Methodology
* Demand Side: IT load (50 MW) with dynamic Power Usage Effectiveness (PUE) 
  scaling between 1.12 and 1.45 based on ambient temperature.
* Supply Side:
    - Solar: Physics-based model using NSRDB irradiance and temperature data.
    - BESS: Hourly State of Charge (SOC) tracking with thermal efficiency factors.
    - Gas: Operational gap-filler logic for unmet demand.
    - Grid: Hourly imports based on regional Locational Marginal Prices (LMP).

## Repository Structure
* data/: Hourly datasets for solar (NSRDB) and grid prices (LMP).
* models/: Governing equations for solar output, battery SOC, and sLCOE.
* config/: Regional parameters for capacity fees, demand charges, and fuel costs.
* notebooks/: Analysis of 8,760-hour dispatch simulations.

## Installation and Usage
1. Clone the repository:
   git clone https://github.com/afroditifragiadaki/Microgrid_project.git
2. Create the environment:
   conda env create -f environment.yml
3. Activate the environment:
   conda activate microgrid_sim