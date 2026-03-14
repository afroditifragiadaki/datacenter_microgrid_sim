"""
ERCOT Datacenter Microgrid — Interactive Dashboard
===================================================
Run with:  streamlit run dashboard.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED = PROJECT_ROOT / "data" / "processed"
YEAR = 2024

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Microgrid Simulator — ERCOT",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme colours (consistent with matplotlib scripts)
# ---------------------------------------------------------------------------
C_SOLAR  = "#e5c07b"
C_BESS   = "#61afef"
C_GAS    = "#e06c75"
C_LOAD   = "#98c379"
C_CURT   = "#abb2bf"
C_GRID   = "#282c34"

# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_demand():
    return pd.read_csv(PROCESSED / f"ercot_demand_{YEAR}.csv",
                       index_col="datetime", parse_dates=True)

@st.cache_data
def load_solar():
    return pd.read_csv(PROCESSED / f"ercot_solar_{YEAR}.csv",
                       index_col="datetime", parse_dates=True)

@st.cache_data
def load_bess_params():
    return pd.read_csv(PROCESSED / f"ercot_bess_params_{YEAR}.csv",
                       index_col="datetime", parse_dates=True)

@st.cache_data
def load_dispatch(label: str):
    tag = label.replace(" ", "_").replace("=", "").replace(",", "")
    return pd.read_csv(PROCESSED / f"ercot_dispatch_{tag}_{YEAR}.csv",
                       index_col="datetime", parse_dates=True)

@st.cache_data
def load_dispatch_summary():
    return pd.read_csv(PROCESSED / "ercot_dispatch_summary_2024.csv",
                       index_col="config")

@st.cache_data
def load_reliability():
    return pd.read_csv(PROCESSED / f"ercot_reliability_surface_{YEAR}.csv")

@st.cache_data
def load_slcoe():
    return pd.read_csv(PROCESSED / f"ercot_slcoe_surface_{YEAR}.csv")

@st.cache_data
def run_custom_dispatch(S_mw: float, B_mwh: float, G_mw: float):
    """Run dispatcher live and cache by (S, B, G)."""
    from models.dispatcher import dispatch, load_timeseries
    ts = load_timeseries(PROCESSED, YEAR)
    results, summary = dispatch(S_mw, B_mwh, G_mw, ts)
    return results, summary


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("⚡ Microgrid Simulator")
st.sidebar.caption("ERCOT · Dallas-Fort Worth, TX · 2024")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Demand", "Solar", "Dispatch", "Reliability", "Economics"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "**Sources:**  NREL ATB 2025 · Lazard LCOE+ 18.0 · "
    "EIA 2026 STEO · Open-Meteo ERA5 · PVWatts V8"
)


# ===========================================================================
# PAGE: OVERVIEW
# ===========================================================================
if page == "Overview":
    st.title("⚡ ERCOT Datacenter Microgrid")
    st.markdown(
        "**Behind-the-meter islanded microgrid** serving a 50 MW IT load in "
        "Dallas-Fort Worth, TX. Solar → BESS → Gas dispatch, no grid imports. "
        "Simulation year: **2024** (8,784 hours)."
    )

    demand = load_demand()
    slcoe  = load_slcoe()
    rel    = load_reliability()

    # ---- KPI cards --------------------------------------------------------
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Annual Demand",    f"{demand['total_load_mw'].sum()/1e3:.0f} GWh")
    col2.metric("Peak Load",        f"{demand['total_load_mw'].max():.1f} MW")
    col3.metric("Avg PUE",          f"{demand['pue'].mean():.3f}")
    col4.metric("Min Gas Required", f"{rel['G_min_mw'].min():.1f} MW",
                delta="always ≈ 67 MW", delta_color="off")
    col5.metric("Baseline sLCOE",
                f"${slcoe.loc[slcoe['slcoe_per_mwh'].idxmin(), 'slcoe_per_mwh']:.2f}/MWh")

    st.markdown("---")

    # ---- System diagram text ----------------------------------------------
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.subheader("Dispatch Priority Stack")
        st.markdown("""
| Priority | Source | Logic |
|----------|--------|-------|
| 1 | ☀️ Solar PV | Zero marginal cost — always taken first |
| 2 | 🔋 BESS | Discharge to cover residual deficit |
| 3 | 🔥 Gas (RICE) | Gap-filler: `P_gas = min(G, unmet)` |
| 4 | ❌ Unserved | Reliability failure (outage) |

**No grid connection** — fully islanded BTM microgrid.
        """)

        st.subheader("Key Physical Findings")
        st.info(
            "**G_min ≈ 67 MW regardless of solar/BESS size.**  "
            "A 24/7 datacenter at 56–68 MW cannot be backed purely by solar + "
            "storage within practical ranges — gas must be sized to peak load. "
            "Solar & BESS reduce gas *runtime* (fuel cost), not gas *capacity*."
        )

    with col_r:
        st.subheader("Pipeline")
        st.markdown("""
```
01  Demand timeseries    Open-Meteo ERA5 → PUE(T) × 50 MW
02  Solar timeseries     PVWatts V8 / NSRDB TMY → solar_cf
03  BESS parameters      η_temp(T) efficiency modifier
04  Gas characterisation Load duration curve · sizing profile
05  Dispatch validation  5 preset (S, B, G) configurations
06  Reliability solver   G_min(S, B) surface — zero outages
07  sLCOE optimiser      Min-cost (S, B, G) configuration
```
        """)

        st.subheader("Simulation Parameters")
        st.markdown("""
| Parameter | Value |
|-----------|-------|
| IT load | 50 MW (constant) |
| Location | 32.78°N 96.80°W (DFW) |
| Solar tilt / azimuth | 32.78° / 180° (fixed) |
| BESS duration | 4 hours |
| η_ch = η_dis | 0.96 |
| Gas heat rate | 9.0 MMBtu/MWh |
| Gas price | $2.50/MMBtu |
| WACC | 7% · 25-year project |
        """)

    # ---- Annual load profile (monthly avg) --------------------------------
    st.markdown("---")
    st.subheader("Annual Load Profile")
    demand["month"] = demand.index.month
    monthly = demand.groupby("month")[["total_load_mw","pue","temp_c"]].mean().reset_index()
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly["month_name"] = [month_names[m-1] for m in monthly["month"]]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=monthly["month_name"], y=monthly["total_load_mw"],
                         name="Avg load (MW)", marker_color=C_LOAD, opacity=0.8))
    fig.add_trace(go.Scatter(x=monthly["month_name"], y=monthly["temp_c"],
                             name="Avg temp (°C)", line=dict(color=C_GAS, width=2),
                             mode="lines+markers"), secondary_y=True)
    fig.update_layout(height=320, margin=dict(t=20, b=20),
                      legend=dict(orientation="h", y=1.1),
                      plot_bgcolor="#1e2127", paper_bgcolor="#1e2127",
                      font_color="white")
    fig.update_yaxes(title_text="Avg Load (MW)", gridcolor="#3e4451")
    fig.update_yaxes(title_text="Avg Temp (°C)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE: DEMAND
# ===========================================================================
elif page == "Demand":
    st.title("Demand Analysis")
    st.caption("PUE-adjusted facility load — Dallas-Fort Worth, TX — 2024")

    demand = load_demand()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Annual Demand", f"{demand['total_load_mw'].sum()/1e3:.1f} GWh")
    col2.metric("Peak Load",     f"{demand['total_load_mw'].max():.2f} MW")
    col3.metric("Min Load",      f"{demand['total_load_mw'].min():.2f} MW")
    col4.metric("Avg PUE",       f"{demand['pue'].mean():.3f}")

    st.markdown("---")

    # Date range selector
    col_a, col_b = st.columns(2)
    view_start = col_a.date_input("From", value=pd.Timestamp("2024-01-01").date())
    view_end   = col_b.date_input("To",   value=pd.Timestamp("2024-01-14").date())
    mask = (demand.index.date >= view_start) & (demand.index.date <= view_end)
    sub  = demand[mask]

    # Load + temperature timeseries
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["Total Load (MW)", "Ambient Temperature (°C)"],
                        vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=sub.index, y=sub["total_load_mw"],
                             fill="tozeroy", fillcolor=f"rgba(152,195,121,0.25)",
                             line=dict(color=C_LOAD, width=1.2), name="Load"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=sub.index, y=sub["temp_c"],
                             line=dict(color=C_GAS, width=1.2), name="Temp °C"),
                  row=2, col=1)
    fig.update_layout(height=420, margin=dict(t=40, b=20),
                      plot_bgcolor="#1e2127", paper_bgcolor="#1e2127",
                      font_color="white", showlegend=False)
    fig.update_yaxes(gridcolor="#3e4451")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col_pue, col_ldc = st.columns(2)

    with col_pue:
        st.subheader("PUE Model")
        t_range = np.linspace(-10, 50, 300)
        T_KNOTS  = [-50, 10, 25, 35, 60]
        PUE_KNOTS = [1.12, 1.12, 1.20, 1.32, 1.45]
        pue_interp = np.interp(t_range, T_KNOTS, PUE_KNOTS)
        fig_pue = go.Figure()
        fig_pue.add_trace(go.Scatter(x=t_range, y=pue_interp,
                                     line=dict(color=C_SOLAR, width=2.5),
                                     name="PUE(T)"))
        fig_pue.add_trace(go.Scatter(x=T_KNOTS[1:-1], y=PUE_KNOTS[1:-1],
                                     mode="markers", marker=dict(color="white", size=8),
                                     name="Knots"))
        fig_pue.add_vline(x=demand["temp_c"].mean(), line_dash="dash",
                          line_color=C_BESS, annotation_text="Annual avg temp")
        fig_pue.update_layout(
            height=300, margin=dict(t=20, b=20),
            xaxis_title="Ambient Temperature (°C)", yaxis_title="PUE",
            plot_bgcolor="#1e2127", paper_bgcolor="#1e2127", font_color="white",
            yaxis=dict(gridcolor="#3e4451"), xaxis=dict(gridcolor="#3e4451"),
        )
        st.plotly_chart(fig_pue, use_container_width=True)

    with col_ldc:
        st.subheader("Load Duration Curve")
        load_sorted = np.sort(demand["total_load_mw"].values)[::-1]
        pct = np.linspace(0, 100, len(load_sorted))
        fig_ldc = go.Figure()
        fig_ldc.add_trace(go.Scatter(x=pct, y=load_sorted,
                                     fill="tozeroy",
                                     fillcolor=f"rgba(152,195,121,0.2)",
                                     line=dict(color=C_LOAD, width=1.5),
                                     name="Load (sorted)"))
        fig_ldc.update_layout(
            height=300, margin=dict(t=20, b=20),
            xaxis_title="Duration (% of hours)", yaxis_title="Load (MW)",
            plot_bgcolor="#1e2127", paper_bgcolor="#1e2127", font_color="white",
            yaxis=dict(gridcolor="#3e4451"), xaxis=dict(gridcolor="#3e4451"),
        )
        st.plotly_chart(fig_ldc, use_container_width=True)

    # Monthly heatmap
    st.subheader("Hourly Load Heatmap (MW)")
    demand["hour"]  = demand.index.hour
    demand["month"] = demand.index.month
    pivot = demand.pivot_table(values="total_load_mw",
                               index="hour", columns="month", aggfunc="mean")
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    fig_heat = go.Figure(go.Heatmap(
        z=pivot.values,
        x=month_labels,
        y=[f"{h:02d}:00" for h in range(24)],
        colorscale=[[0,"#98c379"],[0.5,"#e5c07b"],[1,"#e06c75"]],
        colorbar=dict(title="MW"),
    ))
    fig_heat.update_layout(
        height=380, margin=dict(t=20, b=20),
        xaxis_title="Month", yaxis_title="Hour of Day",
        plot_bgcolor="#1e2127", paper_bgcolor="#1e2127", font_color="white",
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# ===========================================================================
# PAGE: SOLAR
# ===========================================================================
elif page == "Solar":
    st.title("Solar PV Analysis")
    st.caption("PVWatts V8 · NSRDB TMY · Fixed-tilt 32.78° · DFW, TX")

    solar = load_solar()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Annual CF",    f"{solar['solar_cf'].mean()*100:.2f}%")
    col2.metric("Peak CF",      f"{solar['solar_cf'].max()*100:.1f}%")
    col3.metric("Daylight hrs", f"{(solar['solar_cf']>0.01).sum():,}")
    col4.metric("Annual GHI",   f"{solar['poa'].sum()/1e3:.0f} kWh/m²" if "poa" in solar.columns else "—")

    st.markdown("---")

    # Date range
    col_a, col_b = st.columns(2)
    view_start = col_a.date_input("From", value=pd.Timestamp("2024-07-01").date())
    view_end   = col_b.date_input("To",   value=pd.Timestamp("2024-07-14").date())
    mask = (solar.index.date >= view_start) & (solar.index.date <= view_end)
    sub  = solar[mask]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sub.index, y=sub["solar_cf"] * 100,
                             fill="tozeroy", fillcolor=f"rgba(229,192,123,0.25)",
                             line=dict(color=C_SOLAR, width=1.2), name="CF (%)"))
    fig.update_layout(
        height=280, margin=dict(t=20, b=20),
        yaxis_title="Solar Capacity Factor (%)",
        plot_bgcolor="#1e2127", paper_bgcolor="#1e2127", font_color="white",
        yaxis=dict(gridcolor="#3e4451"), xaxis=dict(gridcolor="#3e4451"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col_monthly, col_dur = st.columns(2)

    with col_monthly:
        st.subheader("Monthly Average CF")
        solar["month"] = solar.index.month
        monthly_cf = solar.groupby("month")["solar_cf"].mean() * 100
        month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        fig_m = go.Figure(go.Bar(
            x=month_names, y=monthly_cf.values,
            marker_color=C_SOLAR, opacity=0.85,
        ))
        fig_m.add_hline(y=solar["solar_cf"].mean()*100, line_dash="dash",
                        line_color="white", annotation_text="Annual avg")
        fig_m.update_layout(
            height=300, margin=dict(t=20, b=20),
            yaxis_title="Capacity Factor (%)",
            plot_bgcolor="#1e2127", paper_bgcolor="#1e2127", font_color="white",
            yaxis=dict(gridcolor="#3e4451"),
        )
        st.plotly_chart(fig_m, use_container_width=True)

    with col_dur:
        st.subheader("Solar CF Duration Curve")
        cf_sorted = np.sort(solar["solar_cf"].values)[::-1]
        pct = np.linspace(0, 100, len(cf_sorted))
        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(
            x=pct, y=cf_sorted * 100,
            fill="tozeroy", fillcolor=f"rgba(229,192,123,0.2)",
            line=dict(color=C_SOLAR, width=1.5),
        ))
        fig_d.update_layout(
            height=300, margin=dict(t=20, b=20),
            xaxis_title="Duration (% of hours)", yaxis_title="CF (%)",
            plot_bgcolor="#1e2127", paper_bgcolor="#1e2127", font_color="white",
            yaxis=dict(gridcolor="#3e4451"), xaxis=dict(gridcolor="#3e4451"),
        )
        st.plotly_chart(fig_d, use_container_width=True)

    # Hourly CF heatmap
    st.subheader("Hourly Solar CF Heatmap")
    solar["hour"]  = solar.index.hour
    solar["month"] = solar.index.month
    pivot_s = solar.pivot_table(values="solar_cf", index="hour", columns="month", aggfunc="mean")
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    fig_sh = go.Figure(go.Heatmap(
        z=pivot_s.values * 100,
        x=month_labels,
        y=[f"{h:02d}:00" for h in range(24)],
        colorscale=[[0,"#1e2127"],[0.4,"#e5c07b"],[1,"#ffffff"]],
        colorbar=dict(title="CF (%)"),
    ))
    fig_sh.update_layout(
        height=380, margin=dict(t=20, b=20),
        xaxis_title="Month", yaxis_title="Hour of Day",
        plot_bgcolor="#1e2127", paper_bgcolor="#1e2127", font_color="white",
    )
    st.plotly_chart(fig_sh, use_container_width=True)


# ===========================================================================
# PAGE: DISPATCH
# ===========================================================================
elif page == "Dispatch":
    st.title("Hourly Dispatch Simulation")

    tab_preset, tab_custom = st.tabs(["Preset Configurations", "Custom Run"])

    # ---- Preset tab -------------------------------------------------------
    with tab_preset:
        summary = load_dispatch_summary()

        # KPI comparison table
        st.subheader("Configuration Comparison")
        display_cols = {
            "S_mw": "Solar (MW)", "B_mwh": "Battery (MWh)", "G_mw": "Gas (MW)",
            "solar_share_pct": "Solar (%)", "bess_share_pct": "BESS (%)",
            "gas_share_pct": "Gas (%)", "renewable_share_pct": "Renewable (%)",
            "annual_fuel_cost_usd": "Fuel Cost ($)", "annual_co2_t": "CO2 (t)",
            "unserved_mwh": "Unserved (MWh)",
        }
        disp = summary[[c for c in display_cols if c in summary.columns]].rename(columns=display_cols)
        st.dataframe(disp.style.format({
            "Fuel Cost ($)":   "${:,.0f}",
            "CO2 (t)":         "{:,.0f}",
            "Unserved (MWh)":  "{:.1f}",
            "Solar (%)":       "{:.1f}%",
            "BESS (%)":        "{:.1f}%",
            "Gas (%)":         "{:.1f}%",
            "Renewable (%)":   "{:.1f}%",
        }), use_container_width=True)

        # Energy mix stacked bar
        configs    = list(summary.index)
        colours_5  = [C_CURT, C_SOLAR, C_BESS, C_GAS]
        fig_mix = go.Figure()
        fig_mix.add_trace(go.Bar(name="Solar used", x=configs,
                                 y=summary.get("solar_used_mwh", pd.Series([0]*len(configs))).values / 1e3,
                                 marker_color=C_SOLAR))
        fig_mix.add_trace(go.Bar(name="BESS discharge", x=configs,
                                 y=summary.get("bess_discharge_mwh", pd.Series([0]*len(configs))).values / 1e3,
                                 marker_color=C_BESS))
        fig_mix.add_trace(go.Bar(name="Gas", x=configs,
                                 y=summary["gas_gen_mwh"].values / 1e3,
                                 marker_color=C_GAS))
        fig_mix.add_hline(y=summary["total_demand_mwh"].mean() / 1e3,
                          line_dash="dash", line_color=C_CURT,
                          annotation_text="Annual demand")
        fig_mix.update_layout(
            barmode="stack", height=360, margin=dict(t=20, b=20),
            yaxis_title="Annual Energy (GWh)",
            legend=dict(orientation="h", y=1.08),
            plot_bgcolor="#1e2127", paper_bgcolor="#1e2127", font_color="white",
            yaxis=dict(gridcolor="#3e4451"),
        )
        st.plotly_chart(fig_mix, use_container_width=True)

        # Config selector for detail view
        st.markdown("---")
        st.subheader("Detail View")
        selected = st.selectbox("Select configuration", configs)
        r = load_dispatch(selected)

        col_a, col_b = st.columns(2)
        view_start = col_a.date_input("From", value=pd.Timestamp("2024-07-15").date(), key="ps")
        view_end   = col_b.date_input("To",   value=pd.Timestamp("2024-07-22").date(), key="pe")
        mask = (r.index.date >= view_start) & (r.index.date <= view_end)
        sub  = r[mask]

        fig_week = go.Figure()
        fig_week.add_trace(go.Scatter(x=sub.index,
                                      y=sub["solar_gen_mw"] - sub["curtailed_mw"],
                                      stackgroup="one", name="Solar",
                                      fillcolor=f"rgba(229,192,123,0.75)",
                                      line=dict(width=0)))
        fig_week.add_trace(go.Scatter(x=sub.index, y=sub["bess_discharge_mw"],
                                      stackgroup="one", name="BESS discharge",
                                      fillcolor=f"rgba(97,175,239,0.75)",
                                      line=dict(width=0)))
        fig_week.add_trace(go.Scatter(x=sub.index, y=sub["gas_gen_mw"],
                                      stackgroup="one", name="Gas",
                                      fillcolor=f"rgba(224,108,117,0.75)",
                                      line=dict(width=0)))
        fig_week.add_trace(go.Scatter(x=sub.index, y=sub["load_mw"],
                                      name="Load", line=dict(color="white", width=1.5)))
        fig_week.update_layout(
            height=360, margin=dict(t=20, b=20),
            yaxis_title="Power (MW)",
            legend=dict(orientation="h", y=1.08),
            plot_bgcolor="#1e2127", paper_bgcolor="#1e2127", font_color="white",
            yaxis=dict(gridcolor="#3e4451"),
        )
        st.plotly_chart(fig_week, use_container_width=True)

    # ---- Custom tab -------------------------------------------------------
    with tab_custom:
        st.subheader("Configure and Run")
        st.markdown("Set capacities, hit **Run Dispatch**, and see the live hourly simulation.")

        col1, col2, col3 = st.columns(3)
        S = col1.slider("Solar capacity S (MW DC)", 0, 300, 150, step=25)
        B = col2.slider("Battery capacity B (MWh)", 0, 1000, 400, step=50)
        G = col3.slider("Gas capacity G (MW)", 10, 100, 68, step=1)

        if st.button("Run Dispatch ▶", type="primary"):
            with st.spinner("Running 8,784-hour simulation..."):
                results, summary = run_custom_dispatch(float(S), float(B), float(G))

            # KPI row
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Solar share",     f"{summary['solar_share_pct']:.1f}%")
            c2.metric("BESS share",      f"{summary['bess_share_pct']:.1f}%")
            c3.metric("Gas share",       f"{summary['gas_share_pct']:.1f}%")
            c4.metric("Unserved energy", f"{summary['unserved_mwh']:.1f} MWh",
                      delta_color="inverse")
            c5.metric("Annual fuel cost",f"${summary['annual_fuel_cost_usd']/1e6:.2f}M")

            # Energy mix donut
            col_donut, col_ts = st.columns([1, 2])
            with col_donut:
                labels = ["Solar", "BESS", "Gas"]
                values = [summary["solar_used_mwh"],
                          summary["bess_discharge_mwh"],
                          summary["gas_gen_mwh"]]
                fig_d = go.Figure(go.Pie(
                    labels=labels, values=values,
                    hole=0.55,
                    marker_colors=[C_SOLAR, C_BESS, C_GAS],
                ))
                fig_d.update_layout(
                    height=300, margin=dict(t=20, b=20, l=0, r=0),
                    showlegend=True,
                    legend=dict(orientation="h", y=-0.1),
                    plot_bgcolor="#1e2127", paper_bgcolor="#1e2127", font_color="white",
                    annotations=[dict(text=f"{summary['renewable_share_pct']:.0f}%<br>renewable",
                                      x=0.5, y=0.5, font_size=13, showarrow=False,
                                      font_color="white")]
                )
                st.plotly_chart(fig_d, use_container_width=True)

            with col_ts:
                # Sample summer week
                sw = results.loc["2024-07-15":"2024-07-22"]
                fig_s = go.Figure()
                fig_s.add_trace(go.Scatter(x=sw.index,
                                           y=sw["solar_gen_mw"] - sw["curtailed_mw"],
                                           stackgroup="one", name="Solar",
                                           fillcolor="rgba(229,192,123,0.75)",
                                           line=dict(width=0)))
                fig_s.add_trace(go.Scatter(x=sw.index, y=sw["bess_discharge_mw"],
                                           stackgroup="one", name="BESS",
                                           fillcolor="rgba(97,175,239,0.75)",
                                           line=dict(width=0)))
                fig_s.add_trace(go.Scatter(x=sw.index, y=sw["gas_gen_mw"],
                                           stackgroup="one", name="Gas",
                                           fillcolor="rgba(224,108,117,0.75)",
                                           line=dict(width=0)))
                fig_s.add_trace(go.Scatter(x=sw.index, y=sw["load_mw"],
                                           name="Load", line=dict(color="white", width=1.5)))
                fig_s.update_layout(
                    height=300, margin=dict(t=30, b=20),
                    title="Summer Peak Week (Jul 15-22)",
                    yaxis_title="Power (MW)",
                    legend=dict(orientation="h", y=1.12),
                    plot_bgcolor="#1e2127", paper_bgcolor="#1e2127", font_color="white",
                    yaxis=dict(gridcolor="#3e4451"),
                )
                st.plotly_chart(fig_s, use_container_width=True)

            # BESS SoC
            if B > 0:
                fig_soc = go.Figure()
                fig_soc.add_trace(go.Scatter(x=results.index, y=results["soc_mwh"],
                                             line=dict(color=C_BESS, width=0.6),
                                             fill="tozeroy",
                                             fillcolor="rgba(97,175,239,0.15)"))
                fig_soc.add_hline(y=B * 0.20, line_dash="dash", line_color=C_GAS,
                                  annotation_text="SoC min (20%)")
                fig_soc.add_hline(y=B, line_dash="dash", line_color=C_LOAD,
                                  annotation_text="SoC max (100%)")
                fig_soc.update_layout(
                    height=250, margin=dict(t=30, b=20),
                    title="BESS State-of-Charge (full year)",
                    yaxis_title="SoC (MWh)",
                    plot_bgcolor="#1e2127", paper_bgcolor="#1e2127", font_color="white",
                    yaxis=dict(gridcolor="#3e4451"),
                )
                st.plotly_chart(fig_soc, use_container_width=True)
        else:
            st.info("Set sliders above and click **Run Dispatch** to simulate.")


# ===========================================================================
# PAGE: RELIABILITY
# ===========================================================================
elif page == "Reliability":
    st.title("Reliability Surface")
    st.caption("G_min(S, B) — minimum gas capacity for zero unserved energy at all 8,784 hours")

    rel = load_reliability()

    # Summary metrics
    base = rel.loc[(rel.S_mw == 0) & (rel.B_mwh == 0), "G_min_mw"].values[0]
    min_g = rel["G_min_mw"].min()
    min_g_row = rel.loc[rel["G_min_mw"].idxmin()]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Baseline G_min (S=0, B=0)",  f"{base:.1f} MW")
    col2.metric("Min achievable G_min",         f"{min_g:.1f} MW",
                delta=f"{min_g - base:.1f} MW", delta_color="inverse")
    col3.metric("At S",  f"{min_g_row['S_mw']:.0f} MW")
    col4.metric("At B",  f"{min_g_row['B_mwh']:.0f} MWh")

    st.info(
        f"G_min ranges only {base - min_g:.1f} MW across the entire search space — "
        "solar and battery cannot materially reduce gas capacity requirements for a "
        "24/7 islanded load. Gas must always be sized close to peak load."
    )

    st.markdown("---")

    # Interactive heatmap
    S_vals = sorted(rel["S_mw"].unique())
    B_vals = sorted(rel["B_mwh"].unique())
    pivot  = rel.pivot(index="B_mwh", columns="S_mw", values="G_min_mw")

    fig_h = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(int(s)) for s in S_vals],
        y=[str(int(b)) for b in B_vals],
        colorscale=[[0,"#98c379"],[0.5,"#e5c07b"],[1,"#e06c75"]],
        colorbar=dict(title="G_min (MW)"),
        hovertemplate="S=%{x} MW<br>B=%{y} MWh<br>G_min=%{z:.1f} MW<extra></extra>",
    ))
    fig_h.update_layout(
        height=420, margin=dict(t=20, b=20),
        xaxis_title="Solar Capacity S (MW DC)",
        yaxis_title="Battery Capacity B (MWh)",
        plot_bgcolor="#1e2127", paper_bgcolor="#1e2127", font_color="white",
    )
    st.plotly_chart(fig_h, use_container_width=True)

    st.markdown("---")
    col_vs_s, col_vs_b = st.columns(2)

    with col_vs_s:
        st.subheader("G_min vs Solar — by Battery Size")
        colours_B = ["#abb2bf","#61afef","#98c379","#e5c07b","#c678dd"]
        B_show    = [0, 200, 400, 600, 800]
        fig_s = go.Figure()
        for B_val, col in zip(B_show, colours_B):
            sub = rel[rel.B_mwh == B_val].sort_values("S_mw")
            fig_s.add_trace(go.Scatter(x=sub["S_mw"], y=sub["G_min_mw"],
                                       mode="lines+markers", name=f"B={B_val} MWh",
                                       line=dict(color=col, width=1.8)))
        fig_s.add_hline(y=base, line_dash="dash", line_color=C_GAS,
                        annotation_text=f"Peak load ({base:.1f} MW)")
        fig_s.update_layout(
            height=340, margin=dict(t=20, b=20),
            xaxis_title="Solar (MW DC)", yaxis_title="G_min (MW)",
            legend=dict(orientation="h", y=1.08),
            plot_bgcolor="#1e2127", paper_bgcolor="#1e2127", font_color="white",
            yaxis=dict(gridcolor="#3e4451"), xaxis=dict(gridcolor="#3e4451"),
        )
        st.plotly_chart(fig_s, use_container_width=True)

    with col_vs_b:
        st.subheader("Gas Hours/yr — by Solar Capacity")
        fig_gh = go.Figure()
        colours_S = ["#abb2bf","#e5c07b","#98c379","#61afef","#c678dd"]
        S_show    = [0, 75, 150, 225, 300]
        for S_val, col in zip(S_show, colours_S):
            sub = rel[rel.S_mw == S_val].sort_values("B_mwh")
            fig_gh.add_trace(go.Scatter(x=sub["B_mwh"], y=sub["gas_hours_yr"],
                                        mode="lines+markers", name=f"S={S_val} MW",
                                        line=dict(color=col, width=1.8)))
        fig_gh.add_hline(y=8784, line_dash="dash", line_color=C_GAS,
                         annotation_text="Gas-only (8784 hrs)")
        fig_gh.update_layout(
            height=340, margin=dict(t=20, b=20),
            xaxis_title="Battery (MWh)", yaxis_title="Gas online (hrs/yr)",
            legend=dict(orientation="h", y=1.08),
            plot_bgcolor="#1e2127", paper_bgcolor="#1e2127", font_color="white",
            yaxis=dict(gridcolor="#3e4451"), xaxis=dict(gridcolor="#3e4451"),
        )
        st.plotly_chart(fig_gh, use_container_width=True)


# ===========================================================================
# PAGE: ECONOMICS
# ===========================================================================
elif page == "Economics":
    st.title("System LCOE Optimisation")
    st.caption(
        "Unsubsidized sLCOE ($/MWh) · NREL ATB 2025 capex · "
        "Lazard LCOE+ 18.0 benchmarks · EIA 2026 STEO fuel · 7% WACC · 25 yr"
    )

    slcoe = load_slcoe()
    demand_mwh = slcoe["demand_mwh_yr"].iloc[0]

    # Optimum
    idx_opt   = slcoe["slcoe_per_mwh"].idxmin()
    opt        = slcoe.loc[idx_opt]
    base_row   = slcoe.loc[(slcoe.S_mw == 0) & (slcoe.B_mwh == 0)].iloc[0]
    savings_pct = (base_row["slcoe_per_mwh"] - opt["slcoe_per_mwh"]) / base_row["slcoe_per_mwh"] * 100

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Gas-only sLCOE", f"${base_row['slcoe_per_mwh']:.2f}/MWh")
    col2.metric("Optimum sLCOE",  f"${opt['slcoe_per_mwh']:.2f}/MWh",
                delta=f"{savings_pct:.1f}%", delta_color="inverse")
    col3.metric("Opt. Solar",     f"{opt['S_mw']:.0f} MW DC")
    col4.metric("Opt. Battery",   f"{opt['B_mwh']:.0f} MWh")
    col5.metric("Opt. Gas",       f"{opt['G_min_mw']:.1f} MW")

    st.info(
        "Under **unsubsidized** assumptions, gas-only minimises sLCOE. Solar LCOE "
        f"at 17% CF ≈ $65/MWh > gas variable cost $29.50/MWh. "
        "Adding ITC (30%), carbon pricing, or single-axis tracking changes the result."
    )

    st.markdown("---")

    # ---- sLCOE heatmap ---------------------------------------------------
    st.subheader("sLCOE Surface ($/MWh)")
    S_vals = sorted(slcoe["S_mw"].unique())
    B_vals = sorted(slcoe["B_mwh"].unique())
    pivot_s = slcoe.pivot(index="B_mwh", columns="S_mw", values="slcoe_per_mwh")

    fig_h = go.Figure(go.Heatmap(
        z=pivot_s.values,
        x=[str(int(s)) for s in S_vals],
        y=[str(int(b)) for b in B_vals],
        colorscale=[[0,"#98c379"],[0.5,"#e5c07b"],[1,"#e06c75"]],
        colorbar=dict(title="$/MWh"),
        hovertemplate="S=%{x} MW<br>B=%{y} MWh<br>sLCOE=$%{z:.2f}<extra></extra>",
    ))
    # Mark optimum
    fig_h.add_trace(go.Scatter(
        x=[str(int(opt["S_mw"]))], y=[str(int(opt["B_mwh"]))],
        mode="markers", marker=dict(symbol="star", size=20, color="white"),
        name="Optimum", showlegend=True,
    ))
    fig_h.update_layout(
        height=420, margin=dict(t=20, b=20),
        xaxis_title="Solar Capacity S (MW DC)",
        yaxis_title="Battery Capacity B (MWh)",
        plot_bgcolor="#1e2127", paper_bgcolor="#1e2127", font_color="white",
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(fig_h, use_container_width=True)

    st.markdown("---")
    col_curves, col_breakdown = st.columns([3, 2])

    with col_curves:
        st.subheader("sLCOE vs Solar — by Battery Size")
        colours_B = ["#abb2bf","#61afef","#98c379","#e5c07b","#c678dd"]
        B_show    = [0, 200, 400, 600, 800]
        fig_c = go.Figure()
        for B_val, col in zip(B_show, colours_B):
            sub = slcoe[slcoe.B_mwh == B_val].sort_values("S_mw")
            fig_c.add_trace(go.Scatter(x=sub["S_mw"], y=sub["slcoe_per_mwh"],
                                       mode="lines+markers", name=f"B={B_val} MWh",
                                       line=dict(color=col, width=1.8)))
        fig_c.add_hline(y=base_row["slcoe_per_mwh"], line_dash="dash",
                        line_color=C_GAS,
                        annotation_text=f"Gas-only ${base_row['slcoe_per_mwh']:.2f}")
        fig_c.update_layout(
            height=360, margin=dict(t=20, b=20),
            xaxis_title="Solar (MW DC)", yaxis_title="sLCOE ($/MWh)",
            legend=dict(orientation="h", y=1.1),
            plot_bgcolor="#1e2127", paper_bgcolor="#1e2127", font_color="white",
            yaxis=dict(gridcolor="#3e4451"), xaxis=dict(gridcolor="#3e4451"),
        )
        st.plotly_chart(fig_c, use_container_width=True)

    with col_breakdown:
        st.subheader("Cost Breakdown — Selected Config")
        # Let user pick a config from the surface
        sel_s = st.selectbox("Solar (MW)", sorted(slcoe["S_mw"].unique()), index=6)
        sel_b = st.selectbox("Battery (MWh)", sorted(slcoe["B_mwh"].unique()), index=4)
        sel_row = slcoe.loc[(slcoe.S_mw == sel_s) & (slcoe.B_mwh == sel_b)].iloc[0]

        st.metric("sLCOE", f"${sel_row['slcoe_per_mwh']:.2f}/MWh")

        from models.lcoe_model import GAS_OPEX_VAR_PER_MWH
        from models.gas_model import HEAT_RATE_MMBTU_PER_MWH, GAS_PRICE_PER_MMBTU
        gas_var_total = (GAS_OPEX_VAR_PER_MWH + HEAT_RATE_MMBTU_PER_MWH * GAS_PRICE_PER_MMBTU) * sel_row["gas_gen_mwh"]
        gas_fixed     = sel_row["gas_cost_usd_yr"] - gas_var_total

        labels_p = ["Solar", "BESS", "Gas capex+O&M", "Gas fuel"]
        values_p = [
            sel_row["solar_cost_usd_yr"],
            sel_row["bess_cost_usd_yr"],
            gas_fixed,
            gas_var_total,
        ]
        colors_p = [C_SOLAR, C_BESS, "#c678dd", C_GAS]
        # Filter zero slices
        filtered = [(l, v, c) for l, v, c in zip(labels_p, values_p, colors_p) if v > 0]
        if filtered:
            lf, vf, cf = zip(*filtered)
            fig_p = go.Figure(go.Pie(
                labels=lf, values=vf, hole=0.5,
                marker_colors=cf,
            ))
            fig_p.update_layout(
                height=300, margin=dict(t=20, b=10, l=0, r=0),
                showlegend=True,
                legend=dict(orientation="v", x=1.0),
                plot_bgcolor="#1e2127", paper_bgcolor="#1e2127", font_color="white",
                annotations=[dict(
                    text=f"${sel_row['total_cost_usd_yr']/1e6:.1f}M/yr",
                    x=0.5, y=0.5, font_size=12, showarrow=False, font_color="white",
                )]
            )
            st.plotly_chart(fig_p, use_container_width=True)

    # ---- Sensitivity tornado (bar chart) ---------------------------------
    st.markdown("---")
    st.subheader("Sensitivity Analysis — Optimum Config (±20% on each parameter)")

    sens_data = {
        "Parameter":    ["Gas price", "Gas capex", "Gas var O&M", "Solar capex", "BESS capex", "Discount rate"],
        "Low (−20%)":   [40.92, 42.75, 44.02, 45.42, 45.42, 45.42],
        "Base":         [45.42, 45.42, 45.42, 45.42, 45.42, 45.42],
        "High (+20%)":  [49.92, 48.08, 46.82, 45.42, 45.42, 45.42],
    }
    sens_df = pd.DataFrame(sens_data).sort_values(
        by="Low (−20%)", key=lambda x: abs(x - 45.42)
    )
    base_val = 45.42

    fig_t = go.Figure()
    for _, row in sens_df.iterrows():
        lo = min(row["Low (−20%)"], row["High (+20%)"]) - base_val
        hi = max(row["Low (−20%)"], row["High (+20%)"]) - base_val
        fig_t.add_trace(go.Bar(
            y=[row["Parameter"]],
            x=[lo],
            base=[0],
            orientation="h",
            marker_color=C_LOAD if lo < 0 else C_GAS,
            showlegend=False, width=0.5,
        ))
        fig_t.add_trace(go.Bar(
            y=[row["Parameter"]],
            x=[hi],
            base=[0],
            orientation="h",
            marker_color=C_GAS if hi > 0 else C_LOAD,
            showlegend=False, width=0.5,
        ))
    fig_t.add_vline(x=0, line_color="white", line_width=1)
    fig_t.update_layout(
        barmode="overlay", height=300, margin=dict(t=20, b=20),
        xaxis_title=f"Change in sLCOE vs base ${base_val:.2f}/MWh ($/MWh)",
        plot_bgcolor="#1e2127", paper_bgcolor="#1e2127", font_color="white",
        xaxis=dict(gridcolor="#3e4451"),
    )
    st.plotly_chart(fig_t, use_container_width=True)

    st.caption(
        "Gas price is the dominant driver. Solar/BESS/discount rate have zero "
        "sensitivity at the current optimum (gas-only) since no solar/BESS is deployed."
    )
