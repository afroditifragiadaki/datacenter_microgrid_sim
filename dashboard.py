"""
Microgrid Cost Explorer
Datacenter Energy Optimization Platform
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.iso_registry import get_iso, get_all_isos, get_costs
from models.pipeline import is_complete, run_pipeline

PROCESSED       = PROJECT_ROOT / "data" / "processed"
YEAR            = 2024
BASE_IT_LOAD_MW = 50.0   # load the pipeline was calibrated on
HEAT_RATE       = 9.0    # MMBtu/MWh (gas_model constant)
PROJECT_LIFE    = 25

# ── Palette ──────────────────────────────────────────────────────────────────

BG      = "#060912"
SURFACE = "#0d1220"
BORDER  = "#1a2035"
TEXT    = "#e2e8f0"
MUTED   = "#4a5568"
ACCENT  = "#3b82f6"
C_SOLAR = "#f59e0b"
C_BESS  = "#3b82f6"
C_GAS   = "#ef4444"

# ── Page config (must be first Streamlit call) ────────────────────────────────

st.set_page_config(
    page_title="Microgrid Cost Explorer",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── CSS ───────────────────────────────────────────────────────────────────────

def _css() -> None:
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,200..700&display=swap');

    html, body, * {{ font-family: 'Inter', sans-serif !important; }}

    /* ── Background & base ── */
    .stApp, [data-testid="stAppViewContainer"] {{
        background: {BG} !important;
        color: {TEXT} !important;
    }}
    [data-testid="stMainBlockContainer"], .main .block-container {{
        padding: 0 !important;
        max-width: 100% !important;
    }}

    /* ── Hide Streamlit chrome ── */
    header[data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stStatusWidget"],
    [data-testid="stSidebar"],
    .stDeployButton,
    footer {{ display: none !important; }}

    /* ── Inputs ── */
    .stNumberInput input {{
        background: {SURFACE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 2px !important;
        color: {TEXT} !important;
        font-size: 15px !important;
        padding: 10px 14px !important;
    }}
    .stNumberInput input:focus {{
        border-color: {ACCENT} !important;
        box-shadow: none !important;
        outline: none !important;
    }}

    /* ── Slider ── */
    [data-testid="stSlider"] [data-testid="stTickBarMin"],
    [data-testid="stSlider"] [data-testid="stTickBarMax"] {{
        color: {MUTED} !important;
        font-size: 11px !important;
    }}
    [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {{
        background: {ACCENT} !important;
    }}

    /* ── Buttons ── */
    .stButton > button {{
        border-radius: 2px !important;
        font-size: 11px !important;
        font-weight: 500 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        transition: opacity 0.15s !important;
        cursor: pointer !important;
    }}
    .stButton > button[kind="primary"] {{
        background: {ACCENT} !important;
        color: #fff !important;
        border: none !important;
        padding: 10px 28px !important;
    }}
    .stButton > button[kind="primary"]:hover {{ opacity: 0.82 !important; }}

    .stButton > button[kind="secondary"] {{
        background: transparent !important;
        color: {MUTED} !important;
        border: 1px solid {BORDER} !important;
        padding: 7px 16px !important;
    }}
    .stButton > button[kind="secondary"]:hover {{
        color: {TEXT} !important;
        border-color: {MUTED} !important;
    }}

    /* ── Form labels ── */
    label, .stNumberInput label, [data-testid="stSlider"] label {{
        font-size: 10px !important;
        font-weight: 500 !important;
        letter-spacing: 0.14em !important;
        text-transform: uppercase !important;
        color: {MUTED} !important;
    }}

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER} !important;
        border-radius: 2px !important;
    }}
    [data-testid="stDataFrame"] th {{
        background: {SURFACE} !important;
        color: {MUTED} !important;
        font-size: 10px !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        border-bottom: 1px solid {BORDER} !important;
    }}
    [data-testid="stDataFrame"] td {{
        font-size: 13px !important;
        color: {TEXT} !important;
        border-bottom: 1px solid {BORDER} !important;
    }}

    /* ── Spinner ── */
    [data-testid="stSpinner"] > div {{
        border-top-color: {ACCENT} !important;
    }}

    /* ── Alerts ── */
    [data-testid="stAlert"] {{ border-radius: 2px !important; }}

    /* ── Divider ── */
    hr {{ border-color: {BORDER} !important; opacity: 1 !important; }}

    /* ── Tooltip / help ── */
    [data-testid="stTooltipIcon"] {{ color: {MUTED} !important; }}

    /* ── Column gap normalise ── */
    [data-testid="column"] {{ gap: 0 !important; }}
    </style>
    """, unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────

def _init_state() -> None:
    for key, default in {
        "configured":    False,
        "it_load":       50.0,
        "ren_floor":     30,
        "selected_iso":  None,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ── Data helpers ──────────────────────────────────────────────────────────────

@st.cache_data
def _load_slcoe(iso_id: str) -> pd.DataFrame:
    p = PROCESSED / f"{iso_id.lower()}_slcoe_surface_{YEAR}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def _dispatch_at_opt(iso_id: str, S_mw: float, B_mwh: float, G_mw: float) -> dict:
    """Run hourly dispatch at a given (S, B, G) and return the summary dict."""
    from models.dispatcher import dispatch, load_timeseries
    ts = load_timeseries(PROCESSED, YEAR, iso_id=iso_id)
    _, summary = dispatch(S_mw, B_mwh, G_mw, ts)
    return summary


def _constrained_opt(iso_id: str, it_load_mw: float, ren_floor_frac: float):
    """
    Find the minimum-sLCOE row in the slcoe_surface that satisfies the
    renewable share constraint.

    Returns (opt_dict, scale_factor) where scale_factor = it_load / 50.
    Returns (None, scale) if no feasible configuration exists.
    """
    df = _load_slcoe(iso_id)
    if df.empty:
        return None, it_load_mw / BASE_IT_LOAD_MW

    scale = it_load_mw / BASE_IT_LOAD_MW
    df    = df.copy()
    df["ren_share"] = 1.0 - df["gas_gen_mwh"] / df["demand_mwh_yr"]

    feasible = df[df["ren_share"] >= ren_floor_frac]
    if feasible.empty:
        return None, scale

    best = feasible.loc[feasible["slcoe_per_mwh"].idxmin()].to_dict()
    return best, scale


def _gas_sensitivity(opt: dict, iso_cfg: dict) -> pd.DataFrame:
    """
    Three-scenario gas price sensitivity table.
    Only fuel cost varies; all other costs are fixed at the optimum.
    """
    base   = iso_cfg["gas_price_per_mmbtu"]
    g_gen  = opt["gas_gen_mwh"]
    demand = opt["demand_mwh_yr"]
    base_slcoe = opt["slcoe_per_mwh"]

    rows = []
    for label, price in [("Low (−20%)", base * 0.80),
                          ("Base",       base),
                          ("High (+40%)", base * 1.40)]:
        delta = (price - base) * HEAT_RATE * g_gen / demand
        new_slcoe = base_slcoe + delta
        rows.append({
            "Scenario":       label,
            "Gas ($/MMBtu)":  f"${price:.2f}",
            "sLCOE ($/MWh)":  f"${new_slcoe:.2f}",
            "Δ vs Base":      "—" if label == "Base" else f"{'+'if delta>0 else ''}{delta:.2f}",
        })
    return pd.DataFrame(rows).set_index("Scenario")


# ── UI primitives ─────────────────────────────────────────────────────────────

def _kpi_html(label: str, value: str, sub: str = "") -> str:
    sub_html = (
        f'<div style="font-size:11px;color:{MUTED};margin-top:5px;'
        f'line-height:1.4">{sub}</div>'
        if sub else ""
    )
    return f"""
    <div style="padding:20px 24px 20px 0; border-top:1px solid {BORDER};">
      <div style="font-size:9px;font-weight:600;letter-spacing:0.16em;
                  text-transform:uppercase;color:{MUTED};margin-bottom:9px">
        {label}
      </div>
      <div style="font-size:26px;font-weight:300;color:{TEXT};line-height:1;
                  letter-spacing:-0.01em">
        {value}
      </div>
      {sub_html}
    </div>
    """


def _section_label(text: str) -> None:
    st.markdown(
        f'<div style="font-size:9px;font-weight:600;letter-spacing:0.18em;'
        f'text-transform:uppercase;color:{MUTED};padding:28px 0 14px 0;'
        f'border-top:1px solid {BORDER}">{text}</div>',
        unsafe_allow_html=True,
    )


def _chart_defaults(fig: go.Figure, height: int = 220) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(t=10, b=10, l=0, r=4),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=TEXT, size=11),
        showlegend=False,
    )
    return fig


# ── PAGE: Configure ───────────────────────────────────────────────────────────

def _page_configure() -> None:
    _, col, _ = st.columns([1, 1.1, 1])

    with col:
        st.markdown(f"""
        <div style="padding:80px 0 56px 0">
          <div style="font-size:9px;font-weight:600;letter-spacing:0.22em;
                      text-transform:uppercase;color:{MUTED};margin-bottom:14px">
            Microgrid Cost Explorer
          </div>
          <div style="font-size:34px;font-weight:200;color:{TEXT};
                      line-height:1.18;margin-bottom:14px;letter-spacing:-0.02em">
            Datacenter Energy<br>Optimization Platform
          </div>
          <div style="font-size:13px;color:{MUTED};line-height:1.7;margin-bottom:52px">
            Model the optimal Solar · BESS · Gas microgrid for a collocated
            behind-the-meter facility. Compare system LCOE across all US
            ISO/RTO markets under user-defined ESG constraints.
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            f'<div style="height:1px;background:{BORDER};margin-bottom:36px"></div>',
            unsafe_allow_html=True,
        )

        it_load = st.number_input(
            "IT Load  (MW)",
            min_value=1.0,
            max_value=500.0,
            value=float(st.session_state.it_load),
            step=5.0,
            help="Rated IT power of the datacenter. Total facility load is higher due to PUE.",
        )

        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

        ren_floor = st.slider(
            "Minimum Renewable Share  (%)",
            min_value=0,
            max_value=100,
            value=int(st.session_state.ren_floor),
            step=5,
            help="Solar + BESS must supply at least this share of annual energy. "
                 "Gas covers the remaining gap.",
        )

        st.markdown(f"""
        <div style="font-size:11px;color:{MUTED};margin-top:8px;margin-bottom:44px;
                    line-height:1.6">
          Solar + BESS must supply ≥ <strong style="color:{TEXT}">{ren_floor}%</strong>
          of annual demand. The optimizer finds the lowest-cost (S, B, G) configuration
          that satisfies this ESG constraint across each market.
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            f'<div style="height:1px;background:{BORDER};margin-bottom:32px"></div>',
            unsafe_allow_html=True,
        )

        _, btn_col = st.columns([3, 2])
        with btn_col:
            go_btn = st.button("Analyze all markets →", type="primary",
                               use_container_width=True)

        if go_btn:
            st.session_state.it_load     = it_load
            st.session_state.ren_floor   = ren_floor
            st.session_state.configured  = True
            st.session_state.selected_iso = None
            st.rerun()

        st.markdown(f"""
        <div style="margin-top:72px;font-size:10px;color:{MUTED};
                    letter-spacing:0.05em;line-height:1.8">
          NREL ATB 2025 &nbsp;·&nbsp; Lazard LCOE+ 18.0 &nbsp;·&nbsp;
          EIA 2026 STEO &nbsp;·&nbsp; PVWatts V8 &nbsp;·&nbsp; Open-Meteo ERA5<br>
          Unsubsidised baseline · 2024 USD · 7% WACC · 25-year project life
        </div>
        """, unsafe_allow_html=True)


# ── PAGE: All Markets ─────────────────────────────────────────────────────────

def _page_markets() -> None:
    it_load   = st.session_state.it_load
    ren_floor = st.session_state.ren_floor
    ren_frac  = ren_floor / 100.0
    all_isos  = get_all_isos()

    _, main, _ = st.columns([0.035, 0.93, 0.035])

    with main:
        # ── Header ──
        top_l, top_r = st.columns([3, 1])
        with top_l:
            st.markdown(f"""
            <div style="padding:36px 0 0 0">
              <div style="font-size:9px;font-weight:600;letter-spacing:0.22em;
                          text-transform:uppercase;color:{MUTED};margin-bottom:6px">
                Microgrid Cost Explorer
              </div>
              <div style="font-size:24px;font-weight:300;color:{TEXT};letter-spacing:-0.01em">
                Market Comparison
              </div>
            </div>
            """, unsafe_allow_html=True)
        with top_r:
            st.markdown("<div style='padding-top:44px'>", unsafe_allow_html=True)
            if st.button("← Reconfigure", type="secondary"):
                st.session_state.configured   = False
                st.session_state.selected_iso = None
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="font-size:11px;color:{MUTED};padding-bottom:20px">
          {it_load:.0f} MW IT load &nbsp;·&nbsp;
          ≥{ren_floor}% renewable &nbsp;·&nbsp;
          Unsubsidised &nbsp;·&nbsp; {YEAR} &nbsp;·&nbsp;
          7% WACC &nbsp;·&nbsp; {PROJECT_LIFE}-year project life
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<div style="height:1px;background:{BORDER}"></div>',
                    unsafe_allow_html=True)

        # ── Build rows ──
        ready_rows, pending_rows = [], []

        for iso_id, cfg in all_isos.items():
            computed = is_complete(iso_id, YEAR)
            opt, scale = _constrained_opt(iso_id, it_load, ren_frac) if computed else (None, 1.0)

            row_meta = {"iso": iso_id, "cfg": cfg, "opt": opt, "scale": scale}

            if computed and opt:
                ready_rows.append({
                    **row_meta,
                    "Market":        iso_id,
                    "Location":      cfg["city"],
                    "sLCOE $/MWh":   round(opt["slcoe_per_mwh"], 2),
                    "Renewable":     f"{opt['ren_share']*100:.0f}%",
                    "Solar  MW":     f"{opt['S_mw'] * scale:.0f}",
                    "BESS  MWh":     f"{opt['B_mwh'] * scale:.0f}",
                    "Gas  MW":       f"{opt['G_min_mw'] * scale:.0f}",
                })
            elif computed and not opt:
                # Data exists but no feasible point at this constraint level
                ready_rows.append({
                    **row_meta,
                    "Market":        iso_id,
                    "Location":      cfg["city"],
                    "sLCOE $/MWh":   None,
                    "Renewable":     "Infeasible",
                    "Solar  MW":     "—",
                    "BESS  MWh":     "—",
                    "Gas  MW":       "—",
                })
            else:
                pending_rows.append(row_meta)

        # Sort available by sLCOE (infeasible rows at bottom)
        ready_rows.sort(key=lambda r: (
            r["sLCOE $/MWh"] is None, r["sLCOE $/MWh"] or 9999
        ))

        # ── Available markets table ──
        _section_label("Available Markets — Ranked by System LCOE")

        if ready_rows:
            display_cols = ["Market", "Location", "sLCOE $/MWh",
                            "Renewable", "Solar  MW", "BESS  MWh", "Gas  MW"]
            df_disp = pd.DataFrame(
                [{c: r[c] for c in display_cols} for r in ready_rows]
            )

            event = st.dataframe(
                df_disp,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                column_config={
                    "sLCOE $/MWh": st.column_config.NumberColumn(format="$%.2f"),
                },
                height=min(80 + len(ready_rows) * 35, 360),
            )

            st.markdown(
                f'<div style="font-size:10px;color:{MUTED};margin-top:6px;'
                f'letter-spacing:0.03em">Select a row to explore the optimal '
                f'configuration in detail.</div>',
                unsafe_allow_html=True,
            )

            if event.selection and event.selection.rows:
                sel = ready_rows[event.selection.rows[0]]
                if sel["opt"] is not None:
                    st.session_state.selected_iso = sel["iso"]
                    st.rerun()
        else:
            st.info("No markets have been computed yet. Run a pipeline below.")

        # ── Pending markets ──
        if pending_rows:
            _section_label("Markets Pending — Pipeline Not Yet Run")

            n = len(pending_rows)
            p_cols = st.columns(min(n, 4))

            for i, row in enumerate(pending_rows):
                cfg = row["cfg"]
                with p_cols[i % 4]:
                    st.markdown(f"""
                    <div style="border:1px solid {BORDER};padding:18px 20px 16px;
                                margin-bottom:12px">
                      <div style="font-size:9px;font-weight:600;letter-spacing:0.14em;
                                  text-transform:uppercase;color:{MUTED};margin-bottom:5px">
                        {row['iso']}
                      </div>
                      <div style="font-size:14px;color:{TEXT};margin-bottom:14px">
                        {cfg['city']}
                      </div>
                      <div style="font-size:10px;color:{MUTED};line-height:1.7">
                        Gas &nbsp;${cfg['gas_price_per_mmbtu']:.2f}/MMBtu<br>
                        CAPEX &nbsp;{cfg['capex_multiplier']:.2f}×
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    iso_key = row["iso"]
                    if st.button(f"Run {iso_key}", key=f"run_{iso_key}",
                                 use_container_width=True):
                        _log = st.empty()
                        _msgs: list[str] = []

                        def _make_log(area, msgs):
                            def _fn(m):
                                msgs.append(m)
                                area.code("\n".join(msgs[-16:]))
                            return _fn

                        with st.spinner(f"Running {iso_key} pipeline…"):
                            try:
                                run_pipeline(iso_key, YEAR,
                                             log=_make_log(_log, _msgs))
                                st.cache_data.clear()
                                st.rerun()
                            except Exception as exc:
                                st.error(str(exc))

        # ── Footer ──
        st.markdown(f"""
        <div style="font-size:10px;color:{MUTED};letter-spacing:0.04em;
                    border-top:1px solid {BORDER};margin-top:48px;padding:24px 0">
          Sources: NREL ATB 2025 &nbsp;·&nbsp; Lazard LCOE+ 18.0 &nbsp;·&nbsp;
          EIA 2026 STEO &nbsp;·&nbsp; PVWatts V8 &nbsp;·&nbsp; Open-Meteo ERA5 &nbsp;·&nbsp;
          Unsubsidised baseline (pre-IRA ITC). All costs in 2024 USD.
        </div>
        """, unsafe_allow_html=True)


# ── PAGE: Deep Dive ───────────────────────────────────────────────────────────

def _page_deep_dive(iso_id: str) -> None:
    it_load   = st.session_state.it_load
    ren_floor = st.session_state.ren_floor
    ren_frac  = ren_floor / 100.0
    cfg       = get_iso(iso_id)
    opt, scale = _constrained_opt(iso_id, it_load, ren_frac)

    _, main, _ = st.columns([0.035, 0.93, 0.035])

    with main:
        # ── Header ──
        top_l, top_r = st.columns([3, 1])
        with top_l:
            st.markdown(f"""
            <div style="padding:36px 0 0 0">
              <div style="font-size:9px;font-weight:600;letter-spacing:0.22em;
                          text-transform:uppercase;color:{MUTED};margin-bottom:6px">
                Microgrid Cost Explorer
              </div>
              <div style="font-size:24px;font-weight:300;color:{TEXT};
                          letter-spacing:-0.01em">
                {cfg['name']}
                <span style="color:{MUTED};font-size:18px;font-weight:300">
                  &nbsp;·&nbsp; {cfg['city']}
                </span>
              </div>
            </div>
            """, unsafe_allow_html=True)
        with top_r:
            st.markdown("<div style='padding-top:44px'>", unsafe_allow_html=True)
            if st.button("← All Markets", type="secondary"):
                st.session_state.selected_iso = None
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="font-size:11px;color:{MUTED};padding-bottom:20px">
          {it_load:.0f} MW IT load &nbsp;·&nbsp;
          ≥{ren_floor}% renewable &nbsp;·&nbsp;
          Unsubsidised &nbsp;·&nbsp; {YEAR} &nbsp;·&nbsp;
          7% WACC &nbsp;·&nbsp; {PROJECT_LIFE}-year project life
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<div style="height:1px;background:{BORDER}"></div>',
                    unsafe_allow_html=True)

        # ── Infeasible guard ──
        if opt is None:
            st.markdown(f"""
            <div style="padding:48px 0;color:{MUTED};font-size:14px">
              No feasible configuration exists for a ≥{ren_floor}% renewable constraint
              in this market. Lower the renewable floor and reconfigure.
            </div>
            """, unsafe_allow_html=True)
            return

        # ── Scaled capacities ──
        s_mw   = opt["S_mw"]     * scale
        b_mwh  = opt["B_mwh"]    * scale
        g_mw   = opt["G_min_mw"] * scale
        slcoe  = opt["slcoe_per_mwh"]
        ren_pct = opt["ren_share"] * 100

        # ── KPI row ──
        _section_label("Constrained Optimum")

        k1, k2, k3, k4, k5 = st.columns(5)
        for col, label, val, sub in [
            (k1, "System LCOE",    f"${slcoe:.2f}",    "$/MWh · constrained minimum"),
            (k2, "Solar",          f"{s_mw:.0f} MW",   "DC nameplate · fixed tilt"),
            (k3, "BESS",           f"{b_mwh:.0f} MWh", "4-hour Li-ion · utility scale"),
            (k4, "Gas Backup",     f"{g_mw:.0f} MW",   "RICE · minimum required"),
            (k5, "Renewable Share",f"{ren_pct:.0f}%",  f"≥{ren_floor}% required"),
        ]:
            col.markdown(_kpi_html(label, val, sub), unsafe_allow_html=True)

        # ── Run dispatch to get exact energy split ──
        try:
            disp = _dispatch_at_opt(
                iso_id,
                float(opt["S_mw"]),
                float(opt["B_mwh"]),
                float(opt["G_min_mw"]),
            )
            solar_pct = disp["solar_share_pct"]
            bess_pct  = disp["bess_share_pct"]
            gas_pct   = disp["gas_share_pct"]
        except Exception:
            # fallback: derive from slcoe table
            gas_pct   = (1.0 - opt["ren_share"]) * 100
            solar_pct = opt["ren_share"] * 65
            bess_pct  = opt["ren_share"] * 35

        # ── Cost + Energy charts ──
        _section_label("Cost Breakdown & Energy Mix")

        ch_left, spacer, ch_right = st.columns([5, 0.3, 2])

        with ch_left:
            # Stacked horizontal bar — cost per MWh
            demand  = opt["demand_mwh_yr"]
            c_solar = opt["solar_cost_usd_yr"] / demand
            c_bess  = opt["bess_cost_usd_yr"]  / demand
            c_gas   = opt["gas_cost_usd_yr"]   / demand

            fig_bar = go.Figure()
            for val, label, color in [
                (c_solar, "Solar",  C_SOLAR),
                (c_bess,  "BESS",   C_BESS),
                (c_gas,   "Gas",    C_GAS),
            ]:
                fig_bar.add_trace(go.Bar(
                    x=[val], y=[""],
                    orientation="h",
                    name=label,
                    marker_color=color,
                    marker_line_width=0,
                    text=f"  {label}&nbsp;&nbsp;${val:.2f}",
                    textposition="inside",
                    insidetextanchor="middle",
                    textfont=dict(size=11, color="white", family="Inter"),
                    hovertemplate=f"{label}: $%{{x:.2f}}/MWh<extra></extra>",
                ))

            fig_bar.update_layout(
                barmode="stack",
                height=72,
                margin=dict(t=0, b=0, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                xaxis=dict(
                    showgrid=False, zeroline=False,
                    tickprefix="$", ticksuffix="/MWh",
                    tickfont=dict(size=10, color=MUTED),
                    linecolor=BORDER, tickcolor=BORDER,
                ),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            )
            st.plotly_chart(fig_bar, use_container_width=True,
                            config={"displayModeBar": False})

            st.markdown(f"""
            <div style="font-size:10px;color:{MUTED};line-height:1.7;margin-top:4px">
              Solar + BESS capex annualised over {PROJECT_LIFE} yr at 7% WACC, includes
              fixed O&M. Gas includes capex, fixed O&M, and fuel at
              ${cfg['gas_price_per_mmbtu']:.2f}/MMBtu
              ({HEAT_RATE:.0f} MMBtu/MWh heat rate).
            </div>
            """, unsafe_allow_html=True)

        with ch_right:
            # Thin donut — energy mix
            fig_donut = go.Figure(go.Pie(
                labels=["Solar", "BESS", "Gas"],
                values=[solar_pct, bess_pct, gas_pct],
                hole=0.70,
                marker_colors=[C_SOLAR, C_BESS, C_GAS],
                marker_line=dict(color=BG, width=2),
                textinfo="none",
                hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
            ))
            fig_donut.add_annotation(
                text=f"<b>{ren_pct:.0f}%</b><br>"
                     f"<span style='font-size:9px'>renewable</span>",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=16, color=TEXT, family="Inter"),
                align="center",
            )
            fig_donut.update_layout(
                height=180,
                margin=dict(t=0, b=30, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=True,
                legend=dict(
                    orientation="h", x=0.5, xanchor="center", y=-0.12,
                    font=dict(size=10, color=MUTED, family="Inter"),
                    bgcolor="rgba(0,0,0,0)",
                    itemsizing="constant",
                ),
            )
            st.plotly_chart(fig_donut, use_container_width=True,
                            config={"displayModeBar": False})

        # ── Gas price sensitivity ──
        _section_label("Gas Price Sensitivity")

        sens_l, _, sens_r = st.columns([2, 3, 1])
        with sens_l:
            sens_df = _gas_sensitivity(opt, cfg)
            st.dataframe(sens_df, use_container_width=True)
            st.markdown(
                f'<div style="font-size:10px;color:{MUTED};margin-top:6px">'
                f'Optimal (S, B, G) held fixed. sLCOE shifts only via fuel cost.</div>',
                unsafe_allow_html=True,
            )

        # ── Footer ──
        st.markdown(f"""
        <div style="font-size:10px;color:{MUTED};letter-spacing:0.04em;
                    border-top:1px solid {BORDER};margin-top:48px;padding:24px 0">
          Sources: NREL ATB 2025 &nbsp;·&nbsp; Lazard LCOE+ 18.0 &nbsp;·&nbsp;
          EIA 2026 STEO &nbsp;·&nbsp; PVWatts V8 &nbsp;·&nbsp; Open-Meteo ERA5 &nbsp;·&nbsp;
          Unsubsidised baseline (pre-IRA ITC). All costs in 2024 USD.
        </div>
        """, unsafe_allow_html=True)


# ── Router ────────────────────────────────────────────────────────────────────

_init_state()
_css()

if not st.session_state.configured:
    _page_configure()
elif st.session_state.selected_iso:
    _page_deep_dive(st.session_state.selected_iso)
else:
    _page_markets()
