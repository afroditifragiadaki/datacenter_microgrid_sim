"""
Microgrid Cost Explorer
Datacenter Energy Optimization Platform
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.iso_registry import get_iso, get_all_isos
from models.pipeline import is_complete, run_pipeline

PROCESSED       = PROJECT_ROOT / "data" / "processed"
YEAR            = 2024
BASE_IT_LOAD_MW = 50.0
HEAT_RATE       = 9.0
PROJECT_LIFE    = 25

# ── Palette ───────────────────────────────────────────────────────────────────

BG      = "#060912"
SURFACE = "#0d1220"
BORDER  = "#1a2035"
TEXT    = "#e2e8f0"
MUTED   = "#4a5568"
ACCENT  = "#3b82f6"
C_SOLAR = "#f59e0b"
C_BESS  = "#3b82f6"
C_GAS   = "#ef4444"

# ── Page config ───────────────────────────────────────────────────────────────

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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@200;300;400;500;600&display=swap');

    html, body, * {{ font-family: 'Inter', sans-serif !important; }}

    /* Background */
    .stApp, [data-testid="stAppViewContainer"] {{
        background: {BG} !important;
        color: {TEXT} !important;
    }}
    [data-testid="stMainBlockContainer"], .main .block-container {{
        padding: 0 96px 96px !important;
        max-width: 1440px !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }}

    /* Hide chrome */
    header[data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stStatusWidget"],
    [data-testid="stSidebar"],
    .stDeployButton, footer {{ display: none !important; }}

    /* ── Top navbar via st.tabs ──────────────────────────────────────────── */

    /* Tab list = navbar background */
    [data-testid="stTabs"] > div[data-testid="stTabsListContainer"] {{
        background: {BG} !important;
        border-bottom: 1px solid {BORDER} !important;
        padding: 0 0 0 0 !important;
        gap: 0 !important;
        /* Brand injected via ::before */
        display: flex !important;
        align-items: center !important;
    }}

    /* Brand mark — lives as pseudo-element inside the tab bar */
    [data-testid="stTabs"] > div[data-testid="stTabsListContainer"]::before {{
        content: "◈  MICROGRID COST EXPLORER";
        font-size: 9px;
        font-weight: 600;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: {TEXT};
        white-space: nowrap;
        padding: 0 40px 0 32px;
        flex-shrink: 0;
        border-right: 1px solid {BORDER};
        height: 100%;
        display: flex;
        align-items: center;
        line-height: 52px;
    }}

    /* Each nav tab */
    [data-testid="stTabs"] button[role="tab"] {{
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
        color: {MUTED} !important;
        font-size: 10px !important;
        font-weight: 600 !important;
        letter-spacing: 0.14em !important;
        text-transform: uppercase !important;
        padding: 0 28px !important;
        height: 52px !important;
        margin: 0 !important;
        transition: color 0.15s !important;
    }}
    [data-testid="stTabs"] button[role="tab"]:hover {{
        color: {TEXT} !important;
        background: transparent !important;
    }}
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
        color: {TEXT} !important;
        border-bottom: 2px solid {ACCENT} !important;
        background: transparent !important;
    }}

    /* Remove default tab underline decoration */
    [data-testid="stTabs"] [data-baseweb="tab-highlight"],
    [data-testid="stTabs"] [data-baseweb="tab-border"] {{
        display: none !important;
    }}

    /* Tab content area — no extra padding */
    [data-testid="stTabs"] [data-testid="stTabsContent"] {{
        padding: 0 !important;
        border: none !important;
    }}

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
    }}

    /* ── Slider ── */
    [data-testid="stSlider"] [data-testid="stTickBarMin"],
    [data-testid="stSlider"] [data-testid="stTickBarMax"] {{
        color: {MUTED} !important;
        font-size: 10px !important;
    }}

    /* ── Buttons ── */
    .stButton > button {{
        border-radius: 2px !important;
        font-size: 10px !important;
        font-weight: 600 !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        transition: opacity 0.15s !important;
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
        padding: 7px 18px !important;
    }}
    .stButton > button[kind="secondary"]:hover {{
        color: {TEXT} !important;
        border-color: {MUTED} !important;
    }}

    /* ── Labels ── */
    label,
    .stNumberInput label,
    [data-testid="stSlider"] label {{
        font-size: 9px !important;
        font-weight: 600 !important;
        letter-spacing: 0.16em !important;
        text-transform: uppercase !important;
        color: {MUTED} !important;
    }}

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER} !important;
        border-radius: 2px !important;
    }}
    [data-testid="stDataFrame"] th {{
        font-size: 9px !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
    }}

    /* ── Misc ── */
    hr {{ border-color: {BORDER} !important; opacity: 1 !important; }}
    [data-testid="stSpinner"] > div {{ border-top-color: {ACCENT} !important; }}
    [data-testid="stAlert"] {{ border-radius: 2px !important; }}
    </style>
    """, unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────

def _init_state() -> None:
    for key, val in {
        "configured":    False,
        "it_load":       50.0,
        "ren_floor":     30,
        "selected_iso":  None,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Data helpers ──────────────────────────────────────────────────────────────

@st.cache_data
def _load_slcoe(iso_id: str) -> pd.DataFrame:
    p = PROCESSED / f"{iso_id.lower()}_slcoe_surface_{YEAR}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def _dispatch_at(iso_id: str, S: float, B: float, G: float) -> dict:
    from models.dispatcher import dispatch, load_timeseries
    ts = load_timeseries(PROCESSED, YEAR, iso_id=iso_id)
    _, summary = dispatch(S, B, G, ts)
    return summary


def _constrained_opt(iso_id: str, it_load: float, ren_frac: float):
    df = _load_slcoe(iso_id)
    if df.empty:
        return None, it_load / BASE_IT_LOAD_MW
    scale = it_load / BASE_IT_LOAD_MW
    df = df.copy()
    df["ren_share"] = 1.0 - df["gas_gen_mwh"] / df["demand_mwh_yr"]
    feasible = df[df["ren_share"] >= ren_frac]
    if feasible.empty:
        return None, scale
    return feasible.loc[feasible["slcoe_per_mwh"].idxmin()].to_dict(), scale


def _sensitivity(opt: dict, iso_cfg: dict) -> pd.DataFrame:
    base  = iso_cfg["gas_price_per_mmbtu"]
    ggen  = opt["gas_gen_mwh"]
    dem   = opt["demand_mwh_yr"]
    slcoe = opt["slcoe_per_mwh"]
    rows  = []
    for label, price in [("Low  −20%", base * 0.80),
                          ("Base",      base),
                          ("High +40%", base * 1.40)]:
        d = (price - base) * HEAT_RATE * ggen / dem
        rows.append({
            "Scenario":      label,
            "Gas  $/MMBtu":  f"${price:.2f}",
            "sLCOE  $/MWh":  f"${slcoe + d:.2f}",
            "Δ Base":        "—" if label == "Base" else f"{'+'if d>0 else ''}{d:.2f}",
        })
    return pd.DataFrame(rows).set_index("Scenario")


# ── UI primitives ─────────────────────────────────────────────────────────────

def _kpi(label: str, value: str, sub: str = "") -> str:
    sub_html = (f'<div style="font-size:10px;color:{MUTED};margin-top:5px;'
                f'line-height:1.4">{sub}</div>') if sub else ""
    return f"""
    <div style="padding:20px 24px 20px 0;border-top:1px solid {BORDER}">
      <div style="font-size:9px;font-weight:600;letter-spacing:0.18em;
                  text-transform:uppercase;color:{MUTED};margin-bottom:9px">{label}</div>
      <div style="font-size:26px;font-weight:300;color:{TEXT};
                  line-height:1;letter-spacing:-0.01em">{value}</div>
      {sub_html}
    </div>"""


def _section(text: str) -> None:
    st.markdown(
        f'<div style="font-size:9px;font-weight:600;letter-spacing:0.18em;'
        f'text-transform:uppercase;color:{MUTED};padding:28px 0 14px;'
        f'border-top:1px solid {BORDER}">{text}</div>',
        unsafe_allow_html=True,
    )


# ── PAGE: Configure ───────────────────────────────────────────────────────────

def _page_configure() -> None:
    st.markdown(f"""
    <div style="padding:80px 0 0">
      <div style="font-size:9px;font-weight:600;letter-spacing:0.22em;
                  text-transform:uppercase;color:{MUTED};margin-bottom:14px">
        Microgrid Cost Explorer
      </div>
      <div style="font-size:40px;font-weight:200;color:{TEXT};
                  line-height:1.15;margin-bottom:14px;letter-spacing:-0.02em">
        Datacenter Energy<br>Optimization Platform
      </div>
      <div style="font-size:14px;color:{MUTED};line-height:1.75;
                  margin-bottom:56px;max-width:560px">
        Model the optimal Solar · BESS · Gas microgrid for a collocated
        behind-the-meter facility. Compare system LCOE across all US ISO/RTO
        markets under your ESG constraints.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f'<div style="height:1px;background:{BORDER};margin:0 0 44px"></div>',
        unsafe_allow_html=True,
    )

    # Form — left-aligned, generous width, no centering columns
    form_l, form_r = st.columns([1.2, 2])

    with form_l:
        it_load = st.number_input(
            "IT Load  (MW)",
            min_value=1.0, max_value=500.0,
            value=float(st.session_state.it_load),
            step=5.0,
            help="Rated IT power. Total facility load is higher due to PUE.",
        )

        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

        ren_floor = st.slider(
            "Minimum Renewable Share  (%)",
            min_value=0, max_value=100,
            value=int(st.session_state.ren_floor),
            step=5,
            help="Solar + BESS must supply at least this share of annual energy.",
        )

        st.markdown(f"""
        <div style="font-size:11px;color:{MUTED};margin-top:8px;
                    margin-bottom:44px;line-height:1.65">
          Solar + BESS must supply
          <strong style="color:{TEXT}">&nbsp;≥ {ren_floor}%&nbsp;</strong>
          of annual demand. The optimizer finds the lowest-cost (S, B, G)
          satisfying this ESG constraint across each market.
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            f'<div style="height:1px;background:{BORDER};margin-bottom:32px"></div>',
            unsafe_allow_html=True,
        )

        _, btn_col = st.columns([1, 2])
        with btn_col:
            if st.button("Analyze all markets →", type="primary",
                         use_container_width=True):
                st.session_state.it_load     = it_load
                st.session_state.ren_floor   = ren_floor
                st.session_state.configured  = True
                st.session_state.selected_iso = None
                st.rerun()

    with form_r:
        st.markdown(f"""
        <div style="padding:8px 0 0 64px">
          <div style="font-size:9px;font-weight:600;letter-spacing:0.18em;
                      text-transform:uppercase;color:{MUTED};margin-bottom:20px">
            Data Sources
          </div>
          <div style="font-size:12px;color:{MUTED};line-height:2.2">
            NREL ATB 2025 — Technology capital costs, moderate scenario<br>
            Lazard LCOE+ 18.0 — Levelised cost benchmarks<br>
            EIA 2026 STEO — Regional natural gas prices<br>
            PVWatts V8 / NSRDB — Solar resource (TMY)<br>
            Open-Meteo ERA5 — Hourly temperature, 2024<br>
          </div>
          <div style="margin-top:32px;font-size:12px;color:{MUTED};line-height:2.2">
            7% WACC &nbsp;·&nbsp; {PROJECT_LIFE}-year project life<br>
            Unsubsidised baseline — pre-IRA ITC<br>
            All costs in 2024 USD
          </div>
        </div>
        """, unsafe_allow_html=True)


# ── PAGE: Markets ─────────────────────────────────────────────────────────────

def _page_markets() -> None:
    it_load  = st.session_state.it_load
    ren_frac = st.session_state.ren_floor / 100.0
    all_isos = get_all_isos()

    st.markdown(f"""
    <div style="padding:48px 0 0">
      <div style="display:flex;align-items:flex-end;
                  justify-content:space-between;margin-bottom:6px">
        <div style="font-size:28px;font-weight:300;color:{TEXT};
                    letter-spacing:-0.01em">All Markets</div>
        <div style="font-size:11px;color:{MUTED};padding-bottom:4px">
          {it_load:.0f} MW IT load &nbsp;·&nbsp;
          ≥{st.session_state.ren_floor}% renewable &nbsp;·&nbsp;
          {YEAR} &nbsp;·&nbsp; Unsubsidised
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    btn_col, _ = st.columns([1, 8])
    with btn_col:
        if st.button("← Reconfigure", type="secondary"):
            st.session_state.configured   = False
            st.session_state.selected_iso = None
            st.rerun()

    st.markdown(
        f'<div style="height:1px;background:{BORDER};margin:16px 0 0"></div>',
        unsafe_allow_html=True,
    )

    with st.container():
        ready_rows, pending_rows = [], []

        for iso_id, cfg in all_isos.items():
            computed = is_complete(iso_id, YEAR)
            opt, scale = _constrained_opt(iso_id, it_load, ren_frac) if computed else (None, 1.0)
            meta = {"iso": iso_id, "cfg": cfg, "opt": opt, "scale": scale}

            if computed and opt:
                ready_rows.append({**meta,
                    "Market":       iso_id,
                    "Location":     cfg["city"],
                    "sLCOE $/MWh":  round(opt["slcoe_per_mwh"], 2),
                    "Renewable":    f"{opt['ren_share']*100:.0f}%",
                    "Solar MW":     f"{opt['S_mw'] * scale:.0f}",
                    "BESS MWh":     f"{opt['B_mwh'] * scale:.0f}",
                    "Gas MW":       f"{opt['G_min_mw'] * scale:.0f}",
                })
            elif computed:
                ready_rows.append({**meta,
                    "Market": iso_id, "Location": cfg["city"],
                    "sLCOE $/MWh": None, "Renewable": "Infeasible",
                    "Solar MW": "—", "BESS MWh": "—", "Gas MW": "—",
                })
            else:
                pending_rows.append(meta)

        ready_rows.sort(key=lambda r: (r["sLCOE $/MWh"] is None,
                                        r["sLCOE $/MWh"] or 9999))

        _section("Available Markets — Ranked by System LCOE")

        if ready_rows:
            cols = ["Market", "Location", "sLCOE $/MWh",
                    "Renewable", "Solar MW", "BESS MWh", "Gas MW"]
            df = pd.DataFrame([{c: r[c] for c in cols} for r in ready_rows])

            event = st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                column_config={
                    "sLCOE $/MWh": st.column_config.NumberColumn(format="$%.2f"),
                },
                height=min(80 + len(ready_rows) * 36, 380),
            )
            st.markdown(
                f'<div style="font-size:10px;color:{MUTED};margin-top:8px">'
                f'Select a row to explore the optimal configuration in detail.</div>',
                unsafe_allow_html=True,
            )
            if event.selection and event.selection.rows:
                sel = ready_rows[event.selection.rows[0]]
                if sel["opt"] is not None:
                    st.session_state.selected_iso = sel["iso"]
                    st.rerun()
        else:
            st.markdown(
                f'<div style="color:{MUTED};font-size:13px;padding:24px 0">'
                f'No markets computed yet. Run a pipeline below.</div>',
                unsafe_allow_html=True,
            )

        if pending_rows:
            _section("Pending Markets — Pipeline Not Yet Run")
            p_cols = st.columns(min(len(pending_rows), 4))
            for i, row in enumerate(pending_rows):
                cfg = row["cfg"]
                with p_cols[i % 4]:
                    st.markdown(f"""
                    <div style="border:1px solid {BORDER};padding:18px 20px;
                                margin-bottom:10px">
                      <div style="font-size:9px;font-weight:600;letter-spacing:0.14em;
                                  text-transform:uppercase;color:{MUTED};
                                  margin-bottom:5px">{row['iso']}</div>
                      <div style="font-size:14px;color:{TEXT};margin-bottom:14px">
                        {cfg['city']}</div>
                      <div style="font-size:10px;color:{MUTED};line-height:1.8">
                        Gas ${cfg['gas_price_per_mmbtu']:.2f}/MMBtu &nbsp;·&nbsp;
                        {cfg['capex_multiplier']:.2f}× CAPEX
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                    iso_key = row["iso"]
                    if st.button(f"Run {iso_key}", key=f"run_{iso_key}",
                                 use_container_width=True):
                        _log = st.empty()
                        _msgs: list[str] = []
                        def _mk(area, msgs):
                            def _fn(m):
                                msgs.append(m)
                                area.code("\n".join(msgs[-16:]))
                            return _fn
                        with st.spinner(f"Running {iso_key}…"):
                            try:
                                run_pipeline(iso_key, YEAR, log=_mk(_log, _msgs))
                                st.cache_data.clear()
                                st.rerun()
                            except Exception as exc:
                                st.error(str(exc))

        st.markdown(f"""
        <div style="font-size:10px;color:{MUTED};letter-spacing:0.04em;
                    border-top:1px solid {BORDER};margin-top:48px;padding:24px 0">
          Sources: NREL ATB 2025 · Lazard LCOE+ 18.0 · EIA 2026 STEO ·
          PVWatts V8 · Open-Meteo ERA5 · Unsubsidised (pre-IRA ITC) · 2024 USD
        </div>
        """, unsafe_allow_html=True)


# ── PAGE: Deep Dive ───────────────────────────────────────────────────────────

def _page_deep_dive(iso_id: str) -> None:
    it_load  = st.session_state.it_load
    ren_frac = st.session_state.ren_floor / 100.0
    cfg      = get_iso(iso_id)
    opt, scale = _constrained_opt(iso_id, it_load, ren_frac)

    st.markdown(f"""
    <div style="padding:48px 0 0">
      <div style="display:flex;align-items:flex-end;
                  justify-content:space-between;margin-bottom:6px">
        <div style="font-size:28px;font-weight:300;color:{TEXT};
                    letter-spacing:-0.01em">
          {cfg['name']}
          <span style="color:{MUTED};font-size:20px"> · {cfg['city']}</span>
        </div>
        <div style="font-size:11px;color:{MUTED};padding-bottom:4px">
          {it_load:.0f} MW IT load &nbsp;·&nbsp;
          ≥{st.session_state.ren_floor}% renewable &nbsp;·&nbsp;
          {YEAR} &nbsp;·&nbsp; Unsubsidised
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    btn_col, _ = st.columns([1, 8])
    with btn_col:
        if st.button("← All Markets", type="secondary"):
            st.session_state.selected_iso = None
            st.rerun()

    st.markdown(
        f'<div style="height:1px;background:{BORDER};margin:16px 0 0"></div>',
        unsafe_allow_html=True,
    )

    with st.container():
        if opt is None:
            st.markdown(f"""
            <div style="padding:48px 0;color:{MUTED};font-size:14px">
              No feasible configuration found for ≥{st.session_state.ren_floor}%
              renewable in this market. Lower the renewable floor and reconfigure.
            </div>""", unsafe_allow_html=True)
            return

        s_mw   = opt["S_mw"]      * scale
        b_mwh  = opt["B_mwh"]     * scale
        g_mw   = opt["G_min_mw"]  * scale
        slcoe  = opt["slcoe_per_mwh"]
        ren_pct = opt["ren_share"] * 100

        _section("Constrained Optimum")
        k1, k2, k3, k4, k5 = st.columns(5)
        for col, lbl, val, sub in [
            (k1, "System LCOE",     f"${slcoe:.2f}",    "$/MWh · constrained minimum"),
            (k2, "Solar",           f"{s_mw:.0f} MW",   "DC nameplate · fixed tilt"),
            (k3, "BESS",            f"{b_mwh:.0f} MWh", "4-hour Li-ion"),
            (k4, "Gas Backup",      f"{g_mw:.0f} MW",   "RICE · minimum required"),
            (k5, "Renewable Share", f"{ren_pct:.0f}%",  f"≥{st.session_state.ren_floor}% required"),
        ]:
            col.markdown(_kpi(lbl, val, sub), unsafe_allow_html=True)

        # Energy mix from dispatch
        try:
            d = _dispatch_at(iso_id, float(opt["S_mw"]),
                             float(opt["B_mwh"]), float(opt["G_min_mw"]))
            solar_pct = d["solar_share_pct"]
            bess_pct  = d["bess_share_pct"]
            gas_pct   = d["gas_share_pct"]
        except Exception:
            gas_pct   = (1 - opt["ren_share"]) * 100
            solar_pct = opt["ren_share"] * 65
            bess_pct  = opt["ren_share"] * 35

        _section("Cost Breakdown & Energy Mix")
        c_left, _, c_right = st.columns([5, 0.2, 2])

        with c_left:
            dem   = opt["demand_mwh_yr"]
            c_sol = opt["solar_cost_usd_yr"] / dem
            c_bes = opt["bess_cost_usd_yr"]  / dem
            c_gas = opt["gas_cost_usd_yr"]   / dem

            fig_bar = go.Figure()
            for val, lbl, color in [(c_sol, "Solar", C_SOLAR),
                                     (c_bes, "BESS",  C_BESS),
                                     (c_gas, "Gas",   C_GAS)]:
                fig_bar.add_trace(go.Bar(
                    x=[val], y=[""], orientation="h",
                    marker_color=color, marker_line_width=0,
                    text=f"  {lbl}  ${val:.2f}",
                    textposition="inside", insidetextanchor="middle",
                    textfont=dict(size=11, color="white"),
                    hovertemplate=f"{lbl}: $%{{x:.2f}}/MWh<extra></extra>",
                ))
            fig_bar.update_layout(
                barmode="stack", height=68,
                margin=dict(t=0, b=0, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, tickprefix="$",
                           tickfont=dict(size=10, color=MUTED), linecolor=BORDER),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            )
            st.plotly_chart(fig_bar, use_container_width=True,
                            config={"displayModeBar": False})
            st.markdown(
                f'<div style="font-size:10px;color:{MUTED};margin-top:4px;line-height:1.7">'
                f'CAPEX annualised over {PROJECT_LIFE} yr at 7% WACC · includes fixed O&M '
                f'· gas fuel at ${cfg["gas_price_per_mmbtu"]:.2f}/MMBtu</div>',
                unsafe_allow_html=True,
            )

        with c_right:
            fig_d = go.Figure(go.Pie(
                labels=["Solar", "BESS", "Gas"],
                values=[solar_pct, bess_pct, gas_pct],
                hole=0.70,
                marker_colors=[C_SOLAR, C_BESS, C_GAS],
                marker_line=dict(color=BG, width=2),
                textinfo="none",
                hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
            ))
            fig_d.add_annotation(
                text=f"<b>{ren_pct:.0f}%</b><br>renewable",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=15, color=TEXT, family="Inter"),
                align="center",
            )
            fig_d.update_layout(
                height=180, margin=dict(t=0, b=28, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)", showlegend=True,
                legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.12,
                            font=dict(size=10, color=MUTED),
                            bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_d, use_container_width=True,
                            config={"displayModeBar": False})

        _section("Gas Price Sensitivity")
        s_l, _, _ = st.columns([2, 2, 2])
        with s_l:
            st.dataframe(_sensitivity(opt, cfg), use_container_width=True)
            st.markdown(
                f'<div style="font-size:10px;color:{MUTED};margin-top:6px">'
                f'Optimal (S, B, G) held fixed. Only fuel cost varies.</div>',
                unsafe_allow_html=True,
            )

        st.markdown(f"""
        <div style="font-size:10px;color:{MUTED};letter-spacing:0.04em;
                    border-top:1px solid {BORDER};margin-top:48px;padding:24px 0">
          Sources: NREL ATB 2025 · Lazard LCOE+ 18.0 · EIA 2026 STEO ·
          PVWatts V8 · Open-Meteo ERA5 · Unsubsidised (pre-IRA ITC) · 2024 USD
        </div>
        """, unsafe_allow_html=True)


# ── PAGE: Methodology ────────────────────────────────────────────────────────

def _page_methodology() -> None:
    st.markdown(f"""
    <div style="padding:80px 0 0">
      <div style="font-size:9px;font-weight:600;letter-spacing:0.22em;
                  text-transform:uppercase;color:{MUTED};margin-bottom:14px">
        How It Works
      </div>
      <div style="font-size:40px;font-weight:200;color:{TEXT};
                  line-height:1.15;margin-bottom:14px;letter-spacing:-0.02em">
        Methodology
      </div>
      <div style="font-size:14px;color:{MUTED};line-height:1.75;
                  margin-bottom:56px;max-width:620px">
        A constrained optimisation over a three-asset microgrid — Solar PV,
        Battery Energy Storage (BESS), and gas-fired backup — minimising
        system levelised cost of energy subject to a renewable-share floor.
      </div>
    </div>
    <div style="height:1px;background:{BORDER};margin-bottom:48px"></div>
    """, unsafe_allow_html=True)

    c_l, _, c_r = st.columns([5, 0.4, 4])

    with c_l:
        st.markdown(f"""
        <div style="font-size:9px;font-weight:600;letter-spacing:0.18em;
                    text-transform:uppercase;color:{MUTED};
                    border-top:1px solid {BORDER};padding-top:20px;
                    margin-bottom:18px">System LCOE (sLCOE)</div>
        <div style="font-size:13px;color:{MUTED};line-height:2.0;
                    margin-bottom:32px">
          All capital costs are annualised using a fixed-charge rate derived
          from a <strong style="color:{TEXT}">7% WACC</strong> over a
          <strong style="color:{TEXT}">{PROJECT_LIFE}-year project life</strong>.
          Annual fixed O&amp;M is added, then fuel costs for gas generation are
          stacked on top. The result is divided by total annual site demand to
          yield a single $/MWh figure that is directly comparable across markets
          and configurations.
        </div>

        <div style="font-size:9px;font-weight:600;letter-spacing:0.18em;
                    text-transform:uppercase;color:{MUTED};
                    border-top:1px solid {BORDER};padding-top:20px;
                    margin-bottom:18px">Optimisation</div>
        <div style="font-size:13px;color:{MUTED};line-height:2.0;
                    margin-bottom:32px">
          A brute-force grid search is run over Solar capacity <em>(S)</em>,
          BESS energy capacity <em>(B)</em>, and gas nameplate <em>(G)</em>.
          For each candidate triplet an hourly dispatch simulation determines
          the share of demand met by each source. Configurations that satisfy
          the user-defined renewable-share floor are collected; the
          minimum-sLCOE feasible point is reported as the constrained optimum.
        </div>

        <div style="font-size:9px;font-weight:600;letter-spacing:0.18em;
                    text-transform:uppercase;color:{MUTED};
                    border-top:1px solid {BORDER};padding-top:20px;
                    margin-bottom:18px">Dispatch Logic</div>
        <div style="font-size:13px;color:{MUTED};line-height:2.0">
          Each hour, available solar generation (from PVWatts TMY profiles) is
          used first. Surplus charges the BESS; deficit is served by BESS
          discharge down to its state-of-charge floor, with gas covering any
          remaining shortfall. The BESS model uses a constant round-trip
          efficiency and enforces capacity limits. Gas is treated as a
          fully-flexible peaker with no minimum run constraint.
        </div>
        """, unsafe_allow_html=True)

    with c_r:
        st.markdown(f"""
        <div style="font-size:9px;font-weight:600;letter-spacing:0.18em;
                    text-transform:uppercase;color:{MUTED};
                    border-top:1px solid {BORDER};padding-top:20px;
                    margin-bottom:18px">Data Sources</div>
        """, unsafe_allow_html=True)

        sources = [
            ("NREL ATB 2025", "Technology capital costs — moderate scenario. "
             "Covers utility-scale solar PV, Li-ion BESS, and gas RICE."),
            ("Lazard LCOE+ 18.0", "Independent levelised cost benchmarks used "
             "for cross-validation of sLCOE outputs."),
            ("EIA 2026 STEO", "Regional natural gas wellhead price forecasts "
             "by ISO/RTO market area."),
            ("PVWatts V8 / NSRDB", "Hourly AC generation profiles for a "
             "fixed-tilt utility array at each market's representative "
             "location, using a Typical Meteorological Year."),
            ("Open-Meteo ERA5", "Hourly dry-bulb temperature for 2024, used "
             "to adjust IT cooling load and PUE estimates."),
        ]
        for title, body in sources:
            st.markdown(f"""
            <div style="margin-bottom:22px">
              <div style="font-size:11px;font-weight:600;color:{TEXT};
                          margin-bottom:6px">{title}</div>
              <div style="font-size:12px;color:{MUTED};line-height:1.8">
                {body}
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="font-size:9px;font-weight:600;letter-spacing:0.18em;
                    text-transform:uppercase;color:{MUTED};
                    border-top:1px solid {BORDER};padding-top:20px;
                    margin-top:8px;margin-bottom:18px">Key Assumptions</div>
        <div style="font-size:12px;color:{MUTED};line-height:2.2">
          Unsubsidised baseline — pre-IRA Investment Tax Credit<br>
          PUE 1.35 · DC power draw = IT load × PUE<br>
          BESS round-trip efficiency 85%<br>
          Solar DC:AC ratio 1.3 · fixed-tilt, no tracking<br>
          Gas heat rate {HEAT_RATE} MMBtu/MWh (RICE)<br>
          All costs in real 2024 USD
        </div>
        """, unsafe_allow_html=True)


# ── PAGE: Team ────────────────────────────────────────────────────────────────

TEAM = [
    {
        "name":     "Afroditi Fragkiadaki",
        "email":    "af3619@columbia.edu",
        "linkedin": "https://www.linkedin.com/in/aphroditi-fragkiadaki/",
    },
    {
        "name":     "Daniel Holland",
        "email":    "doh2105@columbia.edu",
        "linkedin": "https://www.linkedin.com/in/daniel-holland-2b3a58218/",
    },
    {
        "name":     "Raphael Vogeley",
        "email":    "rpv2113@columbia.edu",
        "linkedin": "https://www.linkedin.com/in/raphael-vogeley/",
    },
    {
        "name":     "Tselmeg Mendsaikhan",
        "email":    "tm3516@columbia.edu",
        "linkedin": "https://www.linkedin.com/in/tselmeg-mendsaikhan-d12152026/",
    },
]


def _page_team() -> None:
    st.markdown(f"""
    <div style="padding:80px 0 0">
      <div style="font-size:9px;font-weight:600;letter-spacing:0.22em;
                  text-transform:uppercase;color:{MUTED};margin-bottom:14px">
        Columbia University
      </div>
      <div style="font-size:40px;font-weight:200;color:{TEXT};
                  line-height:1.15;margin-bottom:14px;letter-spacing:-0.02em">
        The Team
      </div>
      <div style="font-size:14px;color:{MUTED};line-height:1.75;
                  margin-bottom:56px;max-width:560px">
        ELEN4510 Grid Modernization &amp; Clean Tech — Columbia University.
      </div>
    </div>
    <div style="height:1px;background:{BORDER};margin-bottom:48px"></div>
    """, unsafe_allow_html=True)

    cols = st.columns(min(len(TEAM), 4), gap="large")
    for i, member in enumerate(TEAM):
        with cols[i % 4]:
            initials = "".join(w[0] for w in member["name"].split()[:2]).upper()
            st.markdown(f"""
            <div style="border:1px solid {BORDER};padding:28px 24px 24px">
              <div style="width:48px;height:48px;border-radius:50%;
                          background:{SURFACE};border:1px solid {BORDER};
                          display:flex;align-items:center;justify-content:center;
                          font-size:14px;font-weight:600;color:{TEXT};
                          letter-spacing:0.04em;margin-bottom:20px">
                {initials}
              </div>
              <div style="font-size:15px;font-weight:500;color:{TEXT};
                          margin-bottom:8px">{member['name']}</div>
              <div style="font-size:11px;color:{MUTED};margin-bottom:20px">
                {member['email']}
              </div>
              <a href="{member['linkedin']}" target="_blank"
                 style="font-size:9px;font-weight:600;letter-spacing:0.14em;
                        text-transform:uppercase;color:{MUTED};
                        text-decoration:none;border:1px solid {BORDER};
                        padding:7px 14px;display:inline-block;
                        white-space:nowrap;transition:color 0.15s">
                LinkedIn ↗
              </a>
            </div>
            """, unsafe_allow_html=True)


# ── Router ────────────────────────────────────────────────────────────────────

_init_state()
_css()

# ── Top navbar — add future pages as new tabs here ──
tab_optimizer, tab_methodology, tab_team = st.tabs(
    ["Optimizer", "Methodology", "Team"]
)

with tab_optimizer:
    if not st.session_state.configured:
        _page_configure()
    elif st.session_state.selected_iso:
        _page_deep_dive(st.session_state.selected_iso)
    else:
        _page_markets()

with tab_methodology:
    _page_methodology()

with tab_team:
    _page_team()
