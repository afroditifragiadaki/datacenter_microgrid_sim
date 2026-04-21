"""
Microgrid Cost Explorer
Datacenter Energy Optimization Platform
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

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
C_GRID  = "#10b981"

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
        background: #1e3a5f !important;
        border-bottom: 2px solid #2d5a9e !important;
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
        white-space: nowrap !important;
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
        "configured":      False,
        "it_load":         50.0,
        "ren_floor":       30,
        "selected_iso":    None,
        "goto_methodology": False,
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


@st.cache_data
def _load_grid_slcoe(iso_id: str) -> pd.DataFrame:
    p = PROCESSED / f"{iso_id.lower()}_grid_slcoe_surface_{YEAR}.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


@st.cache_data
def _dispatch_grid_at(iso_id: str, S: float, B: float, G: float) -> dict:
    from models.dispatcher import dispatch_grid, load_timeseries
    from models.gas_model import HEAT_RATE_MMBTU_PER_MWH
    from models.iso_registry import get_costs
    ts = load_timeseries(PROCESSED, YEAR, iso_id=iso_id)
    prices_path = PROCESSED / f"{iso_id.lower()}_grid_prices_{YEAR}.csv"
    prices = pd.read_csv(prices_path, index_col="datetime", parse_dates=True)
    ts = ts.copy()
    ts["grid_price_per_mwh"] = prices["grid_price_per_mwh"].values
    cfg   = get_iso(iso_id)
    costs = get_costs()
    gas_marginal = (
        cfg["gas_price_per_mmbtu"] * HEAT_RATE_MMBTU_PER_MWH
        + costs["gas_rice"]["opex_variable_per_mwh"]
    )
    _, summary = dispatch_grid(S, B, G, ts, gas_marginal)
    return summary


def _constrained_opt_grid(iso_id: str, it_load: float, ren_frac: float):
    df = _load_grid_slcoe(iso_id)
    if df.empty:
        return None, it_load / BASE_IT_LOAD_MW
    scale = it_load / BASE_IT_LOAD_MW
    df = df.copy()
    df["ren_share"] = 1.0 - (
        df["gas_gen_mwh"] + df["grid_import_mwh_yr"]
    ) / df["demand_mwh_yr"]
    feasible = df[df["ren_share"] >= ren_frac]
    if feasible.empty:
        return None, scale
    return feasible.loc[feasible["slcoe_per_mwh"].idxmin()].to_dict(), scale


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


# ── Datacenter SVG background ─────────────────────────────────────────────────

def _datacenter_svg() -> str:
    rows = []
    num_racks, rack_w, rack_h = 8, 148, 420
    spacing, start_x, start_y = 198, 40, 30
    srv_h, srv_gap = 16, 4
    num_srv = rack_h // (srv_h + srv_gap)
    for i in range(num_racks):
        rx = start_x + i * spacing
        rows.append(
            f'<rect x="{rx}" y="{start_y}" width="{rack_w}" height="{rack_h}" '
            f'rx="3" fill="none" stroke="#2d5a9e" stroke-width="0.7"/>'
        )
        for j in range(num_srv):
            sy = start_y + 6 + j * (srv_h + srv_gap)
            rows.append(
                f'<rect x="{rx+4}" y="{sy}" width="{rack_w-8}" height="{srv_h}" '
                f'rx="1.5" fill="#0a1628" stroke="#1a3050" stroke-width="0.4"/>'
            )
            bar_w = 40 + ((i * 7 + j * 11) % 55)
            rows.append(
                f'<rect x="{rx+8}" y="{sy+5}" width="{bar_w}" height="1.5" '
                f'rx="1" fill="#1e3a5f"/>'
            )
            rows.append(
                f'<circle cx="{rx+rack_w-14}" cy="{sy+8}" r="2" '
                f'fill="#3b82f6" opacity="0.85"/>'
            )
            if (i + j) % 3 == 0:
                rows.append(
                    f'<circle cx="{rx+rack_w-24}" cy="{sy+8}" r="2" '
                    f'fill="#22c55e" opacity="0.75"/>'
                )
    total_w = start_x * 2 + num_racks * spacing
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {total_w} {start_y*2+rack_h}" '
        f'preserveAspectRatio="xMidYMid slice">'
        + "".join(rows) + "</svg>"
    )


# ── PAGE: Configure ───────────────────────────────────────────────────────────

def _page_configure() -> None:
    # ── Hero ──────────────────────────────────────────────────────────────────
    _svg = _datacenter_svg()
    st.markdown(f"""
    <div style="padding:80px 0 56px;text-align:center;position:relative;overflow:hidden">
      <div style="position:absolute;inset:0;opacity:0.22;pointer-events:none;
                  display:flex;align-items:center;justify-content:center">
        {_svg}
      </div>
      <div style="position:relative;z-index:1">
        <div style="font-size:9px;font-weight:600;letter-spacing:0.22em;
                    text-transform:uppercase;color:{MUTED};margin-bottom:14px">
          Microgrid Cost Explorer
        </div>
        <div style="font-size:40px;font-weight:200;color:{TEXT};
                    line-height:1.15;margin-bottom:14px;letter-spacing:-0.02em">
          Datacenter Energy<br>Optimization Platform
        </div>
        <div style="font-size:14px;color:{MUTED};line-height:1.75;
                    max-width:560px;margin-left:auto;margin-right:auto">
          Model the optimal Solar · BESS · Gas microgrid for a collocated
          behind-the-meter facility. Compare system LCOE across all US ISO/RTO
          markets under your ESG constraints.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f'<div style="height:1px;background:{BORDER};margin:0 0 44px"></div>',
        unsafe_allow_html=True,
    )

    # ── Optimizer form — centered ──────────────────────────────────────────────
    _, form_col, _ = st.columns([1, 1.4, 1])

    with form_col:
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

        if st.button("Analyze all markets →", type="primary",
                     use_container_width=True):
            st.session_state.it_load      = it_load
            st.session_state.ren_floor    = ren_floor
            st.session_state.configured   = True
            st.session_state.selected_iso = None
            st.rerun()

    # ── Data sources — centered at bottom ─────────────────────────────────────
    st.markdown(
        f'<div style="height:1px;background:{BORDER};margin:56px 0 40px"></div>',
        unsafe_allow_html=True,
    )

    _, ds_col, _ = st.columns([1, 2, 1])

    with ds_col:
        st.markdown(f"""
        <div style="text-align:center">
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
            gridstatus / ISO public APIs — Hourly day-ahead LMP, 2024<br>
          </div>
          <div style="margin-top:20px;font-size:12px;color:{MUTED};line-height:2.2">
            7% WACC &nbsp;·&nbsp; {PROJECT_LIFE}-year project life &nbsp;·&nbsp;
            Unsubsidised baseline — pre-IRA ITC &nbsp;·&nbsp; All costs in 2024 USD
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

        if st.button("View Methodology →", type="secondary", use_container_width=True):
            st.session_state.goto_methodology = True
            st.rerun()


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

    btn_col, _ = st.columns([2, 7])
    with btn_col:
        if st.button("← Reconfigure", type="secondary", use_container_width=True):
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
                f'<div style="font-size:11px;color:{MUTED};margin-top:10px;'
                f'padding:10px 14px;border:1px solid {BORDER};display:inline-block">'
                f'Click any row to deep dive into that market\'s optimal configuration — '
                f'cost breakdown, energy mix, and gas price sensitivity.</div>',
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

        # ── Microgrid vs Grid-Connected comparison ────────────────────────────
        comp_rows = []
        for row in ready_rows:
            if row["opt"] is None:
                continue
            g_opt, _ = _constrained_opt_grid(row["iso"], it_load, ren_frac)
            if g_opt is None:
                continue
            mg = row["opt"]["slcoe_per_mwh"]
            gr = g_opt["slcoe_per_mwh"]
            comp_rows.append({
                "iso":      row["iso"],
                "location": row["cfg"]["city"],
                "mg_slcoe": mg,
                "gr_slcoe": gr,
                "delta":    mg - gr,          # positive = grid cheaper
                "winner":   "Grid" if gr < mg else "Microgrid",
            })

        if comp_rows:
            _section("Microgrid vs Grid-Connected — sLCOE Comparison")

            isos_c  = [r["iso"]      for r in comp_rows]
            mg_vals = [r["mg_slcoe"] for r in comp_rows]
            gr_vals = [r["gr_slcoe"] for r in comp_rows]

            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Bar(
                name="Microgrid", x=isos_c, y=mg_vals,
                marker_color=C_SOLAR, marker_line_width=0,
                text=[f"${v:.0f}" for v in mg_vals],
                textposition="outside",
                textfont=dict(size=10, color=TEXT),
                hovertemplate="Microgrid: $%{y:.2f}/MWh<extra></extra>",
            ))
            fig_cmp.add_trace(go.Bar(
                name="Grid-Connected", x=isos_c, y=gr_vals,
                marker_color=C_GRID, marker_line_width=0,
                text=[f"${v:.0f}" for v in gr_vals],
                textposition="outside",
                textfont=dict(size=10, color=TEXT),
                hovertemplate="Grid-Connected: $%{y:.2f}/MWh<extra></extra>",
            ))
            fig_cmp.update_layout(
                barmode="group", height=320,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=32, b=0, l=0, r=0),
                yaxis=dict(
                    tickprefix="$", ticksuffix="/MWh",
                    tickfont=dict(color=MUTED, size=10),
                    gridcolor=BORDER, zeroline=False,
                ),
                xaxis=dict(tickfont=dict(color=TEXT, size=12), gridcolor="rgba(0,0,0,0)"),
                legend=dict(
                    font=dict(color=MUTED, size=10),
                    bgcolor="rgba(0,0,0,0)",
                    orientation="h", x=0, y=1.08,
                ),
                font=dict(family="Inter"),
                bargap=0.28, bargroupgap=0.06,
            )
            st.plotly_chart(fig_cmp, use_container_width=True,
                            config={"displayModeBar": False})

            # Summary table
            cmp_display = pd.DataFrame([{
                "Market":              r["iso"],
                "Location":            r["location"],
                "Microgrid  $/MWh":    round(r["mg_slcoe"], 2),
                "Grid-Connected $/MWh": round(r["gr_slcoe"], 2),
                "Δ  $/MWh":            round(abs(r["delta"]), 2),
                "Cheaper":             r["winner"],
            } for r in comp_rows])

            st.dataframe(
                cmp_display, use_container_width=True, hide_index=True,
                column_config={
                    "Microgrid  $/MWh":     st.column_config.NumberColumn(format="$%.2f"),
                    "Grid-Connected $/MWh": st.column_config.NumberColumn(format="$%.2f"),
                    "Δ  $/MWh":             st.column_config.NumberColumn(format="$%.2f"),
                },
                height=80 + len(comp_rows) * 36,
            )
            st.markdown(
                f'<div style="font-size:10px;color:{MUTED};margin-top:8px">'
                f'Grid renewable share defined as on-site solar+BESS only — '
                f'grid imports and gas both counted as non-renewable. '
                f'Same renewable floor applied to both models.</div>',
                unsafe_allow_html=True,
            )

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

    btn_col, _ = st.columns([2, 7])
    with btn_col:
        if st.button("← All Markets", type="secondary", use_container_width=True):
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

        # ── vs Grid-Connected ──────────────────────────────────────────────────
        _section("vs Grid-Connected")

        g_opt, _ = _constrained_opt_grid(iso_id, it_load, ren_frac)

        if g_opt is None:
            st.markdown(
                f'<div style="color:{MUTED};font-size:13px;padding:8px 0">'
                f'Grid-connected data not available for this market.</div>',
                unsafe_allow_html=True,
            )
        else:
            gr_slcoe  = g_opt["slcoe_per_mwh"]
            gr_s      = g_opt["S_mw"]      * scale
            gr_b      = g_opt["B_mwh"]     * scale
            gr_g      = g_opt["G_mw"]      * scale
            gr_import = g_opt["grid_import_mwh_yr"] * scale
            gr_dem    = g_opt["demand_mwh_yr"]       * scale
            gr_ren    = g_opt["ren_share"]  * 100
            delta     = slcoe - gr_slcoe   # positive → grid is cheaper

            # Winner banner
            if delta > 0:
                winner_txt  = f"Grid-connected saves&nbsp; <strong>${delta:.2f}/MWh</strong>"
                winner_col  = C_GRID
            else:
                winner_txt  = f"Microgrid saves&nbsp; <strong>${abs(delta):.2f}/MWh</strong>"
                winner_col  = C_SOLAR
            st.markdown(
                f'<div style="border-left:3px solid {winner_col};'
                f'padding:10px 18px;background:{SURFACE};margin-bottom:20px;'
                f'font-size:13px;color:{TEXT}">{winner_txt}</div>',
                unsafe_allow_html=True,
            )

            # KPI row: grid-connected optimum
            gk1, gk2, gk3, gk4, gk5 = st.columns(5)
            for col, lbl, val, sub in [
                (gk1, "Grid sLCOE",    f"${gr_slcoe:.2f}", "$/MWh · constrained min"),
                (gk2, "Solar",         f"{gr_s:.0f} MW",   "DC nameplate"),
                (gk3, "BESS",          f"{gr_b:.0f} MWh",  "4-hour Li-ion"),
                (gk4, "Gas Backup",    f"{gr_g:.0f} MW",   "grid fills remaining gap"),
                (gk5, "Grid Import",   f"{gr_import/1e3:.1f} GWh/yr",
                 f"{gr_import/gr_dem*100:.1f}% of annual demand"),
            ]:
                col.markdown(_kpi(lbl, val, sub), unsafe_allow_html=True)

            # Side-by-side cost breakdown
            _section("Cost Breakdown — Microgrid vs Grid-Connected")
            dem_yr = opt["demand_mwh_yr"]

            # Microgrid components ($/MWh)
            mg_sol = opt["solar_cost_usd_yr"] / dem_yr
            mg_bes = opt["bess_cost_usd_yr"]  / dem_yr
            mg_gas = opt["gas_cost_usd_yr"]   / dem_yr

            # Grid-connected components ($/MWh)
            gr_sol  = g_opt["solar_cost_usd_yr"]          / dem_yr
            gr_bes  = g_opt["bess_cost_usd_yr"]           / dem_yr
            gr_gas  = g_opt["gas_cost_usd_yr"]            / dem_yr
            gr_gfix = g_opt["grid_fixed_cost_usd_yr"]     / dem_yr
            gr_gen  = g_opt["grid_energy_cost_usd_yr"]    / dem_yr

            fig_cmp = go.Figure()
            cats   = ["Microgrid", "Grid-Connected"]
            for label, mg_v, gr_v, color in [
                ("Solar",          mg_sol, gr_sol,  C_SOLAR),
                ("BESS",           mg_bes, gr_bes,  C_BESS),
                ("Gas",            mg_gas, gr_gas,  C_GAS),
                ("Grid (fixed)",   0,      gr_gfix, C_GRID),
                ("Grid (energy)",  0,      gr_gen,  "#34d399"),
            ]:
                fig_cmp.add_trace(go.Bar(
                    name=label, x=cats, y=[mg_v, gr_v],
                    marker_color=color, marker_line_width=0,
                    hovertemplate=f"{label}: $%{{y:.2f}}/MWh<extra></extra>",
                ))

            fig_cmp.update_layout(
                barmode="stack", height=260,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=0, b=0, l=0, r=0),
                yaxis=dict(
                    tickprefix="$", ticksuffix="/MWh",
                    tickfont=dict(color=MUTED, size=10),
                    gridcolor=BORDER, zeroline=False,
                ),
                xaxis=dict(tickfont=dict(color=TEXT, size=12), gridcolor="rgba(0,0,0,0)"),
                legend=dict(
                    orientation="h", x=0, y=-0.18,
                    font=dict(color=MUTED, size=10),
                    bgcolor="rgba(0,0,0,0)",
                ),
                font=dict(family="Inter"),
                bargap=0.35,
            )
            st.plotly_chart(fig_cmp, use_container_width=True,
                            config={"displayModeBar": False})

            # Grid-connected energy mix (dispatch)
            try:
                gd = _dispatch_grid_at(
                    iso_id,
                    float(g_opt["S_mw"]),
                    float(g_opt["B_mwh"]),
                    float(g_opt["G_mw"]),
                )
                gr_solar_pct = gd["solar_share_pct"]
                gr_bess_pct  = gd["bess_share_pct"]
                gr_gas_pct   = gd["gas_share_pct"]
                gr_grid_pct  = gd["grid_share_pct"]
            except Exception:
                gr_solar_pct = gr_ren * 0.65
                gr_bess_pct  = gr_ren * 0.35
                gr_gas_pct   = (100 - gr_ren) * 0.4
                gr_grid_pct  = (100 - gr_ren) * 0.6

            mix_col, _ = st.columns([2, 3])
            with mix_col:
                fig_mix = go.Figure(go.Pie(
                    labels=["Solar", "BESS", "Gas", "Grid Import"],
                    values=[gr_solar_pct, gr_bess_pct, gr_gas_pct, gr_grid_pct],
                    hole=0.68,
                    marker_colors=[C_SOLAR, C_BESS, C_GAS, C_GRID],
                    marker_line=dict(color=BG, width=2),
                    textinfo="none",
                    hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
                ))
                fig_mix.add_annotation(
                    text=f"<b>{gr_ren:.0f}%</b><br>renewable",
                    x=0.5, y=0.5, xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(size=14, color=TEXT, family="Inter"),
                    align="center",
                )
                fig_mix.update_layout(
                    height=180, margin=dict(t=0, b=28, l=0, r=0),
                    paper_bgcolor="rgba(0,0,0,0)", showlegend=True,
                    legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.12,
                                font=dict(size=10, color=MUTED),
                                bgcolor="rgba(0,0,0,0)"),
                    title=dict(text="Grid-Connected Energy Mix",
                               font=dict(size=10, color=MUTED),
                               x=0, pad=dict(b=10)),
                )
                st.plotly_chart(fig_mix, use_container_width=True,
                                config={"displayModeBar": False})

        st.markdown(f"""
        <div style="font-size:10px;color:{MUTED};letter-spacing:0.04em;
                    border-top:1px solid {BORDER};margin-top:48px;padding:24px 0">
          Sources: NREL ATB 2025 · Lazard LCOE+ 18.0 · EIA 2026 STEO ·
          PVWatts V8 · Open-Meteo ERA5 · Unsubsidised (pre-IRA ITC) · 2024 USD
        </div>
        """, unsafe_allow_html=True)


# ── PAGE: Methodology ────────────────────────────────────────────────────────

def _page_methodology() -> None:

    # ── helpers ──────────────────────────────────────────────────────────────
    def flow_box(number: str, title: str, body: str,
                 color: str = BORDER, accent: str = MUTED) -> str:
        return f"""
        <div style="border:1px solid {color};padding:20px 22px;
                    background:{SURFACE};position:relative">
          <div style="font-size:9px;font-weight:600;letter-spacing:0.18em;
                      text-transform:uppercase;color:{accent};margin-bottom:6px">
            Step {number}
          </div>
          <div style="font-size:14px;font-weight:500;color:{TEXT};
                      margin-bottom:8px">{title}</div>
          <div style="font-size:12px;color:{MUTED};line-height:1.8">{body}</div>
        </div>"""

    def arrow() -> str:
        return (f'<div style="text-align:center;color:{MUTED};'
                f'font-size:20px;line-height:1;padding:6px 0">↓</div>')

    def source_card(title: str, desc: str, url: str, tag: str) -> str:
        return f"""
        <div style="border:1px solid {BORDER};padding:18px 20px;
                    background:{SURFACE};margin-bottom:12px">
          <div style="display:flex;justify-content:space-between;
                      align-items:flex-start;margin-bottom:8px">
            <div style="font-size:13px;font-weight:500;color:{TEXT}">{title}</div>
            <div style="font-size:9px;font-weight:600;letter-spacing:0.12em;
                        text-transform:uppercase;color:{ACCENT};
                        background:rgba(59,130,246,0.1);padding:3px 8px;
                        white-space:nowrap;margin-left:12px">{tag}</div>
          </div>
          <div style="font-size:12px;color:{MUTED};line-height:1.7;
                      margin-bottom:12px">{desc}</div>
          <a href="{url}" target="_blank"
             style="font-size:9px;font-weight:600;letter-spacing:0.12em;
                    text-transform:uppercase;color:{MUTED};text-decoration:none;
                    border-bottom:1px solid {BORDER}">
            View Source ↗
          </a>
        </div>"""

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="padding:80px 0 0">
      <div style="font-size:9px;font-weight:600;letter-spacing:0.22em;
                  text-transform:uppercase;color:{MUTED};margin-bottom:14px">
        How It Works
      </div>
      <div style="font-size:40px;font-weight:200;color:{TEXT};
                  line-height:1.15;margin-bottom:16px;letter-spacing:-0.02em">
        Methodology
      </div>
      <div style="font-size:14px;color:{MUTED};line-height:1.75;
                  margin-bottom:56px;max-width:720px">
        We model <strong style="color:{TEXT}">two energy strategies</strong> for a
        behind-the-meter collocated datacenter and compare them across every major
        US electricity market. The <strong style="color:{C_SOLAR}">Microgrid</strong>
        operates fully islanded with Solar PV, BESS, and gas backup.
        The <strong style="color:{C_GRID}">Grid-Connected</strong> model adds a
        utility connection and dispatches gas or grid each hour based on whichever
        is cheaper at that moment. For both models we find the asset sizes that
        minimise total cost while meeting a user-defined renewable energy target.
      </div>
    </div>
    <div style="height:1px;background:{BORDER};margin-bottom:48px"></div>
    """, unsafe_allow_html=True)

    # ── Section 1: Two pipelines side-by-side ────────────────────────────────
    st.markdown(f"""
    <div style="font-size:9px;font-weight:600;letter-spacing:0.18em;
                text-transform:uppercase;color:{MUTED};margin-bottom:28px">
      Optimization Pipelines
    </div>
    """, unsafe_allow_html=True)

    mg_col, gap_col, gr_col = st.columns([5, 0.3, 5])

    with mg_col:
        st.markdown(f"""
        <div style="font-size:10px;font-weight:600;letter-spacing:0.16em;
                    text-transform:uppercase;color:{C_SOLAR};
                    border-bottom:2px solid {C_SOLAR};padding-bottom:10px;
                    margin-bottom:18px">
          Model A — Microgrid (Islanded)
        </div>
        {flow_box("1", "User Inputs",
            "IT load (MW) and minimum on-site renewable share (%) define "
            "the scope and ESG constraint.")}
        {arrow()}
        {flow_box("2", "Weather & Solar",
            "Open-Meteo ERA5 hourly temperature → PUE-scaled load. "
            "PVWatts V8 TMY → hourly solar capacity factor.",
            color=BORDER)}
        {arrow()}
        {flow_box("3", "Reliability Surface",
            "Grid search over Solar MW × BESS MWh. For each pair, dispatch "
            "determines the minimum gas MW (G_min) needed for zero unserved "
            "energy. This enforces 100% reliability.",
            color=ACCENT, accent=ACCENT)}
        {arrow()}
        {flow_box("4", "sLCOE Surface",
            "For each feasible (S, B, G_min) triplet, compute annualised "
            "Solar + BESS + Gas costs ÷ annual demand.",
            color=ACCENT, accent=ACCENT)}
        {arrow()}
        {flow_box("5", "Constrained Optimum",
            "Filter to designs meeting the renewable floor. "
            "Return the minimum-sLCOE survivor.",
            color=C_SOLAR, accent=C_SOLAR)}
        """, unsafe_allow_html=True)

    with gr_col:
        st.markdown(f"""
        <div style="font-size:10px;font-weight:600;letter-spacing:0.16em;
                    text-transform:uppercase;color:{C_GRID};
                    border-bottom:2px solid {C_GRID};padding-bottom:10px;
                    margin-bottom:18px">
          Model B — Grid-Connected
        </div>
        {flow_box("1", "Same User Inputs",
            "Same IT load and renewable floor, applied consistently "
            "so both models are directly comparable.")}
        {arrow()}
        {flow_box("2", "Weather, Solar + Grid Prices",
            "Same weather and solar as Model A, plus real 2024 hourly "
            "day-ahead LMP fetched from each ISO's public API "
            "via gridstatus.",
            color=BORDER)}
        {arrow()}
        {flow_box("3", "Grid Search over (S, B, G)",
            "Same Solar × BESS grid, but G (gas) is now a free variable "
            "from 0 MW up — grid provides unlimited backup so no minimum "
            "gas constraint is imposed.",
            color=ACCENT, accent=ACCENT)}
        {arrow()}
        {flow_box("4", "Grid sLCOE Surface",
            "For each (S, B, G) triplet, dispatch with price-optimal "
            "gas vs grid selection each hour. Compute Solar + BESS + Gas "
            "+ Grid Interconnect + Grid Energy costs ÷ annual demand.",
            color=ACCENT, accent=ACCENT)}
        {arrow()}
        {flow_box("5", "Constrained Optimum",
            "Same renewable floor filter (on-site solar+BESS only). "
            "Return minimum-sLCOE design and compare to Model A.",
            color=C_GRID, accent=C_GRID)}
        """, unsafe_allow_html=True)

    # ── Section 2: sLCOE formulas ─────────────────────────────────────────────
    st.markdown(
        f'<div style="height:1px;background:{BORDER};margin:48px 0 40px"></div>',
        unsafe_allow_html=True,
    )

    st.markdown(f"""
    <div style="font-size:9px;font-weight:600;letter-spacing:0.18em;
                text-transform:uppercase;color:{MUTED};margin-bottom:28px">
      System LCOE Formulas
    </div>
    <div style="font-size:13px;color:{MUTED};line-height:1.8;
                margin-bottom:32px;max-width:720px">
      The <strong style="color:{TEXT}">System Levelised Cost of Energy (sLCOE)</strong>
      annualises all capital costs via a Fixed Charge Rate (FCR), adds fixed and
      variable operating costs, and divides by annual site energy demand — giving
      a single $/MWh figure directly comparable across markets and designs.
    </div>
    """, unsafe_allow_html=True)

    f1, fgap, f2 = st.columns([5, 0.3, 5])

    with f1:
        st.markdown(f"""
        <div style="font-size:10px;font-weight:600;letter-spacing:0.14em;
                    text-transform:uppercase;color:{C_SOLAR};margin-bottom:14px">
          Microgrid sLCOE
        </div>
        <div style="background:{SURFACE};border:1px solid {BORDER};
                    padding:22px;font-family:monospace;font-size:12px;
                    color:{MUTED};line-height:2.3">
          <span style="color:{C_SOLAR}">($950/kW<sub>DC</sub> × mult) × FCR + $10/kW-yr</span><br>
          + <span style="color:{C_BESS}">($280/kWh × mult) × FCR + $5/kW-yr</span><br>
          + <span style="color:{C_GAS}">($1,100/kW × mult) × FCR + $15/kW-yr</span><br>
          + <span style="color:{C_GAS}">gas_price × 9.0 MMBtu/MWh + $5/MWh</span><br>
          <div style="height:1px;background:{BORDER};margin:12px 0"></div>
          ÷ &nbsp;Annual Demand MWh
        </div>
        <div style="font-size:11px;color:{MUTED};margin-top:10px;line-height:1.7">
          <span style="color:{C_SOLAR}">■</span> Solar PV &nbsp;
          <span style="color:{C_BESS}">■</span> BESS &nbsp;
          <span style="color:{C_GAS}">■</span> Gas RICE<br>
          Gas sized to G_min — minimum capacity for zero unserved energy.
        </div>
        """, unsafe_allow_html=True)

    with f2:
        st.markdown(f"""
        <div style="font-size:10px;font-weight:600;letter-spacing:0.14em;
                    text-transform:uppercase;color:{C_GRID};margin-bottom:14px">
          Grid-Connected sLCOE
        </div>
        <div style="background:{SURFACE};border:1px solid {BORDER};
                    padding:22px;font-family:monospace;font-size:12px;
                    color:{MUTED};line-height:2.3">
          <span style="color:{C_SOLAR}">($950/kW<sub>DC</sub> × mult) × FCR + $10/kW-yr</span><br>
          + <span style="color:{C_BESS}">($280/kWh × mult) × FCR + $5/kW-yr</span><br>
          + <span style="color:{C_GAS}">($1,100/kW × mult) × FCR + $15/kW-yr</span><br>
          + <span style="color:{C_GAS}">gas_price × 9.0 MMBtu/MWh + $5/MWh</span><br>
          + <span style="color:{C_GRID}">($100/kW × mult) × FCR + $2/kW-yr</span><br>
          + <span style="color:{C_GRID}">Σ max(0, price[t]) × grid_import[t]</span><br>
          <div style="height:1px;background:{BORDER};margin:12px 0"></div>
          ÷ &nbsp;Annual Demand MWh
        </div>
        <div style="font-size:11px;color:{MUTED};margin-top:10px;line-height:1.7">
          <span style="color:{C_GRID}">■</span> Grid interconnect sized to peak import MW.<br>
          Gas is a free variable (0–200 MW); grid is the infinite backstop.
        </div>
        """, unsafe_allow_html=True)

    # Technology costs table
    st.markdown("<div style='height:36px'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:9px;font-weight:600;letter-spacing:0.16em;
                text-transform:uppercase;color:{MUTED};margin-bottom:16px">
      Technology Cost Assumptions &nbsp;·&nbsp; NREL ATB 2025 Moderate · Lazard LCOS V18 · Real 2024 USD
    </div>
    """, unsafe_allow_html=True)

    tc1, tc2, tc3, tc4 = st.columns(4)
    for col, color, title, rows in [
        (tc1, C_SOLAR, "Solar PV", [
            ("CAPEX", "$950 /kW<sub>DC</sub>"),
            ("Fixed O&M", "$10 /kW-yr"),
            ("Lifetime", "30 yr"),
            ("Degradation", "0.5% /yr"),
            ("DC:AC ratio", "1.3 fixed-tilt"),
        ]),
        (tc2, C_BESS, "BESS (4-hr AC)", [
            ("CAPEX", "$280 /kWh"),
            ("Fixed O&M", "$5 /kW-yr"),
            ("Lifetime", "15 yr"),
            ("Augmentation", "2.5% /yr"),
            ("Round-trip η", "85%"),
        ]),
        (tc3, C_GAS, "Gas RICE", [
            ("CAPEX", "$1,100 /kW"),
            ("Fixed O&M", "$15 /kW-yr"),
            ("Var O&M", "$5 /MWh"),
            ("Heat rate", "9.0 MMBtu/MWh"),
            ("Lifetime", "20 yr"),
        ]),
        (tc4, C_GRID, "Grid Interconnect", [
            ("CAPEX", "$100 /kW"),
            ("Fixed O&M", "$2 /kW-yr"),
            ("Lifetime", "25 yr"),
            ("Sizing", "Peak import MW"),
            ("", ""),
        ]),
    ]:
        rows_html = "".join(
            f'<tr><td style="color:{MUTED};padding:4px 0;font-size:11px">{k}</td>'
            f'<td style="color:{TEXT};padding:4px 0 4px 12px;font-size:11px;'
            f'text-align:right">{v}</td></tr>'
            for k, v in rows if k
        )
        col.markdown(f"""
        <div style="background:{SURFACE};border:1px solid {BORDER};
                    border-top:2px solid {color};padding:16px">
          <div style="font-size:10px;font-weight:600;letter-spacing:0.12em;
                      text-transform:uppercase;color:{color};margin-bottom:12px">{title}</div>
          <table style="width:100%;border-collapse:collapse">{rows_html}</table>
        </div>
        """, unsafe_allow_html=True)

    # ISO multipliers + gas prices
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:9px;font-weight:600;letter-spacing:0.16em;
                text-transform:uppercase;color:{MUTED};margin-bottom:16px">
      Regional CAPEX Multipliers &amp; Gas Prices
    </div>
    """, unsafe_allow_html=True)

    iso_mult_data = [
        ("ERCOT", "1.00×", "$2.50/MMBtu", "$22.50/MWh marginal"),
        ("PJM",   "1.05×", "$3.20/MMBtu", "$28.80/MWh marginal"),
        ("MISO",  "1.03×", "$3.00/MMBtu", "$27.00/MWh marginal"),
        ("CAISO", "1.15×", "$5.00/MMBtu", "$45.00/MWh marginal"),
        ("SPP",   "1.00×", "$2.80/MMBtu", "$25.20/MWh marginal"),
        ("NYISO", "1.18×", "$5.50/MMBtu", "$49.50/MWh marginal"),
        ("ISONE", "1.20×", "$6.00/MMBtu", "$54.00/MWh marginal"),
    ]
    hdr_style = (f"font-size:10px;font-weight:600;color:{MUTED};"
                 "letter-spacing:0.1em;text-transform:uppercase;padding:6px 10px")
    cell_style = f"font-size:12px;color:{TEXT};padding:6px 10px"
    muted_cell = f"font-size:11px;color:{MUTED};padding:6px 10px"
    rows_html = "".join(
        f'<tr style="border-top:1px solid {BORDER}">'
        f'<td style="{cell_style};font-weight:600">{iso}</td>'
        f'<td style="{cell_style}">{mult}</td>'
        f'<td style="{cell_style}">{gas}</td>'
        f'<td style="{muted_cell}">{note}</td>'
        f'</tr>'
        for iso, mult, gas, note in iso_mult_data
    )
    st.markdown(f"""
    <div style="background:{SURFACE};border:1px solid {BORDER};overflow:hidden">
      <table style="width:100%;border-collapse:collapse">
        <tr style="background:{BORDER}20">
          <th style="{hdr_style};text-align:left">ISO</th>
          <th style="{hdr_style};text-align:left">CAPEX Mult</th>
          <th style="{hdr_style};text-align:left">Gas Price</th>
          <th style="{hdr_style};text-align:left">Gas Marginal (fuel + $5/MWh var O&M)</th>
        </tr>
        {rows_html}
      </table>
    </div>
    """, unsafe_allow_html=True)

    # Key parameters grid
    st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
    p1, p2, p3, p4, p5, p6 = st.columns(6)
    for col, label, value in [
        (p1, "WACC",              "7%"),
        (p2, "Project Life",      f"{PROJECT_LIFE} yr"),
        (p3, "FCR",               "8.58%"),
        (p4, "Gas Heat Rate",     f"{HEAT_RATE} MMBtu/MWh"),
        (p5, "Grid Interconnect", "$100/kW"),
        (p6, "PUE",               "1.35"),
    ]:
        col.markdown(f"""
        <div style="background:{SURFACE};border:1px solid {BORDER};
                    padding:14px 16px">
          <div style="font-size:9px;letter-spacing:0.14em;text-transform:uppercase;
                      color:{MUTED};margin-bottom:6px">{label}</div>
          <div style="font-size:18px;font-weight:300;color:{TEXT}">{value}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="font-size:11px;color:{MUTED};margin-top:16px;line-height:1.8">
      FCR = WACC × (1+WACC)ⁿ / ((1+WACC)ⁿ − 1) &nbsp;·&nbsp;
      Unsubsidised (pre-IRA ITC) &nbsp;·&nbsp; All costs in real 2024 USD &nbsp;·&nbsp;
      BESS round-trip efficiency 85% &nbsp;·&nbsp; Solar DC:AC ratio 1.3, fixed-tilt
    </div>
    """, unsafe_allow_html=True)

    # ── Section 3: Dispatch logic ─────────────────────────────────────────────
    st.markdown(
        f'<div style="height:1px;background:{BORDER};margin:48px 0 40px"></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"""
    <div style="font-size:9px;font-weight:600;letter-spacing:0.18em;
                text-transform:uppercase;color:{MUTED};margin-bottom:10px">
      Hourly Dispatch Logic — 8,784 Hours per Year
    </div>
    <div style="font-size:13px;color:{MUTED};line-height:1.8;
                margin-bottom:32px;max-width:720px">
      Each hour the simulator decides how to power the datacenter.
      Both models share the same first two steps; they diverge on how to
      handle residual load after BESS.
    </div>
    """, unsafe_allow_html=True)

    da, db, dc, dd1, dd2 = st.columns(5)
    shared = [
        (C_SOLAR, "1", "Solar First",
         "Solar output (PVWatts hourly CF × installed MW) serves load "
         "directly. Zero marginal cost — always dispatched first."),
        (C_BESS, "2", "BESS Charge / Discharge",
         "Surplus solar charges the battery (up to power and SoC limits). "
         "Deficit draws the battery down to a 20% SoC floor. "
         "Temperature-adjusted round-trip efficiency applied each hour."),
    ]
    mg_only = [
        (C_GAS, "3A", "Gas Fills the Gap",
         "<em>Microgrid only.</em> All remaining unmet load goes to gas. "
         "Gas must be sized large enough to guarantee zero unserved energy "
         "across all 8,784 hours — this is the G_min constraint."),
    ]
    gr_only = [
        (C_GAS, "3B", "Gas or Grid — Cheapest Wins",
         "<em>Grid-connected only.</em> Compare gas short-run marginal cost "
         "(fuel + var O&amp;M) vs grid price[t] each hour. Cheaper source "
         "serves remaining load. Gas is capped at G_mw; grid covers any rest."),
        (C_GRID, "4", "Annual Grid Import",
         "Σ grid_import[t] logged across all hours. Used as a metric and "
         "as the basis for grid interconnection sizing (peak import MW)."),
    ]

    for col, (color, num, title, body) in zip([da, db], shared):
        col.markdown(f"""
        <div style="border:1px solid {BORDER};padding:20px 18px;height:100%">
          <div style="font-size:9px;font-weight:600;letter-spacing:0.14em;
                      text-transform:uppercase;color:{MUTED};margin-bottom:8px">
            Step {num}
          </div>
          <div style="font-size:12px;font-weight:500;color:{TEXT};
                      margin-bottom:8px">{title}</div>
          <div style="font-size:11px;color:{MUTED};line-height:1.8">{body}</div>
        </div>
        """, unsafe_allow_html=True)

    for col, (color, num, title, body) in zip([dc], mg_only):
        col.markdown(f"""
        <div style="border:1px solid {color};padding:20px 18px;height:100%">
          <div style="font-size:9px;font-weight:600;letter-spacing:0.14em;
                      text-transform:uppercase;color:{color};margin-bottom:8px">
            Step {num}
          </div>
          <div style="font-size:12px;font-weight:500;color:{TEXT};
                      margin-bottom:8px">{title}</div>
          <div style="font-size:11px;color:{MUTED};line-height:1.8">{body}</div>
        </div>
        """, unsafe_allow_html=True)

    for col, (color, num, title, body) in zip([dd1, dd2], gr_only):
        col.markdown(f"""
        <div style="border:1px solid {color};padding:20px 18px;height:100%">
          <div style="font-size:9px;font-weight:600;letter-spacing:0.14em;
                      text-transform:uppercase;color:{color};margin-bottom:8px">
            Step {num}
          </div>
          <div style="font-size:12px;font-weight:500;color:{TEXT};
                      margin-bottom:8px">{title}</div>
          <div style="font-size:11px;color:{MUTED};line-height:1.8">{body}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="font-size:11px;color:{MUTED};margin-top:14px;line-height:1.9;
                border-left:3px solid {BORDER};padding-left:16px">
      <strong style="color:{TEXT}">Renewable share</strong> is defined identically
      in both models: (solar MWh used + BESS discharge MWh) ÷ total demand MWh.
      Grid imports and gas are both counted as non-renewable, so the renewable floor
      binds on on-site clean generation only.
    </div>
    """, unsafe_allow_html=True)

    # ── Section 4: Data sources ───────────────────────────────────────────────
    st.markdown(f"""
    <div style="height:1px;background:{BORDER};margin:48px 0 32px"></div>
    <div style="font-size:9px;font-weight:600;letter-spacing:0.18em;
                text-transform:uppercase;color:{MUTED};margin-bottom:24px">
      Data Sources
    </div>
    """, unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    col_sources = [s1, s2, s3]
    sources = [
        ("NREL ATB 2025",
         "Technology capital and O&M costs for utility-scale Solar PV, "
         "Li-ion BESS, and gas RICE — moderate scenario. Grid interconnection "
         "cost proxy: $100/kW installed.",
         "https://atb.nrel.gov/", "CAPEX · O&M"),
        ("EIA 2026 Short-Term Energy Outlook",
         "Regional natural gas wellhead price forecasts used as fuel cost "
         "inputs for each ISO/RTO market.",
         "https://www.eia.gov/outlooks/steo/", "Gas Prices"),
        ("PVWatts V8 / NSRDB",
         "Hourly AC solar generation profiles for a fixed-tilt utility array "
         "at each market's representative city, using a Typical Meteorological Year.",
         "https://pvwatts.nrel.gov/", "Solar Resource"),
        ("Open-Meteo ERA5",
         "Hourly dry-bulb temperature reanalysis data for 2024, used to "
         "estimate datacenter cooling load and BESS thermal efficiency.",
         "https://open-meteo.com/", "Temperature"),
        ("ISO Public APIs — gridstatus",
         "Real 2024 hourly day-ahead LMP data fetched from ERCOT, CAISO, PJM, "
         "and NYISO via the open-source gridstatus library. MISO, ISONE, and SPP "
         "use calibrated synthetic profiles where the API was too slow.",
         "https://github.com/gridstatus/gridstatus", "Grid Prices"),
        ("Lazard LCOE+ 18.0",
         "Independent levelised cost benchmarks used to cross-validate our "
         "sLCOE results against industry consensus estimates.",
         "https://www.lazard.com/research-insights/levelized-cost-of-energyplus/",
         "Benchmarks"),
    ]
    for i, (title, desc, url, tag) in enumerate(sources):
        with col_sources[i % 3]:
            st.markdown(source_card(title, desc, url, tag),
                        unsafe_allow_html=True)


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

# ── Tab redirect via JS (must run after tabs are rendered) ────────────────────
if st.session_state.get("goto_methodology"):
    st.session_state.goto_methodology = False
    components.html("""
    <script>
    setTimeout(function() {
        try {
            var doc = window.top.document;
            var tabs = Array.from(doc.querySelectorAll('button[role="tab"]'));
            var target = tabs.find(function(t) {
                return t.innerText.trim().toUpperCase().includes('METHODOLOGY');
            });
            if (target) { target.click(); }
            else if (tabs.length > 1) { tabs[1].click(); }
        } catch(e) {}
    }, 200);
    </script>
    """, height=30)
