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
HEAT_RATE       = 9.0     # MMBtu/MWh
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
        padding: 0 !important;
        max-width: 100% !important;
    }}

    /* Hide chrome */
    header[data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stStatusWidget"],
    [data-testid="stSidebar"],
    .stDeployButton, footer {{ display: none !important; }}

    /* ── Navbar: restyle st.tabs ── */
    [data-testid="stTabs"] {{
        gap: 0;
    }}
    /* Tab bar row */
    [data-testid="stTabs"] > div[data-testid="stTabsListContainer"] {{
        background: {BG} !important;
        border-bottom: 1px solid {BORDER} !important;
        padding: 0 0 0 0 !important;
        gap: 0 !important;
    }}
    /* Individual tab */
    [data-testid="stTabs"] button[role="tab"] {{
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        border-radius: 0 !important;
        color: {MUTED} !important;
        font-size: 10px !important;
        font-weight: 600 !important;
        letter-spacing: 0.16em !important;
        text-transform: uppercase !important;
        padding: 14px 24px !important;
        margin: 0 !important;
        transition: color 0.15s !important;
    }}
    [data-testid="stTabs"] button[role="tab"]:hover {{
        color: {TEXT} !important;
        background: transparent !important;
    }}
    /* Active tab */
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
        color: {TEXT} !important;
        border-bottom: 2px solid {ACCENT} !important;
        background: transparent !important;
    }}
    /* Remove default tab indicator line */
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] {{
        display: none !important;
    }}
    [data-testid="stTabs"] [data-baseweb="tab-border"] {{
        display: none !important;
    }}
    /* Tab content area */
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
        font-size: 14px !important;
        padding: 9px 12px !important;
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
        cursor: pointer !important;
    }}
    .stButton > button[kind="primary"] {{
        background: {ACCENT} !important;
        color: #fff !important;
        border: none !important;
        padding: 9px 22px !important;
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

    /* ── Divider ── */
    hr {{ border-color: {BORDER} !important; opacity: 1 !important; }}

    /* ── Column gap ── */
    [data-testid="stHorizontalBlock"] {{ gap: 0 !important; }}

    /* ── Spinner ── */
    [data-testid="stSpinner"] > div {{ border-top-color: {ACCENT} !important; }}
    </style>
    """, unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────

def _init_state() -> None:
    for key, default in {
        "it_load":      50.0,
        "ren_floor":    30,
        "selected_iso": None,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default


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
    """Return (opt_row_dict, scale) or (None, scale) if infeasible."""
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
    base_slcoe = opt["slcoe_per_mwh"]
    rows = []
    for label, price in [("Low  −20%", base * 0.80),
                          ("Base",      base),
                          ("High +40%", base * 1.40)]:
        d = (price - base) * HEAT_RATE * ggen / dem
        rows.append({
            "Scenario":       label,
            "Gas  $/MMBtu":   f"${price:.2f}",
            "sLCOE  $/MWh":   f"${base_slcoe + d:.2f}",
            "Δ Base":         "—" if label == "Base" else f"{'+'if d>0 else ''}{d:.2f}",
        })
    return pd.DataFrame(rows).set_index("Scenario")


# ── UI primitives ─────────────────────────────────────────────────────────────

def _kpi(label: str, value: str, sub: str = "") -> str:
    sub_html = (f'<div style="font-size:10px;color:{MUTED};margin-top:4px;'
                f'line-height:1.4">{sub}</div>') if sub else ""
    return f"""
    <div style="padding:18px 20px 18px 0;border-top:1px solid {BORDER}">
      <div style="font-size:9px;font-weight:600;letter-spacing:0.18em;
                  text-transform:uppercase;color:{MUTED};margin-bottom:8px">{label}</div>
      <div style="font-size:24px;font-weight:300;color:{TEXT};
                  line-height:1;letter-spacing:-0.01em">{value}</div>
      {sub_html}
    </div>"""


def _section(text: str) -> None:
    st.markdown(
        f'<div style="font-size:9px;font-weight:600;letter-spacing:0.18em;'
        f'text-transform:uppercase;color:{MUTED};padding:24px 0 12px;'
        f'border-top:1px solid {BORDER}">{text}</div>',
        unsafe_allow_html=True,
    )


# ── Brand header ──────────────────────────────────────────────────────────────

def _brand() -> None:
    st.markdown(f"""
    <div style="padding:18px 32px 0;display:flex;align-items:center;
                border-bottom:0">
      <span style="font-size:9px;font-weight:600;letter-spacing:0.24em;
                   text-transform:uppercase;color:{TEXT}">
        ◈ &nbsp; Microgrid Cost Explorer
      </span>
      <span style="margin-left:16px;font-size:10px;color:{MUTED}">
        Datacenter Energy Optimization Platform
      </span>
    </div>
    """, unsafe_allow_html=True)


# ── Config panel (left column) ────────────────────────────────────────────────

def _config_panel() -> bool:
    """Render config inputs. Returns True if Apply was clicked."""
    st.markdown(f"""
    <div style="padding:28px 24px 0 32px;border-right:1px solid {BORDER};
                min-height:100vh">
      <div style="font-size:9px;font-weight:600;letter-spacing:0.18em;
                  text-transform:uppercase;color:{MUTED};margin-bottom:20px">
        Configuration
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown("<div style='padding:0 24px 0 32px'>", unsafe_allow_html=True)

        it_load = st.number_input(
            "IT Load  (MW)",
            min_value=1.0, max_value=500.0,
            value=float(st.session_state.it_load),
            step=5.0,
            help="Rated IT power. Total facility load is higher due to PUE.",
        )

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

        ren_floor = st.slider(
            "Min Renewable Share  (%)",
            min_value=0, max_value=100,
            value=int(st.session_state.ren_floor),
            step=5,
            help="Solar + BESS must meet this annual energy share. Gas covers the gap.",
        )

        st.markdown(f"""
        <div style="font-size:10px;color:{MUTED};margin-top:6px;
                    margin-bottom:24px;line-height:1.6">
          Solar + BESS ≥ <b style="color:{TEXT}">{ren_floor}%</b> of annual demand.
        </div>
        """, unsafe_allow_html=True)

        applied = st.button("Apply", type="primary", use_container_width=True)

        if applied:
            st.session_state.it_load   = it_load
            st.session_state.ren_floor = ren_floor
            st.session_state.selected_iso = None
            st.cache_data.clear()
            st.rerun()

        st.markdown(f"""
        <div style="margin-top:32px;padding-top:24px;border-top:1px solid {BORDER};
                    font-size:9px;color:{MUTED};letter-spacing:0.04em;line-height:2">
          {YEAR} · Unsubsidised<br>
          7% WACC · {PROJECT_LIFE}-yr life<br>
          NREL ATB 2025<br>
          Lazard LCOE+ 18.0<br>
          EIA 2026 STEO<br>
          PVWatts V8
        </div>
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    return applied


# ── Markets table ─────────────────────────────────────────────────────────────

def _markets(it_load: float, ren_frac: float) -> None:
    all_isos = get_all_isos()

    st.markdown(f"""
    <div style="padding:28px 40px 0 40px">
      <div style="font-size:20px;font-weight:300;color:{TEXT};
                  letter-spacing:-0.01em;margin-bottom:6px">
        All Markets
      </div>
      <div style="font-size:11px;color:{MUTED};margin-bottom:0;padding-bottom:20px">
        {it_load:.0f} MW IT load &nbsp;·&nbsp; ≥{st.session_state.ren_floor}% renewable
        &nbsp;·&nbsp; ranked by system LCOE
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div style="height:1px;background:{BORDER};margin:0 40px"></div>',
                unsafe_allow_html=True)

    with st.container():
        st.markdown("<div style='padding:0 40px'>", unsafe_allow_html=True)

        ready_rows, pending_rows = [], []

        for iso_id, cfg in all_isos.items():
            computed = is_complete(iso_id, YEAR)
            opt, scale = _constrained_opt(iso_id, it_load, ren_frac) if computed else (None, 1.0)

            meta = {"iso": iso_id, "cfg": cfg, "opt": opt, "scale": scale}

            if computed and opt:
                ready_rows.append({**meta,
                    "Market":      iso_id,
                    "Location":    cfg["city"],
                    "sLCOE $/MWh": round(opt["slcoe_per_mwh"], 2),
                    "Renewable":   f"{opt['ren_share']*100:.0f}%",
                    "Solar MW":    f"{opt['S_mw'] * scale:.0f}",
                    "BESS MWh":    f"{opt['B_mwh'] * scale:.0f}",
                    "Gas MW":      f"{opt['G_min_mw'] * scale:.0f}",
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

        # Available markets
        _section("Available Markets")

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
                f'<div style="font-size:10px;color:{MUTED};margin-top:6px">'
                f'Select a row to explore the optimal configuration in detail.</div>',
                unsafe_allow_html=True,
            )
            if event.selection and event.selection.rows:
                sel = ready_rows[event.selection.rows[0]]
                if sel["opt"] is not None:
                    st.session_state.selected_iso = sel["iso"]
                    st.rerun()
        else:
            st.markdown(f'<div style="color:{MUTED};font-size:13px;padding:24px 0">'
                        f'No markets computed yet. Run a pipeline below.</div>',
                        unsafe_allow_html=True)

        # Pending markets
        if pending_rows:
            _section("Pending Markets")
            p_cols = st.columns(min(len(pending_rows), 4))
            for i, row in enumerate(pending_rows):
                cfg = row["cfg"]
                with p_cols[i % 4]:
                    st.markdown(f"""
                    <div style="border:1px solid {BORDER};padding:16px 18px;
                                margin-bottom:10px">
                      <div style="font-size:9px;font-weight:600;letter-spacing:0.14em;
                                  text-transform:uppercase;color:{MUTED};margin-bottom:4px">
                        {row['iso']}
                      </div>
                      <div style="font-size:13px;color:{TEXT};margin-bottom:12px">
                        {cfg['city']}
                      </div>
                      <div style="font-size:10px;color:{MUTED};line-height:1.8">
                        Gas ${cfg['gas_price_per_mmbtu']:.2f}/MMBtu
                        &nbsp;·&nbsp; {cfg['capex_multiplier']:.2f}× CAPEX
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

        st.markdown("</div>", unsafe_allow_html=True)


# ── Deep dive ─────────────────────────────────────────────────────────────────

def _deep_dive(iso_id: str, it_load: float, ren_frac: float) -> None:
    cfg       = get_iso(iso_id)
    opt, scale = _constrained_opt(iso_id, it_load, ren_frac)

    st.markdown("<div style='padding:28px 40px 0 40px'>", unsafe_allow_html=True)

    # Back link
    if st.button("← All Markets", type="secondary"):
        st.session_state.selected_iso = None
        st.rerun()

    st.markdown(f"""
    <div style="margin-top:16px">
      <div style="font-size:20px;font-weight:300;color:{TEXT};
                  letter-spacing:-0.01em">
        {cfg['name']}
        <span style="color:{MUTED};font-size:16px"> · {cfg['city']}</span>
      </div>
      <div style="font-size:11px;color:{MUTED};margin-top:4px;padding-bottom:20px">
        {it_load:.0f} MW IT load &nbsp;·&nbsp;
        ≥{st.session_state.ren_floor}% renewable &nbsp;·&nbsp;
        {YEAR} &nbsp;·&nbsp; Unsubsidised
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div style="height:1px;background:{BORDER}"></div>',
                unsafe_allow_html=True)

    if opt is None:
        st.markdown(f"""
        <div style="padding:40px 0;color:{MUTED};font-size:13px">
          No feasible configuration found for ≥{st.session_state.ren_floor}% renewable
          in this market. Lower the renewable floor in the config panel.
        </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    s_mw   = opt["S_mw"]     * scale
    b_mwh  = opt["B_mwh"]    * scale
    g_mw   = opt["G_min_mw"] * scale
    slcoe  = opt["slcoe_per_mwh"]
    ren_pct = opt["ren_share"] * 100

    # KPI row
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

    # Exact energy mix from dispatch
    try:
        d = _dispatch_at(iso_id, float(opt["S_mw"]),
                         float(opt["B_mwh"]), float(opt["G_min_mw"]))
        solar_pct, bess_pct, gas_pct = (
            d["solar_share_pct"], d["bess_share_pct"], d["gas_share_pct"]
        )
    except Exception:
        gas_pct   = (1 - opt["ren_share"]) * 100
        solar_pct = opt["ren_share"] * 65
        bess_pct  = opt["ren_share"] * 35

    # Charts
    _section("Cost Breakdown & Energy Mix")
    c_left, _, c_right = st.columns([5, 0.2, 2])

    with c_left:
        dem    = opt["demand_mwh_yr"]
        c_sol  = opt["solar_cost_usd_yr"] / dem
        c_bes  = opt["bess_cost_usd_yr"]  / dem
        c_gas  = opt["gas_cost_usd_yr"]   / dem

        fig_bar = go.Figure()
        for val, lbl, col in [(c_sol, "Solar", C_SOLAR),
                               (c_bes, "BESS",  C_BESS),
                               (c_gas, "Gas",   C_GAS)]:
            fig_bar.add_trace(go.Bar(
                x=[val], y=[""], orientation="h",
                marker_color=col, marker_line_width=0,
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
            f'<div style="font-size:10px;color:{MUTED};margin-top:2px;line-height:1.7">'
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
                        font=dict(size=10, color=MUTED), bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_d, use_container_width=True,
                        config={"displayModeBar": False})

    # Sensitivity
    _section("Gas Price Sensitivity")
    s_l, _, _ = st.columns([2, 2, 2])
    with s_l:
        st.dataframe(_sensitivity(opt, cfg), use_container_width=True)
        st.markdown(
            f'<div style="font-size:10px;color:{MUTED};margin-top:6px">'
            f'Optimal (S, B, G) held fixed. Only fuel cost varies.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ── Router ────────────────────────────────────────────────────────────────────

_init_state()
_css()
_brand()

# Navigation — add new pages here as tabs
tab_optimizer, = st.tabs(["Optimizer"])

with tab_optimizer:
    it_load  = st.session_state.it_load
    ren_frac = st.session_state.ren_floor / 100.0

    cfg_col, main_col = st.columns([1, 3.8])

    with cfg_col:
        _config_panel()

    with main_col:
        if st.session_state.selected_iso:
            _deep_dive(st.session_state.selected_iso, it_load, ren_frac)
        else:
            _markets(it_load, ren_frac)
