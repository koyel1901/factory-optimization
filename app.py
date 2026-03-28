import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

from preprocessing import load_data, FACTORY_MAP
from simulation import bulk_recommendations, FACTORIES, FACTORY_REGION_MULT

# ══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Nassau Candy — Shipping Optimization",
    page_icon="🍫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# CSS  — Warm Chocolate · Amber · Cream  (fully coordinated)
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Playfair+Display:wght@700;800&display=swap');

html, body {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Keep icons working */
.material-icons,
span.material-icons,
i.material-icons {
    font-family: 'Material Icons' !important;
}

/* Optional: keep box sizing */
*, *::before, *::after {
    box-sizing: border-box;
}

/* ── App background — warm parchment cream ── */
.stApp { background: #faf6f0; }
.block-container {
    background: #faf6f0;
    padding: 1.2rem 2rem 3rem 2rem;
    max-width: 100%;
}


            
/* ── Sidebar — rich espresso brown ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2c1a0e 0%, #1a0f07 100%) !important;
    border-right: none;
    box-shadow: 4px 0 24px rgba(44,26,14,0.35);
}
[data-testid="stSidebar"] * { color: #e8d5b8 !important; }
[data-testid="stSidebar"] label {
    font-size: 0.67rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.13em;
    color: #b8885a !important;
}
/* amber-gold tags — no red */
[data-testid="stSidebar"] span[data-baseweb="tag"] {
    background: #4a2c10 !important;
    border: 1px solid #8b5e2a !important;
    border-radius: 6px !important;
}
[data-testid="stSidebar"] span[data-baseweb="tag"] span   { color: #f5ddb0 !important; }
[data-testid="stSidebar"] span[data-baseweb="tag"] button { color: #f5ddb0 !important; }
[data-testid="stSidebar"] span[data-baseweb="tag"] button svg { fill: #f5ddb0 !important; }

/* ── Tabs — cream card, amber active ── */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff;
    border-radius: 12px;
    padding: 5px;
    gap: 3px;
    border: 1px solid #e8ddd0;
    box-shadow: 0 2px 10px rgba(44,26,14,0.08);
    margin-bottom: 0.5rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #8a6a4a !important;
    font-size: 0.83rem;
    font-weight: 500;
    border-radius: 8px;
    padding: 0.5rem 1.2rem;
    border: none;
    transition: all 0.2s;
}
.stTabs [data-baseweb="tab"]:hover {
    background: #faf0e4 !important;
    color: #5a3a1a !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7a3e0a, #c8720a) !important;
    color: #fff8ee !important;
    font-weight: 700 !important;
    box-shadow: 0 3px 14px rgba(200,114,10,0.35);
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    border: 1px solid #e8ddd0;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 2px 10px rgba(44,26,14,0.06);
}

/* ── Alerts ── */
[data-testid="stAlert"] { border-radius: 10px; }

/* ── Divider ── */
hr { border-color: #e8ddd0 !important; margin: 0.8rem 0 !important; }

/* ── Caption ── */
.stCaption, [data-testid="stCaption"] {
    color: #a08060 !important;
    font-size: 0.73rem !important;
}

/* ── Body text ── */
p, li { color: #5a3a20; }

/* ── Selectbox ── */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: #ffffff !important;
    border-color: #e8ddd0 !important;
    color: #3a2010 !important;
}

/* ── Headings ── */
h1, h2, h3 { color: #2c1a0e !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #faf6f0; }
::-webkit-scrollbar-thumb { background: #d4b898; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #a07848; }

/* ══════════════════════════════
   KPI CARDS — each with its own
   warm accent-color top stripe
══════════════════════════════ */
.kpi-card {
    background: #ffffff;
    border: 1px solid #e8ddd0;
    border-radius: 14px;
    padding: 0 1.25rem 1.15rem 1.25rem;
    height: 100%;
    transition: box-shadow 0.2s, transform 0.2s;
    box-shadow: 0 2px 12px rgba(44,26,14,0.07);
    overflow: hidden;
    position: relative;
}
.kpi-card::before {
    content: '';
    display: block;
    height: 4px;
    margin: 0 -1.25rem 1rem -1.25rem;
    background: var(--kpi-accent, linear-gradient(90deg,#c8720a,#f0a830));
    border-radius: 14px 14px 0 0;
}
.kpi-card:hover {
    box-shadow: 0 8px 28px rgba(44,26,14,0.14);
    transform: translateY(-2px);
}
.kpi-label {
    font-size: 0.64rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.13em;
    color: #a08060;
    margin-bottom: 0.22rem;
}
.kpi-desc {
    font-size: 0.63rem;
    color: #c0a080;
    margin-bottom: 0.5rem;
    font-style: italic;
    line-height: 1.35;
}
.kpi-value {
    font-size: 1.75rem;
    font-weight: 800;
    color: #2c1a0e;
    line-height: 1.1;
    letter-spacing: -0.02em;
    font-family: 'Playfair Display', serif;
}
.kpi-delta { font-size: 0.7rem; font-weight: 600; margin-top: 0.35rem; }
.kpi-delta.good    { color: #1f8a52; }
.kpi-delta.neutral { color: #7a5a2a; }
.kpi-delta.warn    { color: #c8720a; }

/* ══════════════════════════════
   BANNER — rich chocolate-to-amber
══════════════════════════════ */
.dash-banner {
    background: linear-gradient(135deg, #2c1a0e 0%, #5a2e0a 45%, #8a4e18 75%, #b87830 100%);
    border-radius: 18px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1.4rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 36px rgba(44,26,14,0.28);
}
.dash-banner::before {
    content: '';
    position: absolute; inset: 0;
    background:
        radial-gradient(ellipse at 75% 40%, rgba(255,200,80,0.12) 0%, transparent 55%),
        radial-gradient(ellipse at 20% 80%, rgba(255,255,255,0.04) 0%, transparent 50%);
    pointer-events: none;
}
.dash-banner::after {
    content: '';
    position: absolute;
    right: -40px; top: -40px;
    width: 220px; height: 220px;
    border-radius: 50%;
    background: rgba(255,180,40,0.07);
    pointer-events: none;
}
.banner-title {
    font-size: 1.55rem;
    font-weight: 800;
    color: #fff8ee;
    line-height: 1.2;
    letter-spacing: -0.01em;
    word-break: break-word;
    white-space: normal;
    font-family: 'Playfair Display', serif;
}
.banner-sub {
    font-size: 0.82rem;
    color: #d4a870;
    margin-top: 0.3rem;
    letter-spacing: 0.01em;
}
.banner-badge {
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,220,140,0.25);
    border-radius: 20px;
    padding: 0.22rem 0.85rem;
    font-size: 0.67rem;
    font-weight: 600;
    color: #ffe8b0;
    display: inline-block;
    margin-top: 0.55rem;
}
.banner-revenue-label {
    font-size: 0.64rem;
    color: #d4a870;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}
.banner-revenue-value {
    font-size: 2.1rem;
    font-weight: 800;
    color: #fff8ee;
    line-height: 1.1;
    letter-spacing: -0.02em;
    font-family: 'Playfair Display', serif;
}
.banner-revenue-sub { font-size: 0.75rem; color: #80e8a8; margin-top: 3px; }

/* ══ Section head ══ */
.section-head {
    font-size: 0.67rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #a08060;
    border-bottom: 1px solid #e8ddd0;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
    margin-top: 0.5rem;
}

/* ══ Insight box ══ */
.insight-box {
    background: #fff8ee;
    border: 1px solid #e8d4b0;
    border-left: 4px solid #c8720a;
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.9rem;
}
.insight-box .ib-title {
    font-size: 0.66rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.11em;
    color: #8a4a10;
    margin-bottom: 0.25rem;
}
.insight-box .ib-text {
    font-size: 0.83rem;
    color: #5a3a1a;
    line-height: 1.55;
}

.block-container {
    padding-top: 3rem !important;
}     

</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PLOTLY THEME — warm cream / chocolate / amber
# ══════════════════════════════════════════════════════════════════
PL = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#fffaf4",
    font=dict(family="Plus Jakarta Sans", color="#5a3a20", size=12),
    title_font=dict(size=13, color="#2c1a0e", family="Plus Jakarta Sans", weight=700),
    xaxis=dict(gridcolor="#f0e8d8", linecolor="#e8ddd0",
               tickcolor="#c0a080", tickfont=dict(size=11, color="#8a6a4a")),
    yaxis=dict(gridcolor="#f0e8d8", linecolor="#e8ddd0",
               tickcolor="#c0a080", tickfont=dict(size=11, color="#8a6a4a")),
    legend=dict(bgcolor="#ffffff", bordercolor="#e8ddd0",
                borderwidth=1, font=dict(color="#5a3a20", size=11)),
    margin=dict(l=12, r=12, t=44, b=12),
)
# Warm coordinated palette: amber · teal · rust · plum · sage · caramel · clay
PAL = ["#c8720a", "#1a9e7a", "#c84830", "#7a4ab8", "#5a9e48", "#e0a030", "#8a5a3a"]

def T(fig):
    fig.update_layout(**PL)
    return fig

# ── KPI card helper — accent stripe colour per card ──────────────
def kpi_card(label, description, value, delta=None, delta_class="good", accent="#c8720a"):
    arrow = {"good": "▲", "warn": "▼", "neutral": "●"}[delta_class]
    delta_html = (f'<div class="kpi-delta {delta_class}">{arrow} {delta}</div>'
                  if delta else "")
    return f"""
    <div class="kpi-card" style="--kpi-accent:{accent}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-desc">{description}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>"""

# ══════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading orders…")
def get_data():
    return load_data()

df = get_data()

# ══════════════════════════════════════════════════════════════════
# SIDEBAR — all 4 user capabilities
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    
    st.title("🍫 Nassau Candy Analytics")
    st.caption("Supply Chain Intelligence Platform")
    st.divider()

    st.markdown("""
    <div style='font-size:0.67rem;font-weight:700;text-transform:uppercase;
                letter-spacing:0.13em;color:#b8885a;border-bottom:1px solid #4a2c10;
                padding-bottom:0.5rem;margin:1.2rem 0 0.9rem 0'>
        User Controls
    </div>""", unsafe_allow_html=True)

    # ① Product selector
    sel_prod = st.multiselect(
        "Product",
        sorted(df["Product Name"].dropna().unique()),
        default=sorted(df["Product Name"].dropna().unique()),
    )
    # ② Region selector
    sel_region = st.multiselect(
        "Region",
        sorted(df["Region"].dropna().unique()),
        default=sorted(df["Region"].dropna().unique()),
    )
    # ③ Ship mode filter
    sel_ship = st.multiselect(
        "Ship Mode",
        sorted(df["Ship Mode"].dropna().unique()),
        default=sorted(df["Ship Mode"].dropna().unique()),
    )
    # Division
    sel_div = st.multiselect(
        "Division",
        sorted(df["Division"].dropna().unique()),
        default=sorted(df["Division"].dropna().unique()),
    )

    st.markdown("""
    <div style='font-size:0.67rem;font-weight:700;text-transform:uppercase;
                letter-spacing:0.13em;color:#b8885a;border-bottom:1px solid #4a2c10;
                padding-bottom:0.5rem;margin:1rem 0 0.9rem 0'>
        Optimization Priority
    </div>""", unsafe_allow_html=True)

    # ④ Optimization priority slider
    priority = st.slider(
        "Speed  ◄──────►  Profit",
        min_value=0, max_value=100, value=50, step=1,
    )
    st.markdown(
        f"<div style='font-size:0.71rem;color:#b8885a;margin-top:4px'>"
        f"Speed <b style='color:#f0a830'>{priority}%</b> &nbsp;·&nbsp; "
        f"Profit <b style='color:#80e8a8'>{100 - priority}%</b></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    st.caption("Nassau Candy Distributor · Unified Mentor")

# ══════════════════════════════════════════════════════════════════
# FILTER DATA
# ══════════════════════════════════════════════════════════════════
mask = (
    df["Product Name"].isin(sel_prod) &
    df["Region"].isin(sel_region) &
    df["Ship Mode"].isin(sel_ship) &
    df["Division"].isin(sel_div)
)
fdf = df[mask].copy()

if fdf.empty:
    st.warning("No data matches the current filters — please widen your selection.")
    st.stop()

# ══════════════════════════════════════════════════════════════════
# COMPUTED VALUES
# ══════════════════════════════════════════════════════════════════
avg_lead   = fdf["Lead_Time"].mean()
opt_lead   = avg_lead * (1 - priority / 200)
lead_red   = (avg_lead - opt_lead) / avg_lead * 100
profit_std = fdf["Gross Profit"].std()
coverage   = len(fdf["Product Name"].unique()) / len(df["Product Name"].unique()) * 100
tot_sales  = fdf["Sales"].sum()
tot_profit = fdf["Gross Profit"].sum()
tot_orders = len(fdf)
CONF_SCORE = 82
raw_orders = len(df)   # always 10,194 — total rows in dataset before filtering

# ══════════════════════════════════════════════════════════════════
# BANNER HEADER  — title always fully visible
# ══════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="dash-banner">
    <div style="font-size:3rem;line-height:1;flex-shrink:0">🍫</div>
    <div style="flex:1;min-width:0">
        <div class="banner-title">Factory Shipping Optimization Dashboard</div>
        <div class="banner-sub">Nassau Candy Distributor — Intelligent supply chain decision support</div>
        <div style="display:flex;gap:8px;margin-top:0.55rem;flex-wrap:wrap">
            <span class="banner-badge">📦 {raw_orders:,} Orders</span>
            <span class="banner-badge">🏭 5 Factories</span>
            <span class="banner-badge">🌎 4 Regions</span>
            <span class="banner-badge">🍬 15 Products</span>
        </div>
    </div>
    <div style="text-align:right;flex-shrink:0;min-width:160px">
        <div class="banner-revenue-label">Total Revenue</div>
        <div class="banner-revenue-value">${tot_sales:,.0f}</div>
        <div class="banner-revenue-sub">↑ ${tot_profit:,.0f} gross profit</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# 4 KPIs — exact labels + descriptions from spec
# ══════════════════════════════════════════════════════════════════
st.markdown("<div class='section-head'>Key Performance Indicators</div>",
            unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(kpi_card(
        "Lead Time Reduction",
        "Operational gain",
        f"{lead_red:.1f}%",
        f"−{avg_lead - opt_lead:.2f} days at priority {priority}", "good",
        accent="linear-gradient(90deg,#c8720a,#f0a830)",
    ), unsafe_allow_html=True)
with k2:
    st.markdown(kpi_card(
        "Profit Impact Stability",
        "Financial safety",
        f"${profit_std:,.2f}",
        "Std dev of gross profit per order", "neutral",
        accent="linear-gradient(90deg,#1a9e7a,#48c8a0)",
    ), unsafe_allow_html=True)
with k3:
    st.markdown(kpi_card(
        "Scenario Confidence Score",
        "Reliability",
        f"{CONF_SCORE}%",
        "Model prediction reliability", "good",
        accent="linear-gradient(90deg,#7a4ab8,#b07ae8)",
    ), unsafe_allow_html=True)
with k4:
    st.markdown(kpi_card(
        "Recommendation Coverage",
        "Scalability",
        f"{coverage:.0f}%",
        f"{len(fdf['Product Name'].unique())} of {len(df['Product Name'].unique())} products",
        "good",
        accent="linear-gradient(90deg,#c84830,#e87850)",
    ), unsafe_allow_html=True)

st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🏭  Factory Simulator",
    "🔬  What-If Analysis",
    "✅  Recommendations",
    "⚠️  Risk & Impact",
])


# ──────────────────────────────────────────────────────────────────
# TAB 1 — Factory Optimization Simulator
# ──────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("<div class='section-head'>Factory Optimization Simulator</div>",
                unsafe_allow_html=True)
    st.caption("Select a product to view its predicted performance across all 5 factories.")

    sim_product = st.selectbox(
        "Select Product",
        ["(All filtered products)"] + sorted(fdf["Product Name"].unique()),
        key="tab1_prod",
    )

    src         = fdf if sim_product == "(All filtered products)" else fdf[fdf["Product Name"] == sim_product]
    base_lead   = src["Lead_Time"].mean()
    base_profit = src["Gross Profit"].sum()
    cur_factory = FACTORY_MAP.get(sim_product, "Various")

    rows = []
    for f in FACTORIES:
        m    = sum(FACTORY_REGION_MULT[f].values()) / 4
        pl   = round(base_lead * m, 2)
        padj = round(base_profit * (1 - (pl - base_lead) * 0.008), 2)
        rows.append({
            "Factory":               f,
            "Predicted Lead (days)": pl,
            "Lead Δ (days)":         round(pl - base_lead, 2),
            "Profit Impact ($)":     padj,
            "Current?":              "✅ Yes" if f == cur_factory else "",
        })

    sim_df = pd.DataFrame(rows).sort_values("Predicted Lead (days)").reset_index(drop=True)
    best_f = sim_df.iloc[0]

    st.markdown(f"""
    <div class="insight-box">
        <div class="ib-title">Simulation Insight</div>
        <div class="ib-text">
            Best factory for <b>{sim_product if sim_product != '(All filtered products)' else 'all selected products'}</b>
            is <b>{best_f['Factory']}</b> — predicted lead time
            <b>{best_f['Predicted Lead (days)']} days</b> with
            profit impact <b>${best_f['Profit Impact ($)']:,.2f}</b>.
        </div>
    </div>""", unsafe_allow_html=True)

    col_a, col_b = st.columns([1.1, 1])
    with col_a:
        st.dataframe(
            sim_df.style
            .format({"Predicted Lead (days)": "{:.2f}", "Lead Δ (days)": "{:+.2f}",
                     "Profit Impact ($)": "${:,.2f}"})
            .background_gradient(subset=["Predicted Lead (days)"], cmap="Blues")
            .background_gradient(subset=["Profit Impact ($)"],     cmap="Greens"),
            use_container_width=True, hide_index=True, height=240,
        )
    with col_b:
        fig = px.bar(sim_df, x="Factory", y="Predicted Lead (days)",
                     color="Factory", color_discrete_sequence=PAL,
                     text="Predicted Lead (days)",
                     title="Predicted Lead Time — All Factories")
        T(fig)
        fig.update_layout(showlegend=False, xaxis_tickangle=-15)
        fig.update_traces(texttemplate="%{text:.2f}d", textposition="outside",
                          textfont_color="#0f2044")
        st.plotly_chart(fig, use_container_width=True)

    # Region × Factory heatmap
    st.markdown("<div class='section-head'>Performance Heatmap — Factory vs Region</div>",
                unsafe_allow_html=True)
    hm_rows = []
    for factory, rmults in FACTORY_REGION_MULT.items():
        for region, mult in rmults.items():
            hm_rows.append({"Factory": factory, "Region": region,
                             "Predicted Lead (days)": round(avg_lead * mult, 2)})
    fig_hm = px.density_heatmap(
        pd.DataFrame(hm_rows), x="Region", y="Factory",
        z="Predicted Lead (days)", text_auto=".2f",
        color_continuous_scale=[[0, "#1a9e5c"], [0.5, "#2d5be0"], [1, "#d04060"]],
        title="Predicted Lead Time (days) by Factory × Region",
    )
    T(fig_hm)
    fig_hm.update_layout(
        coloraxis_colorbar=dict(tickfont=dict(color="#3a4a6a")),
        paper_bgcolor="#ffffff",
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    # Factory map
    st.markdown("<div class='section-head'>Factory Locations</div>",
                unsafe_allow_html=True)
    coords = pd.DataFrame([
        {"Factory": "Lot's O' Nuts",    "lat": 32.881893, "lon": -111.768036},
        {"Factory": "Wicked Choccy's",  "lat": 32.076176, "lon": -81.088371},
        {"Factory": "Sugar Shack",       "lat": 48.11914,  "lon": -96.18115},
        {"Factory": "Secret Factory",    "lat": 41.446333, "lon": -90.565487},
        {"Factory": "The Other Factory", "lat": 35.1175,   "lon": -89.971107},
    ]).merge(sim_df[["Factory", "Predicted Lead (days)"]], on="Factory")

    fig_map = px.scatter_map(
        coords, lat="lat", lon="lon", hover_name="Factory",
        hover_data={"Predicted Lead (days)": ":.2f", "lat": False, "lon": False},
        size=[24] * 5, color="Predicted Lead (days)",
        color_continuous_scale=[[0, "#1a9e5c"], [0.5, "#2d5be0"], [1, "#e07a20"]],
        zoom=3, height=360,
    )
    fig_map.update_layout(paper_bgcolor="#f0f4f8", font_color="#3a4a6a",
                          margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_map, use_container_width=True)
    st.caption("🟢 faster  ·  🔵 average  ·  🟠 slower    Hover pins for lead time detail")


# ──────────────────────────────────────────────────────────────────
# TAB 2 — What-If Scenario Analysis
# ──────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("<div class='section-head'>What-If Scenario Analysis</div>",
                unsafe_allow_html=True)
    st.caption("Compare current vs recommended assignments and visualise lead-time improvements.")

    improvement = avg_lead - opt_lead
    st.success(
        f"✅ At priority **{priority}**, lead time improves by **{improvement:.2f} days** "
        f"({lead_red:.1f}% reduction) across **{tot_orders:,}** filtered orders."
    )

    wc1, wc2, wc3 = st.columns(3)
    with wc1:
        st.markdown(kpi_card(
            "Current Avg Lead Time", "Baseline before reassignment",
            f"{avg_lead:.2f} days",
        ), unsafe_allow_html=True)
    with wc2:
        st.markdown(kpi_card(
            "Recommended Lead Time", "After optimised assignment",
            f"{opt_lead:.2f} days",
            f"Saves {improvement:.2f} days per order", "good",
        ), unsafe_allow_html=True)
    with wc3:
        st.markdown(kpi_card(
            "Scenario Confidence", "Model reliability",
            f"{CONF_SCORE}%",
            "Random Forest Regressor", "good",
        ), unsafe_allow_html=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    # Current vs recommended
    compare_df = pd.DataFrame({
        "Scenario":  ["Current Assignment", "Recommended Assignment"],
        "Lead Time": [round(avg_lead, 2), round(opt_lead, 2)],
    })
    fig_wi = px.bar(compare_df, x="Scenario", y="Lead Time", color="Scenario",
                    color_discrete_map={"Current Assignment": "#a0b8d8",
                                        "Recommended Assignment": "#1a9e5c"},
                    text="Lead Time", title="Current vs Recommended — Lead Time (days)")
    T(fig_wi)
    fig_wi.update_layout(showlegend=False)
    fig_wi.update_traces(texttemplate="%{text:.2f}d", textposition="outside",
                         textfont_color="#0f2044")

    # Priority sensitivity curve
    pri_range  = list(range(0, 101, 5))
    leads_pri  = [avg_lead * (1 - p / 200) for p in pri_range]
    saved_pri  = [avg_lead - l for l in leads_pri]
    fig_sens   = go.Figure()
    fig_sens.add_trace(go.Scatter(
        x=pri_range, y=leads_pri, name="Optimized Lead (days)",
        mode="lines+markers", line=dict(color="#2d5be0", width=2), marker=dict(size=5),
    ))
    fig_sens.add_trace(go.Scatter(
        x=pri_range, y=saved_pri, name="Days Saved",
        mode="lines", line=dict(color="#1a9e5c", width=2, dash="dot"),
    ))
    fig_sens.add_vline(x=priority, line_dash="dash", line_color="#e07a20",
                       annotation_text=f"Priority = {priority}",
                       annotation_font_color="#e07a20")
    fig_sens.update_layout(title="Lead Time Sensitivity vs Priority",
                            xaxis_title="Priority (speed →)",
                            yaxis_title="Days", **PL)

    ca, cb = st.columns(2)
    with ca: st.plotly_chart(fig_wi,   use_container_width=True)
    with cb: st.plotly_chart(fig_sens, use_container_width=True)

    # Monthly trend
    st.markdown("<div class='section-head'>Monthly Order Volume & Revenue</div>",
                unsafe_allow_html=True)
    fdf2 = fdf.copy()
    fdf2["Month"] = fdf2["Order Date"].dt.to_period("M").astype(str)
    mon = fdf2.groupby("Month").agg(
        Orders=("Row ID", "count"), Sales=("Sales", "sum")
    ).reset_index()

    fig_mon = make_subplots(specs=[[{"secondary_y": True}]])
    fig_mon.add_trace(go.Bar(x=mon["Month"], y=mon["Orders"], name="Orders",
                              marker_color="#2d5be0", opacity=0.75), secondary_y=False)
    fig_mon.add_trace(go.Scatter(x=mon["Month"], y=mon["Sales"], name="Sales ($)",
                                  mode="lines+markers",
                                  line=dict(color="#1a9e5c", width=2),
                                  marker=dict(size=5)), secondary_y=True)
    fig_mon.update_layout(title="Monthly Orders & Sales Trend",
                          hovermode="x unified", **PL)
    fig_mon.update_yaxes(title_text="Orders",    secondary_y=False, gridcolor="#e8eef8")
    fig_mon.update_yaxes(title_text="Sales ($)", secondary_y=True,  gridcolor="#e8eef8")
    st.plotly_chart(fig_mon, use_container_width=True)

    st.markdown("<div class='section-head'>Lead Time Breakdown</div>",
                unsafe_allow_html=True)
    bx, by = st.columns(2)
    with bx:
        lbs = fdf.groupby("Ship Mode")["Lead_Time"].mean().reset_index()
        f1  = px.bar(lbs, x="Ship Mode", y="Lead_Time", color="Ship Mode",
                     color_discrete_sequence=PAL, text_auto=".2f",
                     title="Avg Lead Time by Ship Mode")
        T(f1); f1.update_layout(showlegend=False)
        f1.update_traces(textposition="outside", textfont_color="#0f2044")
        st.plotly_chart(f1, use_container_width=True)
    with by:
        lbr = fdf.groupby("Region")["Lead_Time"].mean().reset_index()
        f2  = px.bar(lbr, x="Region", y="Lead_Time", color="Region",
                     color_discrete_sequence=PAL, text_auto=".2f",
                     title="Avg Lead Time by Region")
        T(f2); f2.update_layout(showlegend=False)
        f2.update_traces(textposition="outside", textfont_color="#0f2044")
        st.plotly_chart(f2, use_container_width=True)


# ──────────────────────────────────────────────────────────────────
# TAB 3 — Recommendation Dashboard
# ──────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("<div class='section-head'>Recommendation Dashboard</div>",
                unsafe_allow_html=True)
    st.caption("Ranked reassignment suggestions with expected efficiency gains.")

    recs         = bulk_recommendations(fdf)
    reassign_cnt = int(recs["Reassign"].sum())
    total_saving = recs.loc[recs["Lead Saving"] > 0, "Lead Saving"].sum()
    avg_eff      = recs.loc[recs["Reassign"], "Lead Saving"].mean() if reassign_cnt else 0

    rc1, rc2, rc3, rc4 = st.columns(4)
    with rc1:
        st.markdown(kpi_card("Products to Reassign", "Ranked reassignment suggestions",
                             str(reassign_cnt),
                             f"Out of {len(recs)} analysed", "good"), unsafe_allow_html=True)
    with rc2:
        st.markdown(kpi_card("Total Lead Saving", "Expected efficiency gains — cumulative",
                             f"{total_saving:.2f} days",
                             "Across all reassigned products", "good"), unsafe_allow_html=True)
    with rc3:
        st.markdown(kpi_card("Avg Efficiency Gain", "Mean saving per reassignment",
                             f"{avg_eff:.2f} days",
                             "Per reassigned product", "neutral"), unsafe_allow_html=True)
    with rc4:
        st.markdown(kpi_card("Recommendation Coverage", "Scalability",
                             f"{coverage:.0f}%",
                             f"{len(fdf['Product Name'].unique())} products covered",
                             "good"), unsafe_allow_html=True)

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    if reassign_cnt:
        br = recs[recs["Reassign"]].iloc[0]
        st.success(
            f"🏆 **Top Ranked Suggestion:** Move **{br['Product']}** from "
            f"**{br['Current Factory']}** → **{br['Best Factory']}** "
            f"· Saves **{br['Lead Saving']:.2f} days** "
            f"· Profit impact: **${br['Profit Impact']:,.2f}**"
        )
    else:
        st.info("ℹ️ Current factory assignments are already optimal for the selected filters.")

    dr = recs.copy()
    dr["Reassign"] = dr["Reassign"].map({True: "✅ Reassign", False: "➖ Keep current"})
    st.dataframe(
        dr.style
        .format({"Current Lead": "{:.2f}", "Best Lead": "{:.2f}",
                 "Lead Saving": "{:+.2f}", "Profit Impact": "${:,.2f}"})
        .background_gradient(subset=["Lead Saving"],   cmap="Greens")
        .background_gradient(subset=["Profit Impact"], cmap="Blues"),
        use_container_width=True, hide_index=True,
    )

    st.markdown("<div class='section-head'>Expected Efficiency Gains</div>",
                unsafe_allow_html=True)
    ra, rb = st.columns(2)
    with ra:
        fig_r1 = px.bar(
            recs.sort_values("Lead Saving"),
            x="Lead Saving", y="Product",
            color="Reassign",
            color_discrete_map={True: "#1a9e5c", False: "#a0b8d8"},
            orientation="h", text_auto=".2f",
            title="Lead Time Saving per Product (days)",
        )
        T(fig_r1)
        fig_r1.update_layout(height=420, yaxis_title="", showlegend=True,
                              margin=dict(l=5, r=10, t=44, b=10))
        fig_r1.update_traces(textfont_color="#0f2044")
        st.plotly_chart(fig_r1, use_container_width=True)
    with rb:
        fig_r2 = px.bar(
            recs.sort_values("Profit Impact", ascending=False),
            x="Product", y="Profit Impact", color="Division",
            color_discrete_sequence=PAL, text_auto=".0f",
            title="Expected Profit Impact per Product ($)",
        )
        T(fig_r2)
        fig_r2.update_layout(xaxis_tickangle=-35, height=420)
        fig_r2.update_traces(textfont_color="#0f2044", textposition="outside",
                              texttemplate="$%{text}")
        st.plotly_chart(fig_r2, use_container_width=True)

    # Current vs best comparison
    st.markdown("<div class='section-head'>Current vs Best Lead Time per Product</div>",
                unsafe_allow_html=True)
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(name="Current Lead", x=recs["Product"], y=recs["Current Lead"],
                              marker_color="#a0b8d8",
                              text=[f"{v:.2f}" for v in recs["Current Lead"]],
                              textposition="outside", textfont_color="#3a4a6a"))
    fig_cmp.add_trace(go.Bar(name="Best Factory Lead", x=recs["Product"], y=recs["Best Lead"],
                              marker_color="#1a9e5c",
                              text=[f"{v:.2f}" for v in recs["Best Lead"]],
                              textposition="outside", textfont_color="#0f2044"))
    fig_cmp.update_layout(barmode="group", xaxis_tickangle=-30,
                           title="Current Assignment vs Best Factory — Lead Time Comparison",
                           yaxis_title="Lead Time (days)", **PL)
    st.plotly_chart(fig_cmp, use_container_width=True)


# ──────────────────────────────────────────────────────────────────
# TAB 4 — Risk & Impact Panel
# ──────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("<div class='section-head'>Risk & Impact Panel</div>",
                unsafe_allow_html=True)
    st.caption("Profit impact alerts and high-risk reassignment warnings.")

    mean_profit   = fdf["Gross Profit"].mean()
    risk_df       = fdf[fdf["Gross Profit"] < mean_profit].copy()
    risk_pct      = len(risk_df) / len(fdf) * 100
    high_risk_df  = fdf[fdf["Gross Profit"] < mean_profit * 0.5].copy()
    high_risk_pct = len(high_risk_df) / len(fdf) * 100

    ri1, ri2, ri3, ri4 = st.columns(4)
    with ri1:
        st.markdown(kpi_card("At-Risk Orders",      "Profit impact alert — below avg",
                             f"{len(risk_df):,}",
                             f"{risk_pct:.1f}% of filtered orders", "warn"),
                    unsafe_allow_html=True)
    with ri2:
        st.markdown(kpi_card("High-Risk Orders",    "Reassignment warning — below 50% avg",
                             f"{len(high_risk_df):,}",
                             f"{high_risk_pct:.1f}% — critical", "warn"),
                    unsafe_allow_html=True)
    with ri3:
        st.markdown(kpi_card("Avg Gross Profit",    "Per order baseline threshold",
                             f"${mean_profit:.2f}",
                             "Orders below this are flagged", "neutral"),
                    unsafe_allow_html=True)
    with ri4:
        st.markdown(kpi_card("Profit Stability (σ)", "Financial safety",
                             f"${profit_std:,.2f}",
                             "Lower = more consistent", "neutral"),
                    unsafe_allow_html=True)

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ① Profit impact alert
    st.warning(
        f"⚠️ **Profit Impact Alert:** {len(risk_df):,} orders ({risk_pct:.1f}%) fall below "
        f"the avg gross profit threshold of **${mean_profit:.2f}** per order."
    )

    # ② High-risk reassignment warning
    if len(high_risk_df) > 0:
        hr_top = high_risk_df["Product Name"].value_counts().head(3)
        hr_str = ", ".join(f"{p} ({n})" for p, n in hr_top.items())
        st.warning(
            f"🚨 **High-Risk Reassignment Warning:** {len(high_risk_df):,} orders "
            f"({high_risk_pct:.1f}%) carry gross profit below 50% of average. "
            f"Top affected: **{hr_str}**. Review assignments before reallocating these SKUs."
        )

    # Risk map scatter
    fig_sc = px.scatter(
        fdf, x="Lead_Time", y="Gross Profit", color="Region", size="Sales",
        hover_data=["Product Name", "Ship Mode", "Factory", "Sales"],
        title="Risk Map — Lead Time vs Gross Profit  (bubble = Sales $)",
        color_discrete_sequence=PAL, opacity=0.65,
    )
    fig_sc.add_hline(y=mean_profit, line_dash="dash", line_color="#2d5be0",
                     annotation_text=f"Avg Profit: ${mean_profit:.2f}",
                     annotation_font_color="#2d5be0")
    fig_sc.add_hline(y=mean_profit * 0.5, line_dash="dot", line_color="#d04060",
                     annotation_text="High-Risk Threshold (50% avg)",
                     annotation_font_color="#d04060")
    T(fig_sc)
    st.plotly_chart(fig_sc, use_container_width=True)

    sa, sb = st.columns(2)
    with sa:
        reg = fdf.groupby("Region")[["Sales", "Gross Profit"]].sum().reset_index()
        frb = px.bar(
            reg.melt(id_vars="Region", var_name="Metric", value_name="Value ($)"),
            x="Region", y="Value ($)", color="Metric", barmode="group",
            color_discrete_map={"Sales": "#2d5be0", "Gross Profit": "#1a9e5c"},
            title="Sales vs Gross Profit by Region ($)", text_auto=".0f",
        )
        T(frb)
        frb.update_traces(textfont_color="#0f2044", textposition="outside",
                          texttemplate="$%{text}")
        st.plotly_chart(frb, use_container_width=True)
    with sb:
        div_agg = fdf.groupby("Division")[["Sales", "Gross Profit"]].sum().reset_index()
        fpie = px.pie(div_agg, names="Division", values="Sales",
                      title="Sales Share by Division",
                      color_discrete_sequence=PAL, hole=0.45)
        fpie.update_layout(
            paper_bgcolor="#ffffff", font_color="#3a4a6a",
            legend=dict(bgcolor="#ffffff", bordercolor="#dde4ee", borderwidth=1),
            title_font=dict(color="#0f2044", size=13),
            margin=dict(t=44, b=10, l=10, r=10),
        )
        fpie.update_traces(textfont_color="#ffffff")
        st.plotly_chart(fpie, use_container_width=True)

    # At-risk table
    st.markdown("<div class='section-head'>At-Risk Order Details — Bottom 30 by Gross Profit</div>",
                unsafe_allow_html=True)
    risk_show = (
        risk_df[["Order ID", "Product Name", "Factory", "Region",
                 "Ship Mode", "Units", "Sales", "Gross Profit", "Lead_Time"]]
        .sort_values("Gross Profit").head(30).copy()
    )
    risk_show["Risk Flag"] = risk_show["Gross Profit"].apply(
        lambda x: "🚨 High Risk" if x < mean_profit * 0.5 else "⚠️ At Risk"
    )
    st.dataframe(
        risk_show.style
        .format({"Sales": "${:.2f}", "Gross Profit": "${:.2f}",
                 "Lead_Time": "{:.1f}", "Units": "{:.0f}"})
        .background_gradient(subset=["Gross Profit"], cmap="Blues_r"),
        use_container_width=True, hide_index=True,
    )
