import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────── PAGE CONFIG ───────────────────────────
st.set_page_config(
    page_title="DeliverIQ · Analytics Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── GLOBAL CSS ───────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Root variables ── */
:root {
    --bg-base:    #0a0e1a;
    --bg-card:    #111827;
    --bg-card2:   #1a2235;
    --accent1:    #7c3aed;
    --accent2:    #06b6d4;
    --accent3:    #f59e0b;
    --accent4:    #10b981;
    --accent5:    #f43f5e;
    --text-main:  #e2e8f0;
    --text-muted: #94a3b8;
    --border:     rgba(124,58,237,0.25);
    --glow1:      rgba(124,58,237,0.15);
    --glow2:      rgba(6,182,212,0.12);
}

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
    background-color: var(--bg-base) !important;
    color: var(--text-main) !important;
}
.main .block-container { padding: 1.5rem 2rem 3rem !important; }

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0d0a2e 0%, #1a0533 40%, #0a1628 80%, #051520 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.8rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 0 60px var(--glow1), 0 0 30px var(--glow2);
}
.hero-banner::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 20% 50%, rgba(124,58,237,0.18) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 50%, rgba(6,182,212,0.12) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-size: 3rem; font-weight: 800; letter-spacing: -1px;
    background: linear-gradient(90deg, #a78bfa, #38bdf8, #34d399, #fbbf24);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0 0 0.4rem;
}
.hero-sub { color: var(--text-muted); font-size: 1.1rem; font-weight: 300; margin: 0; }

/* ── KPI Cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    position: relative; overflow: hidden;
    transition: transform .2s, box-shadow .2s;
}
.kpi-card:hover { transform: translateY(-4px); box-shadow: 0 10px 40px var(--glow1); }
.kpi-card::after {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    border-radius: 14px 14px 0 0;
}
.kpi-card.c1::after { background: linear-gradient(90deg,#7c3aed,#a78bfa); }
.kpi-card.c2::after { background: linear-gradient(90deg,#06b6d4,#38bdf8); }
.kpi-card.c3::after { background: linear-gradient(90deg,#f59e0b,#fbbf24); }
.kpi-card.c4::after { background: linear-gradient(90deg,#10b981,#34d399); }
.kpi-icon { font-size: 1.8rem; margin-bottom: .3rem; }
.kpi-value { font-size: 2rem; font-weight: 800; line-height: 1; margin-bottom: .1rem; }
.kpi-label { font-size: .8rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; }
.kpi-delta { font-size: .75rem; margin-top: .4rem; }

/* ── Section Headers ── */
.section-header {
    display: flex; align-items: center; gap: .7rem;
    margin: 2rem 0 1rem; padding-bottom: .6rem;
    border-bottom: 1px solid var(--border);
}
.section-header .dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: linear-gradient(135deg,#7c3aed,#06b6d4);
    box-shadow: 0 0 10px rgba(124,58,237,0.6);
    flex-shrink: 0;
}
.section-header h2 { margin: 0; font-size: 1.25rem; font-weight: 700; color: var(--text-main); }

/* ── Prediction Panel ── */
.pred-card {
    background: linear-gradient(145deg, var(--bg-card), #0f1d35);
    border: 1px solid rgba(6,182,212,0.3);
    border-radius: 18px; padding: 2rem;
    box-shadow: 0 0 30px rgba(6,182,212,0.08);
}
.pred-result {
    background: linear-gradient(135deg, rgba(124,58,237,0.15), rgba(6,182,212,0.1));
    border: 1px solid rgba(124,58,237,0.4);
    border-radius: 14px; padding: 1.5rem; text-align: center;
    margin-top: 1rem;
}
.pred-value { font-size: 3.5rem; font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #38bdf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.pred-unit { font-size: 1rem; color: var(--text-muted); }

/* ── Insight Cards ── */
.insight-row { display: flex; gap: 1rem; margin: 1rem 0; flex-wrap: wrap; }
.insight-chip {
    background: var(--bg-card2); border: 1px solid var(--border);
    border-radius: 10px; padding: .6rem 1rem;
    font-size: .85rem; color: var(--text-muted);
}
.insight-chip b { color: var(--text-main); }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0c111f !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem !important; }

/* ── Plotly containers ── */
.js-plotly-plot { border-radius: 14px !important; }
div[data-testid="stPlotlyChart"] { border-radius: 14px; overflow: hidden; }

/* ── Tabs ── */
[data-testid="stTabs"] > div > div { gap: .5rem !important; }
[data-testid="stTab"] {
    font-weight: 600 !important; font-size: .9rem !important;
    border-radius: 10px 10px 0 0 !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important; border-bottom: none !important;
}
[data-testid="stTab"][aria-selected="true"] {
    background: linear-gradient(180deg,rgba(124,58,237,0.25),transparent) !important;
    border-color: rgba(124,58,237,0.5) !important; color: #a78bfa !important;
}

/* ── Selectbox / Slider ── */
div[data-baseweb="select"] > div {
    background: var(--bg-card) !important;
    border-color: var(--border) !important; border-radius: 10px !important;
}
div[data-testid="stSlider"] .rc-slider-track { background: var(--accent1) !important; }

/* ── Metric overrides ── */
div[data-testid="stMetric"] {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 12px; padding: .8rem 1rem;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--accent1); border-radius: 3px; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── HELPERS ───────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,24,39,0.6)",
    font=dict(family="Outfit, sans-serif", color="#94a3b8", size=12),
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(bgcolor="rgba(0,0,0,0)", font_color="#94a3b8"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
)
PALETTE = ["#7c3aed","#06b6d4","#f59e0b","#10b981","#f43f5e","#a78bfa","#38bdf8","#fbbf24","#34d399","#fb7185"]

import os

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "Food_Delivery_Times (1).csv")
    df = pd.read_csv(file_path)
    df_clean = df.dropna().copy()
    return df, df_clean
@st.cache_resource
def train_model(df_clean):
    le_weather = LabelEncoder(); le_traffic = LabelEncoder()
    le_time    = LabelEncoder(); le_vehicle  = LabelEncoder()
    df = df_clean.copy()
    df["Weather_enc"]  = le_weather.fit_transform(df["Weather"])
    df["Traffic_enc"]  = le_traffic.fit_transform(df["Traffic_Level"])
    df["Time_enc"]     = le_time.fit_transform(df["Time_of_Day"])
    df["Vehicle_enc"]  = le_vehicle.fit_transform(df["Vehicle_Type"])
    feats = ["Distance_km","Weather_enc","Traffic_enc","Time_enc","Vehicle_enc",
             "Preparation_Time_min","Courier_Experience_yrs"]
    X, y = df[feats], df["Delivery_Time_min"]
    model = GradientBoostingRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    r2  = r2_score(y, model.predict(X))
    mae = mean_absolute_error(y, model.predict(X))
    return model, le_weather, le_traffic, le_time, le_vehicle, feats, r2, mae

def apply_layout(fig, title=""):
    fig.update_layout(**PLOTLY_LAYOUT, title=dict(text=title, font=dict(color="#e2e8f0", size=14)))
    return fig

# ─────────────────────────── LOAD ───────────────────────────
df_raw, df = load_data()
model, le_w, le_t, le_tod, le_v, feats, r2, mae = train_model(df)

# ─────────────────────────── SIDEBAR ───────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 1.5rem'>
      <div style='font-size:2.4rem'>🚀</div>
      <div style='font-size:1.3rem;font-weight:800;
           background:linear-gradient(90deg,#a78bfa,#38bdf8);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;
           background-clip:text'>DeliverIQ</div>
      <div style='font-size:.75rem;color:#64748b;margin-top:.2rem'>Food Delivery Intelligence</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("**🎛️ Filters**")
    weather_opts  = ["All"] + sorted(df["Weather"].unique().tolist())
    traffic_opts  = ["All"] + sorted(df["Traffic_Level"].unique().tolist())
    vehicle_opts  = ["All"] + sorted(df["Vehicle_Type"].unique().tolist())
    time_opts     = ["All"] + sorted(df["Time_of_Day"].unique().tolist())

    sel_weather = st.selectbox("☁️ Weather",   weather_opts)
    sel_traffic = st.selectbox("🚦 Traffic",   traffic_opts)
    sel_vehicle = st.selectbox("🛵 Vehicle",   vehicle_opts)
    sel_time    = st.selectbox("🕐 Time of Day", time_opts)
    dist_range  = st.slider("📍 Distance (km)", float(df.Distance_km.min()), float(df.Distance_km.max()),
                            (float(df.Distance_km.min()), float(df.Distance_km.max())), 0.5)
    st.divider()
    st.markdown(f"<div style='font-size:.8rem;color:#64748b'>📊 {len(df_raw)} orders total<br>✅ {len(df)} after cleaning</div>", unsafe_allow_html=True)

# ── Apply filters ──
dff = df.copy()
if sel_weather != "All": dff = dff[dff.Weather == sel_weather]
if sel_traffic != "All": dff = dff[dff.Traffic_Level == sel_traffic]
if sel_vehicle != "All": dff = dff[dff.Vehicle_Type == sel_vehicle]
if sel_time    != "All": dff = dff[dff.Time_of_Day == sel_time]
dff = dff[(dff.Distance_km >= dist_range[0]) & (dff.Distance_km <= dist_range[1])]

# ─────────────────────────── HERO ───────────────────────────
st.markdown("""
<div class='hero-banner'>
  <div class='hero-title'>🚀 DeliverIQ Analytics</div>
  <p class='hero-sub'>AI-powered food delivery intelligence · Predict · Explore · Optimise</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────── KPI CARDS ───────────────────────────
avg_t   = dff.Delivery_Time_min.mean()
med_t   = dff.Delivery_Time_min.median()
avg_d   = dff.Distance_km.mean()
fast_c  = (dff.Delivery_Time_min < 30).sum()

st.markdown(f"""
<div class='kpi-grid'>
  <div class='kpi-card c1'>
    <div class='kpi-icon'>⏱️</div>
    <div class='kpi-value' style='color:#a78bfa'>{avg_t:.1f}</div>
    <div class='kpi-label'>Avg Delivery (min)</div>
    <div class='kpi-delta' style='color:#10b981'>● Median {med_t:.1f} min</div>
  </div>
  <div class='kpi-card c2'>
    <div class='kpi-icon'>📍</div>
    <div class='kpi-value' style='color:#38bdf8'>{avg_d:.2f}</div>
    <div class='kpi-label'>Avg Distance (km)</div>
    <div class='kpi-delta' style='color:#94a3b8'>● Max {dff.Distance_km.max():.1f} km</div>
  </div>
  <div class='kpi-card c3'>
    <div class='kpi-icon'>⚡</div>
    <div class='kpi-value' style='color:#fbbf24'>{fast_c}</div>
    <div class='kpi-label'>Fast Orders < 30 min</div>
    <div class='kpi-delta' style='color:#10b981'>● {fast_c/len(dff)*100:.1f}% of filtered</div>
  </div>
  <div class='kpi-card c4'>
    <div class='kpi-icon'>🤖</div>
    <div class='kpi-value' style='color:#34d399'>{r2:.3f}</div>
    <div class='kpi-label'>Model R² Score</div>
    <div class='kpi-delta' style='color:#94a3b8'>● MAE {mae:.1f} min</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────── TABS ───────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Overview",
    "🔬  Deep Dive",
    "🤖  Predictor",
    "🧠  Model Insights",
])

# ══════════════════════════════════ TAB 1 ══════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'><div class='dot'></div><h2>Delivery Time Distribution</h2></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        # Histogram with KDE overlay
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=dff.Delivery_Time_min, nbinsx=35,
            marker=dict(color="#7c3aed", opacity=.7,
                        line=dict(color="rgba(167,139,250,0.4)", width=.5)),
            name="Count",
        ))
        # Add a smooth density overlay
        from scipy.stats import gaussian_kde
        kde_x = np.linspace(dff.Delivery_Time_min.min(), dff.Delivery_Time_min.max(), 200)
        kde_y = gaussian_kde(dff.Delivery_Time_min)(kde_x)
        kde_y_scaled = kde_y * len(dff) * (dff.Delivery_Time_min.max()-dff.Delivery_Time_min.min()) / 35
        fig.add_trace(go.Scatter(
            x=kde_x, y=kde_y_scaled, mode="lines",
            line=dict(color="#06b6d4", width=2.5), name="Density"
        ))
        fig.add_vline(x=avg_t, line_dash="dot", line_color="#f59e0b",
                      annotation_text=f"Mean {avg_t:.1f}", annotation_font_color="#f59e0b")
        fig = apply_layout(fig, "Delivery Time Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Donut by vehicle type
        vc = dff.Vehicle_Type.value_counts()
        fig2 = go.Figure(go.Pie(
            labels=vc.index, values=vc.values,
            hole=.55,
            marker=dict(colors=["#7c3aed","#06b6d4","#f59e0b"],
                        line=dict(color="rgba(0,0,0,0.3)", width=2)),
        ))
        fig2.update_traces(textinfo="label+percent", textfont_size=12, textfont_color="#e2e8f0")
        fig2.add_annotation(text=f"<b>{len(dff)}</b><br><span style='font-size:10px'>Orders</span>",
                            showarrow=False, font=dict(size=15, color="#e2e8f0"))
        fig2 = apply_layout(fig2, "Orders by Vehicle Type")
        st.plotly_chart(fig2, use_container_width=True)

    # Row 2
    c3, c4 = st.columns(2)
    with c3:
        avg_by_weather = dff.groupby("Weather").Delivery_Time_min.mean().sort_values(ascending=False)
        fig3 = go.Figure(go.Bar(
            x=avg_by_weather.index, y=avg_by_weather.values,
            marker=dict(
                color=avg_by_weather.values,
                colorscale=[[0,"#06b6d4"],[0.5,"#7c3aed"],[1,"#f43f5e"]],
                showscale=False,
                line=dict(color="rgba(0,0,0,0.3)", width=1),
            ),
            text=[f"{v:.1f}" for v in avg_by_weather.values],
            textposition="outside", textfont=dict(color="#e2e8f0", size=11),
        ))
        fig3 = apply_layout(fig3, "Avg Delivery Time by Weather")
        fig3.update_xaxes(tickfont_color="#94a3b8")
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        avg_by_traffic = dff.groupby("Traffic_Level").Delivery_Time_min.mean()
        colors_traffic = {"Low":"#10b981","Medium":"#f59e0b","High":"#f43f5e"}
        fig4 = go.Figure(go.Bar(
            x=avg_by_traffic.index,
            y=avg_by_traffic.values,
            marker_color=[colors_traffic.get(k,"#7c3aed") for k in avg_by_traffic.index],
            text=[f"{v:.1f} min" for v in avg_by_traffic.values],
            textposition="outside", textfont=dict(color="#e2e8f0", size=11),
        ))
        fig4 = apply_layout(fig4, "Avg Delivery Time by Traffic Level")
        st.plotly_chart(fig4, use_container_width=True)

    # Time of day heatmap (Traffic × Time)
    st.markdown("<div class='section-header'><div class='dot'></div><h2>Traffic × Time of Day Heatmap</h2></div>", unsafe_allow_html=True)
    pivot = dff.groupby(["Traffic_Level","Time_of_Day"]).Delivery_Time_min.mean().unstack(fill_value=0)
    fig5 = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns, y=pivot.index,
        colorscale=[[0,"#0a0e1a"],[0.3,"#1e1b4b"],[0.6,"#7c3aed"],[0.85,"#f59e0b"],[1,"#f43f5e"]],
        text=np.round(pivot.values,1), texttemplate="%{text} min",
        textfont=dict(size=12, color="white"),
        showscale=True,
        colorbar=dict(tickfont=dict(color="#94a3b8")),
    ))
    fig5 = apply_layout(fig5, "Average Delivery Time (min): Traffic Level × Time of Day")
    st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════ TAB 2 ══════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'><div class='dot'></div><h2>Distance vs Delivery Time</h2></div>", unsafe_allow_html=True)

    c1, c2 = st.columns([3,2])
    with c1:
        fig6 = px.scatter(
            dff, x="Distance_km", y="Delivery_Time_min",
            color="Traffic_Level",
            color_discrete_map={"Low":"#10b981","Medium":"#f59e0b","High":"#f43f5e"},
            trendline="ols",
            size="Preparation_Time_min",
            hover_data=["Weather","Vehicle_Type","Courier_Experience_yrs"],
            opacity=.7,
        )
        fig6.update_traces(marker=dict(line=dict(color="rgba(0,0,0,0.3)",width=.5)))
        fig6 = apply_layout(fig6, "Distance vs Delivery Time (sized by Prep Time)")
        st.plotly_chart(fig6, use_container_width=True)

    with c2:
        # Box plots by time of day
        tod_order = ["Morning","Afternoon","Evening","Night"]
        tod_present = [t for t in tod_order if t in dff.Time_of_Day.unique()]
        fig7 = go.Figure()
        for i, tod in enumerate(tod_present):
            sub = dff[dff.Time_of_Day == tod].Delivery_Time_min
            fig7.add_trace(go.Box(
                y=sub, name=tod,
                marker_color=PALETTE[i],
                boxmean=True, fillcolor=f"rgba({','.join(str(int(PALETTE[i].lstrip('#')[j:j+2],16)) for j in (0,2,4))},0.2)",
                line_color=PALETTE[i],
            ))
        fig7 = apply_layout(fig7, "Delivery Time by Time of Day")
        st.plotly_chart(fig7, use_container_width=True)

    # Courier experience vs delivery time
    st.markdown("<div class='section-header'><div class='dot'></div><h2>Courier Experience Effect</h2></div>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        exp_avg = dff.groupby("Courier_Experience_yrs").Delivery_Time_min.mean().reset_index()
        fig8 = go.Figure()
        fig8.add_trace(go.Scatter(
            x=exp_avg.Courier_Experience_yrs, y=exp_avg.Delivery_Time_min,
            mode="lines+markers",
            line=dict(color="#a78bfa", width=3),
            marker=dict(size=8, color="#7c3aed", line=dict(color="#a78bfa",width=2)),
            fill="tozeroy", fillcolor="rgba(124,58,237,0.1)",
        ))
        fig8 = apply_layout(fig8, "Avg Delivery Time vs Courier Experience (yrs)")
        fig8.update_xaxes(title="Experience (years)")
        fig8.update_yaxes(title="Avg Delivery Time (min)")
        st.plotly_chart(fig8, use_container_width=True)

    with c4:
        # Vehicle × Weather matrix
        vw = dff.groupby(["Vehicle_Type","Weather"]).Delivery_Time_min.mean().unstack(fill_value=0)
        fig9 = go.Figure(go.Heatmap(
            z=vw.values, x=vw.columns, y=vw.index,
            colorscale=[[0,"#0a1628"],[0.5,"#06b6d4"],[1,"#7c3aed"]],
            text=np.round(vw.values,1), texttemplate="%{text}",
            textfont=dict(size=12, color="white"),
            colorbar=dict(tickfont=dict(color="#94a3b8")),
        ))
        fig9 = apply_layout(fig9, "Avg Delivery Time: Vehicle × Weather")
        st.plotly_chart(fig9, use_container_width=True)

    # Preparation time distribution by vehicle
    st.markdown("<div class='section-header'><div class='dot'></div><h2>Preparation Time Analysis</h2></div>", unsafe_allow_html=True)
    fig10 = go.Figure()
    for i, vt in enumerate(dff.Vehicle_Type.unique()):
        sub = dff[dff.Vehicle_Type == vt].Preparation_Time_min
        fig10.add_trace(go.Violin(
            y=sub, name=vt, fillcolor=f"rgba({','.join(str(int(PALETTE[i].lstrip('#')[j:j+2],16)) for j in (0,2,4))},0.25)",
            line_color=PALETTE[i], box_visible=True, meanline_visible=True, opacity=.8,
        ))
    fig10 = apply_layout(fig10, "Preparation Time Distribution by Vehicle Type")
    st.plotly_chart(fig10, use_container_width=True)


# ══════════════════════════════════ TAB 3 ══════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'><div class='dot'></div><h2>🤖 AI Delivery Time Predictor</h2></div>", unsafe_allow_html=True)

    col_form, col_gauge = st.columns([1, 1])

    with col_form:
        st.markdown("<div class='pred-card'>", unsafe_allow_html=True)
        st.markdown("**Enter Order Details**")
        distance    = st.slider("📍 Distance (km)", 0.5, 20.0, 8.0, 0.1)
        prep_time   = st.slider("🍳 Preparation Time (min)", 1, 30, 10)
        exp_yrs     = st.slider("👤 Courier Experience (yrs)", 0.0, 9.0, 4.0, 0.5)
        weather_sel = st.selectbox("☁️ Weather Condition", list(le_w.classes_))
        traffic_sel = st.selectbox("🚦 Traffic Level",     list(le_t.classes_))
        time_sel    = st.selectbox("🕐 Time of Day",       list(le_tod.classes_))
        vehicle_sel = st.selectbox("🛵 Vehicle Type",      list(le_v.classes_))
        st.markdown("</div>", unsafe_allow_html=True)

        predict_btn = st.button("🚀 Predict Delivery Time", use_container_width=True,
                                type="primary")

    with col_gauge:
        if predict_btn:
            X_in = np.array([[
                distance,
                le_w.transform([weather_sel])[0],
                le_t.transform([traffic_sel])[0],
                le_tod.transform([time_sel])[0],
                le_v.transform([vehicle_sel])[0],
                prep_time,
                exp_yrs,
            ]])
            pred = float(model.predict(X_in)[0])
            pred = max(5, pred)

            # Speed category
            if pred < 30:
                speed_label, speed_color = "⚡ Lightning Fast", "#10b981"
            elif pred < 50:
                speed_label, speed_color = "✅ On Time", "#06b6d4"
            elif pred < 75:
                speed_label, speed_color = "⚠️ Moderate", "#f59e0b"
            else:
                speed_label, speed_color = "🔴 Slow Delivery", "#f43f5e"

            st.markdown(f"""
            <div class='pred-result'>
              <div style='font-size:.9rem;color:#94a3b8;margin-bottom:.3rem'>Estimated Delivery Time</div>
              <div class='pred-value'>{pred:.0f}</div>
              <div class='pred-unit'>minutes</div>
              <div style='margin-top:.8rem;font-size:1.1rem;font-weight:700;color:{speed_color}'>{speed_label}</div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred,
                number=dict(suffix=" min", font=dict(size=28, color="#e2e8f0")),
                gauge=dict(
                    axis=dict(range=[0, 150], tickfont=dict(color="#94a3b8")),
                    bar=dict(color=speed_color, thickness=.3),
                    bgcolor="rgba(0,0,0,0)",
                    bordercolor="rgba(0,0,0,0)",
                    steps=[
                        dict(range=[0, 30],   color="rgba(16,185,129,0.15)"),
                        dict(range=[30, 50],  color="rgba(6,182,212,0.12)"),
                        dict(range=[50, 75],  color="rgba(245,158,11,0.12)"),
                        dict(range=[75, 150], color="rgba(244,63,94,0.12)"),
                    ],
                    threshold=dict(
                        line=dict(color="#a78bfa", width=3),
                        thickness=.75, value=pred,
                    ),
                ),
                domain=dict(x=[0,1], y=[0,1]),
            ))
            fig_g.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Outfit", color="#e2e8f0"),
                margin=dict(l=20,r=20,t=30,b=0),
                height=260,
            )
            st.plotly_chart(fig_g, use_container_width=True)

            # Breakdown insight chips
            st.markdown(f"""
            <div class='insight-row'>
              <div class='insight-chip'>📍 Distance: <b>{distance:.1f} km</b></div>
              <div class='insight-chip'>🍳 Prep: <b>{prep_time} min</b></div>
              <div class='insight-chip'>👤 Exp: <b>{exp_yrs} yrs</b></div>
              <div class='insight-chip'>☁️ <b>{weather_sel}</b></div>
              <div class='insight-chip'>🚦 <b>{traffic_sel}</b></div>
              <div class='insight-chip'>🛵 <b>{vehicle_sel}</b></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='display:flex;flex-direction:column;align-items:center;justify-content:center;
                 height:300px;border:1px dashed rgba(124,58,237,0.3);border-radius:14px;
                 color:#64748b;font-size:.95rem;gap:.6rem'>
              <div style='font-size:3rem'>🤖</div>
              <div>Configure parameters and hit <b style='color:#a78bfa'>Predict</b></div>
            </div>
            """, unsafe_allow_html=True)

    # Sensitivity analysis
    if predict_btn:
        st.markdown("<div class='section-header'><div class='dot'></div><h2>Sensitivity Analysis</h2></div>", unsafe_allow_html=True)
        c_s1, c_s2 = st.columns(2)
        with c_s1:
            dist_range_s = np.linspace(0.5, 20, 30)
            preds_d = []
            for d in dist_range_s:
                xi = np.array([[d, le_w.transform([weather_sel])[0], le_t.transform([traffic_sel])[0],
                                le_tod.transform([time_sel])[0], le_v.transform([vehicle_sel])[0],
                                prep_time, exp_yrs]])
                preds_d.append(float(model.predict(xi)[0]))
            fig_s1 = go.Figure()
            fig_s1.add_trace(go.Scatter(
                x=dist_range_s, y=preds_d, mode="lines",
                line=dict(color="#7c3aed", width=3),
                fill="tozeroy", fillcolor="rgba(124,58,237,0.1)",
            ))
            fig_s1.add_vline(x=distance, line_dash="dot", line_color="#f59e0b",
                             annotation_text=f"Selected {distance:.1f} km", annotation_font_color="#f59e0b")
            fig_s1 = apply_layout(fig_s1, "Delivery Time vs Distance")
            fig_s1.update_xaxes(title="Distance (km)")
            fig_s1.update_yaxes(title="Predicted Time (min)")
            st.plotly_chart(fig_s1, use_container_width=True)

        with c_s2:
            exp_range_s = np.linspace(0, 9, 20)
            preds_e = []
            for e in exp_range_s:
                xi = np.array([[distance, le_w.transform([weather_sel])[0], le_t.transform([traffic_sel])[0],
                                le_tod.transform([time_sel])[0], le_v.transform([vehicle_sel])[0],
                                prep_time, e]])
                preds_e.append(float(model.predict(xi)[0]))
            fig_s2 = go.Figure()
            fig_s2.add_trace(go.Scatter(
                x=exp_range_s, y=preds_e, mode="lines",
                line=dict(color="#06b6d4", width=3),
                fill="tozeroy", fillcolor="rgba(6,182,212,0.1)",
            ))
            fig_s2.add_vline(x=exp_yrs, line_dash="dot", line_color="#f59e0b",
                             annotation_text=f"Selected {exp_yrs} yrs", annotation_font_color="#f59e0b")
            fig_s2 = apply_layout(fig_s2, "Delivery Time vs Courier Experience")
            fig_s2.update_xaxes(title="Experience (years)")
            fig_s2.update_yaxes(title="Predicted Time (min)")
            st.plotly_chart(fig_s2, use_container_width=True)


# ══════════════════════════════════ TAB 4 ══════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'><div class='dot'></div><h2>Feature Importance</h2></div>", unsafe_allow_html=True)

    importances = dict(zip(feats, model.feature_importances_))
    labels = {
        "Distance_km": "Distance (km)",
        "Weather_enc": "Weather",
        "Traffic_enc": "Traffic Level",
        "Time_enc": "Time of Day",
        "Vehicle_enc": "Vehicle Type",
        "Preparation_Time_min": "Prep Time",
        "Courier_Experience_yrs": "Courier Exp",
    }
    imp_df = pd.DataFrame({
        "Feature": [labels[k] for k in importances],
        "Importance": list(importances.values()),
    }).sort_values("Importance", ascending=True)

    fig_imp = go.Figure(go.Bar(
        x=imp_df.Importance, y=imp_df.Feature, orientation="h",
        marker=dict(
            color=imp_df.Importance,
            colorscale=[[0,"#1e1b4b"],[0.4,"#7c3aed"],[0.7,"#06b6d4"],[1,"#10b981"]],
            showscale=False,
            line=dict(color="rgba(0,0,0,0.3)", width=1),
        ),
        text=[f"{v*100:.1f}%" for v in imp_df.Importance],
        textposition="outside", textfont=dict(color="#e2e8f0", size=11),
    ))
    fig_imp = apply_layout(fig_imp, "Feature Importance (Gradient Boosting)")
    fig_imp.update_layout(height=350, yaxis=dict(tickfont=dict(color="#94a3b8", size=12)))
    st.plotly_chart(fig_imp, use_container_width=True)

    # Actual vs Predicted
    st.markdown("<div class='section-header'><div class='dot'></div><h2>Actual vs Predicted</h2></div>", unsafe_allow_html=True)

    df_enc = df.copy()
    df_enc["Weather_enc"]  = le_w.transform(df_enc["Weather"])
    df_enc["Traffic_enc"]  = le_t.transform(df_enc["Traffic_Level"])
    df_enc["Time_enc"]     = le_tod.transform(df_enc["Time_of_Day"])
    df_enc["Vehicle_enc"]  = le_v.transform(df_enc["Vehicle_Type"])
    X_all = df_enc[feats]
    y_pred_all = model.predict(X_all)
    residuals = df_enc["Delivery_Time_min"].values - y_pred_all

    c1, c2 = st.columns(2)
    with c1:
        sample = min(300, len(df_enc))
        idx = np.random.choice(len(df_enc), sample, replace=False)
        fig_av = go.Figure()
        fig_av.add_trace(go.Scatter(
            x=df_enc["Delivery_Time_min"].iloc[idx],
            y=y_pred_all[idx],
            mode="markers",
            marker=dict(color="#7c3aed", opacity=.65, size=6,
                        line=dict(color="rgba(167,139,250,0.3)", width=.5)),
            name="Orders",
        ))
        rng = [df_enc.Delivery_Time_min.min(), df_enc.Delivery_Time_min.max()]
        fig_av.add_trace(go.Scatter(x=rng, y=rng, mode="lines",
                                    line=dict(color="#f43f5e", dash="dash", width=2), name="Perfect"))
        fig_av = apply_layout(fig_av, "Actual vs Predicted Delivery Time")
        fig_av.update_xaxes(title="Actual (min)")
        fig_av.update_yaxes(title="Predicted (min)")
        st.plotly_chart(fig_av, use_container_width=True)

    with c2:
        fig_res = go.Figure()
        fig_res.add_trace(go.Histogram(
            x=residuals, nbinsx=40,
            marker=dict(color="#06b6d4", opacity=.75,
                        line=dict(color="rgba(6,182,212,0.4)", width=.5)),
        ))
        fig_res.add_vline(x=0, line_dash="dash", line_color="#f43f5e",
                          annotation_text="Zero error", annotation_font_color="#f43f5e")
        fig_res = apply_layout(fig_res, "Residuals Distribution")
        fig_res.update_xaxes(title="Residual (min)")
        st.plotly_chart(fig_res, use_container_width=True)

    # Model stats
    st.markdown("<div class='section-header'><div class='dot'></div><h2>Model Performance Summary</h2></div>", unsafe_allow_html=True)
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("R² Score",  f"{r2:.4f}")
    mc2.metric("MAE",       f"{mae:.2f} min")
    mc3.metric("Training Samples", f"{len(df)}")
    mc4.metric("Trees",     "200")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background:var(--bg-card);border:1px solid var(--border);border-radius:12px;padding:1.2rem 1.5rem;
         font-size:.85rem;color:var(--text-muted);line-height:1.7'>
      🤖 <b style='color:#a78bfa'>Model:</b> Gradient Boosting Regressor (sklearn) &nbsp;|&nbsp;
      📊 <b style='color:#38bdf8'>R²:</b> {r2:.4f} &nbsp;|&nbsp;
      📉 <b style='color:#34d399'>MAE:</b> {mae:.2f} min &nbsp;|&nbsp;
      🏋️ <b style='color:#fbbf24'>Estimators:</b> 200 &nbsp;|&nbsp;
      🔑 <b style='color:#fb7185'>Top Feature:</b> Distance (76.2%)
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ───
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;padding:1.5rem;color:#334155;font-size:.8rem;border-top:1px solid rgba(124,58,237,0.15)'>
  🚀 <b style='color:#64748b'>DeliverIQ</b> · Built with Streamlit & Plotly ·
  <span style='color:#7c3aed'>Gradient Boosting</span> AI Model
</div>
""", unsafe_allow_html=True)