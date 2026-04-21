import streamlit as st
import pandas as pd
import joblib
import xgboost
import numpy as np
import matplotlib.pyplot as plt

# Page settings
st.set_page_config(page_title="Salary Prediction App", layout="centered")

# ══════════════════════════════════════════════════════════════════════════════
#  LUXURY CSS — Amber & Orange · Warm Espresso · Soft White
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600;700&family=Jost:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ─── Design Tokens — Amber & Orange · Warm Espresso ─────── */
:root {
    --rose:        #f5a623;
    --rose-bright: #ffc05a;
    --rose-deep:   #d4780a;
    --gold:        #ff7043;
    --gold-light:  #ff9a76;
    --gold-pale:   #fff0e6;
    --rose-glow:   rgba(245,166,35,.25);
    --gold-glow:   rgba(255,112,67,.2);
    --bg:          #0f0a05;
    --bg2:         #160e06;
    --panel:       #1e1208;
    --panel2:      #271808;
    --line:        #3a2410;
    --line-rose:   rgba(245,166,35,.12);
    --text:        #fff8f0;
    --sub:         #8a6440;
    --white:       #ffffff;
}

/* ─── Global ────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Jost', sans-serif !important;
}
#MainMenu, footer, header,
[data-testid="stDecoration"],
[data-testid="stToolbar"] { display: none !important; }

/* ─── Warm radial background ─────────────────────────────── */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 70% 55% at 0%   0%,   rgba(245,166,35,.08)  0%, transparent 55%),
        radial-gradient(ellipse 60% 50% at 100% 100%, rgba(255,112,67,.07)  0%, transparent 55%),
        radial-gradient(ellipse 40% 40% at 100% 0%,  rgba(245,166,35,.05)  0%, transparent 50%),
        radial-gradient(ellipse 40% 35% at 0%   100%, rgba(255,112,67,.04) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

/* ─── Main block ─────────────────────────────────────────── */
[data-testid="stAppViewContainer"] > .main > .block-container {
    max-width: 860px;
    padding: 2.5rem 2rem 5rem;
    position: relative;
    z-index: 1;
}

/* ─── Hide native Streamlit titles (we replace with custom HTML) ── */
[data-testid="stAppViewContainer"] h1 { display: none !important; }

/* ══════════════════════════════════════════════════════════
   PAGE HERO BANNER  (reusable for every page)
   ══════════════════════════════════════════════════════════ */
.page-hero {
    background: linear-gradient(145deg, #231508 0%, #180e04 55%, #231200 100%);
    border: 1px solid rgba(245,166,35,.18);
    border-radius: 20px;
    padding: 2.8rem 2.8rem 2.4rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    box-shadow:
        0 1px 0 rgba(245,166,35,.08) inset,
        0 28px 70px rgba(0,0,0,.65),
        0 0 0 1px rgba(255,112,67,.05);
}
.page-hero::before {
    content: '';
    position: absolute;
    top: -80px; left: -80px;
    width: 320px; height: 320px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(245,166,35,.12) 0%, transparent 65%);
    pointer-events: none;
    animation: breathe 5s ease-in-out infinite;
}
.page-hero::after {
    content: '';
    position: absolute;
    bottom: -100px; right: -80px;
    width: 360px; height: 360px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(255,112,67,.09) 0%, transparent 65%);
    pointer-events: none;
    animation: breathe 5s 2.5s ease-in-out infinite;
}
@keyframes breathe {
    0%,100% { transform: scale(1);    opacity: 1; }
    50%      { transform: scale(1.1); opacity: .6; }
}
.hero-flourish {
    position: absolute; top: 22px; left: 26px;
    display: flex; flex-direction: column; gap: 5px; opacity: .35;
}
.hero-flourish span { display: block; height: 1px; background: linear-gradient(90deg, var(--gold), transparent); }
.hero-flourish span:nth-child(1) { width: 42px; }
.hero-flourish span:nth-child(2) { width: 28px; }
.hero-flourish span:nth-child(3) { width: 16px; }
.hero-diamond {
    position: absolute; top: 24px; right: 28px;
    width: 10px; height: 10px;
    border: 1px solid rgba(245,166,35,.4);
    transform: rotate(45deg);
    box-shadow: 0 0 10px var(--rose-glow);
}
.hero-badge {
    display: inline-flex; align-items: center; gap: .5rem;
    background: rgba(245,166,35,.08);
    border: 1px solid rgba(245,166,35,.22);
    color: var(--rose);
    font-family: 'DM Mono', monospace;
    font-size: .67rem; letter-spacing: .18em; text-transform: uppercase;
    padding: .35rem 1.1rem; border-radius: 100px; margin-bottom: 1.3rem;
}
.badge-gem {
    width: 6px; height: 6px; border-radius: 50%;
    background: linear-gradient(135deg, var(--rose-bright), var(--gold));
    box-shadow: 0 0 8px var(--rose-glow);
    animation: gem-glow 3s ease-in-out infinite;
}
@keyframes gem-glow {
    0%,100% { box-shadow: 0 0 6px var(--rose-glow); }
    50%      { box-shadow: 0 0 14px var(--rose-glow), 0 0 24px rgba(245,166,35,.15); }
}
.page-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3rem; font-weight: 700; line-height: 1.08;
    margin: 0 0 .8rem;
    background: linear-gradient(135deg, var(--gold-pale) 0%, var(--rose-bright) 40%, var(--gold) 75%, var(--rose-deep) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-divider {
    width: 60px; height: 1px;
    background: linear-gradient(90deg, var(--gold), var(--rose), transparent);
    margin: 0 0 1.1rem; opacity: .6;
}
.page-sub {
    color: var(--sub); font-size: .95rem; line-height: 1.72;
    margin: 0 0 1.8rem; max-width: 520px; font-weight: 300; letter-spacing: .02em;
}
.hero-chips { display: flex; flex-wrap: wrap; gap: .5rem; }
.hero-chip {
    background: rgba(255,112,67,.06);
    border: 1px solid rgba(255,112,67,.2);
    border-radius: 100px; padding: .3rem .9rem;
    font-size: .74rem; font-weight: 500; color: rgba(255,154,118,.75);
    display: flex; align-items: center; gap: .4rem; letter-spacing: .04em;
    transition: all .25s;
}
.hero-chip:hover { background: rgba(255,112,67,.13); border-color: var(--gold); color: var(--gold-light); }

/* ══════════════════════════════════════════════════════════
   METRIC CARDS
   ══════════════════════════════════════════════════════════ */
.metrics-grid {
    display: grid; grid-template-columns: repeat(4,1fr);
    gap: .85rem; margin-bottom: 2rem;
}
.metrics-grid-3 {
    display: grid; grid-template-columns: repeat(3,1fr);
    gap: .85rem; margin-bottom: 2rem;
}
.m-card {
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 14px; padding: 1.3rem 1rem; text-align: center;
    position: relative; overflow: hidden; transition: all .3s ease;
}
.m-card::after {
    content: ''; position: absolute; inset: 0;
    background: linear-gradient(145deg, rgba(245,166,35,.04) 0%, transparent 55%);
    border-radius: 14px; pointer-events: none;
}
.m-card:hover {
    border-color: rgba(245,166,35,.28);
    box-shadow: 0 8px 32px rgba(0,0,0,.45), 0 0 0 1px rgba(245,166,35,.1);
    transform: translateY(-3px);
}
.m-top-line {
    position: absolute; top: 0; left: 20%; right: 20%; height: 1px;
    background: linear-gradient(90deg, transparent, var(--rose), transparent);
    opacity: 0; transition: opacity .3s;
}
.m-card:hover .m-top-line { opacity: .7; }
.m-icon  { font-size: 1.45rem; margin-bottom: .55rem; display: block; }
.m-val   { font-family: 'Cormorant Garamond', serif; font-size: 1.7rem; font-weight: 700; color: var(--rose-bright); display: block; line-height: 1; }
.m-lbl   { font-family: 'DM Mono', monospace; font-size: .6rem; letter-spacing: .14em; text-transform: uppercase; color: var(--sub); margin-top: .5rem; display: block; }

/* ══════════════════════════════════════════════════════════
   HOW IT WORKS STRIP
   ══════════════════════════════════════════════════════════ */
.how-strip {
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 14px; padding: 1.6rem 2rem;
    display: flex; align-items: center; margin-bottom: 2rem;
    position: relative; overflow: hidden;
}
.how-strip::before {
    content: ''; position: absolute; top: 0; left: 10%; right: 10%; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,112,67,.28), transparent);
}
.how-strip::after {
    content: ''; position: absolute; bottom: 0; left: 10%; right: 10%; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(245,166,35,.2), transparent);
}
.how-step { flex: 1; text-align: center; padding: 0 .6rem; position: relative; }
.how-step:not(:last-child)::after {
    content: '·  ·  ·'; position: absolute; right: -14px; top: 50%;
    transform: translateY(-50%); color: var(--line); font-size: .7rem; letter-spacing: .1em;
}
.how-num {
    width: 32px; height: 32px; border-radius: 50%;
    background: rgba(245,166,35,.08); border: 1px solid rgba(245,166,35,.22);
    color: var(--rose); font-family: 'DM Mono', monospace; font-size: .72rem;
    display: flex; align-items: center; justify-content: center; margin: 0 auto .55rem;
}
.how-icon { font-size: 1.1rem; display: block; margin-bottom: .4rem; }
.how-txt  { font-size: .7rem; font-weight: 500; color: var(--sub); text-transform: uppercase; letter-spacing: .1em; font-family: 'DM Mono', monospace; }

/* ══════════════════════════════════════════════════════════
   SECTION HEADINGS (replaces st.subheader look)
   ══════════════════════════════════════════════════════════ */
.sec-head {
    font-family: 'DM Mono', monospace;
    font-size: .65rem; font-weight: 500; letter-spacing: .2em;
    text-transform: uppercase; color: var(--gold);
    margin: 2rem 0 .9rem;
    display: flex; align-items: center; gap: .7rem;
    opacity: .75;
}
.sec-head::before { content: '✦'; color: var(--rose); font-size: .65rem; }
.sec-head::after  { content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, rgba(255,112,67,.22), transparent); }

/* Streamlit h3 (st.subheader) */
[data-testid="stAppViewContainer"] h3 {
    font-family: 'Jost', sans-serif !important;
    font-size: .67rem !important; font-weight: 600 !important;
    letter-spacing: .18em !important; text-transform: uppercase !important;
    color: var(--sub) !important; margin: 2rem 0 .75rem !important;
}

/* ══════════════════════════════════════════════════════════
   CARDS / PANELS
   ══════════════════════════════════════════════════════════ */
.info-card {
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 16px; padding: 1.6rem 1.8rem; margin-bottom: 1.2rem;
    position: relative; overflow: hidden;
}
.info-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(245,166,35,.22), transparent);
}

/* insight pills */
.insight-row { display: flex; gap: .75rem; flex-wrap: wrap; margin-bottom: 1.2rem; }
.insight-pill {
    flex: 1; min-width: 180px;
    background: var(--panel2); border: 1px solid var(--line);
    border-radius: 12px; padding: 1.1rem 1.2rem;
    display: flex; align-items: center; gap: .9rem;
}
.pill-icon { font-size: 1.4rem; }
.pill-label { font-family: 'DM Mono', monospace; font-size: .62rem; letter-spacing: .12em; text-transform: uppercase; color: var(--sub); display: block; margin-bottom: .2rem; }
.pill-val   { font-family: 'Cormorant Garamond', serif; font-size: 1.15rem; font-weight: 700; color: var(--rose-bright); }

/* workflow step list */
.workflow {
    display: flex; flex-direction: column; gap: .6rem; margin-top: .5rem;
}
.wf-step {
    display: flex; align-items: center; gap: 1rem;
    background: var(--panel2); border: 1px solid var(--line);
    border-radius: 10px; padding: .75rem 1.1rem;
}
.wf-num {
    width: 26px; height: 26px; border-radius: 50%; flex-shrink: 0;
    background: rgba(245,166,35,.09); border: 1px solid rgba(245,166,35,.22);
    color: var(--rose); font-family: 'DM Mono', monospace; font-size: .65rem;
    display: flex; align-items: center; justify-content: center;
}
.wf-txt { font-size: .88rem; color: var(--text); font-weight: 400; }
.wf-arrow { color: var(--gold); font-size: .75rem; opacity: .4; }

/* ══════════════════════════════════════════════════════════
   MATPLOTLIB CHART STYLING
   ══════════════════════════════════════════════════════════ */
.chart-wrap {
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 16px; padding: 1.4rem 1.4rem 1rem;
    margin-bottom: 1.4rem; overflow: hidden;
}

/* ══════════════════════════════════════════════════════════
   ST.SUCCESS  (salary result)
   ══════════════════════════════════════════════════════════ */
[data-testid="stSuccess"] {
    background: linear-gradient(145deg,
        rgba(245,166,35,.09) 0%,
        rgba(255,112,67,.05) 50%,
        rgba(245,166,35,.04) 100%) !important;
    border: 1px solid rgba(245,166,35,.28) !important;
    border-radius: 18px !important;
    padding: 2.4rem 2rem !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 2.8rem !important; font-weight: 700 !important;
    color: var(--rose-bright) !important; text-align: center !important;
    box-shadow: 0 0 0 1px rgba(255,112,67,.08), 0 20px 60px rgba(0,0,0,.5) !important;
    letter-spacing: .04em !important;
}
[data-testid="stSuccess"] svg { display: none !important; }
[data-testid="stSuccess"] p  { color: var(--rose-bright) !important; font-size: inherit !important; font-family: inherit !important; font-weight: inherit !important; }

/* ST.METRIC */
[data-testid="stMetric"] {
    background: var(--panel) !important;
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
    padding: 1.2rem 1.3rem !important;
}
[data-testid="stMetricLabel"] p  { font-family: 'DM Mono', monospace !important; font-size: .62rem !important; letter-spacing: .14em !important; text-transform: uppercase !important; color: var(--sub) !important; }
[data-testid="stMetricValue"]    { font-family: 'Cormorant Garamond', serif !important; font-size: 1.7rem !important; font-weight: 700 !important; color: var(--rose-bright) !important; }

/* ST.WARNING */
[data-testid="stWarning"] {
    background: rgba(255,112,67,.07) !important;
    border: 1px solid rgba(255,112,67,.25) !important;
    border-radius: 12px !important; color: var(--gold-light) !important;
    font-size: .88rem !important; padding: 1rem 1.2rem !important;
}

/* ST.DATAFRAME */
[data-testid="stDataFrame"] {
    border-radius: 12px !important; overflow: hidden !important;
    border: 1px solid var(--line) !important;
}
[data-testid="stDataFrame"] table { background: var(--panel) !important; }
[data-testid="stDataFrame"] th {
    background: var(--panel2) !important;
    color: var(--rose) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: .65rem !important; letter-spacing: .12em !important; text-transform: uppercase !important;
    border-bottom: 1px solid var(--line) !important;
}
[data-testid="stDataFrame"] td { color: var(--text) !important; font-size: .85rem !important; border-color: var(--line) !important; }

/* ST.BAR_CHART */
[data-testid="stArrowVegaLiteChart"] {
    background: var(--panel) !important;
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
    padding: 1rem !important;
    margin-bottom: 1.4rem !important;
}

/* DOWNLOAD BUTTON */
[data-testid="stDownloadButton"] button {
    background: linear-gradient(135deg, rgba(245,166,35,.13), rgba(255,112,67,.1)) !important;
    border: 1px solid rgba(245,166,35,.32) !important;
    border-radius: 10px !important;
    color: var(--rose-bright) !important;
    font-family: 'Jost', sans-serif !important;
    font-size: .8rem !important; font-weight: 600 !important; letter-spacing: .1em !important;
    padding: .65rem 1.4rem !important;
    transition: all .25s !important;
}
[data-testid="stDownloadButton"] button:hover {
    background: linear-gradient(135deg, rgba(245,166,35,.22), rgba(255,112,67,.17)) !important;
    border-color: rgba(245,166,35,.55) !important;
    transform: translateY(-1px) !important;
}

/* ══════════════════════════════════════════════════════════
   SIDEBAR
   ══════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid rgba(245,166,35,.12) !important;
}
[data-testid="stSidebar"] > div { padding: 1.8rem 1.4rem !important; }

/* Nav title */
[data-testid="stSidebar"] h1,
[data-testid="stSidebarContent"] h1 {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.3rem !important; font-weight: 700 !important;
    color: var(--rose-bright) !important; margin-bottom: .8rem !important;
    letter-spacing: .04em !important;
}
/* Nav selectbox label */
[data-testid="stSidebar"] h2,
[data-testid="stSidebarContent"] h2 {
    font-family: 'Jost', sans-serif !important;
    font-size: .67rem !important; font-weight: 600 !important;
    letter-spacing: .2em !important; text-transform: uppercase !important;
    color: var(--rose) !important; margin-bottom: 1.6rem !important;
    padding-bottom: 1rem !important; border-bottom: 1px solid var(--line) !important;
}
[data-testid="stSidebar"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: .68rem !important; color: var(--sub) !important;
    text-transform: uppercase !important; letter-spacing: .1em !important;
    margin-bottom: .3rem !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: var(--panel) !important; border: 1px solid var(--line) !important;
    border-radius: 10px !important; color: var(--text) !important;
    font-size: .88rem !important; font-family: 'Jost', sans-serif !important;
    transition: border-color .25s, box-shadow .25s !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div:hover,
[data-testid="stSidebar"] .stSelectbox > div > div:focus-within {
    border-color: rgba(245,166,35,.4) !important;
    box-shadow: 0 0 0 3px rgba(245,166,35,.07) !important;
}
/* Slider */
[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
    background: linear-gradient(135deg, var(--rose-bright), var(--gold)) !important;
    border: 2px solid var(--bg) !important; box-shadow: 0 0 12px var(--rose-glow) !important;
    width: 20px !important; height: 20px !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stSliderTrackFill"] {
    background: linear-gradient(90deg, var(--rose-deep), var(--rose), var(--gold)) !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stSliderTrack"] {
    background: var(--line) !important; height: 3px !important; border-radius: 4px !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] p {
    font-family: 'DM Mono', monospace !important; font-size: .78rem !important; color: var(--text) !important;
}
/* Predict button */
[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #2a1800 0%, #1e1000 50%, #231200 100%) !important;
    color: var(--rose-bright) !important;
    font-family: 'Jost', sans-serif !important;
    font-size: .8rem !important; font-weight: 600 !important; letter-spacing: .2em !important;
    text-transform: uppercase !important;
    border: 1px solid rgba(245,166,35,.32) !important;
    border-radius: 12px !important; padding: 1rem 1.2rem !important;
    margin-top: 1.8rem !important; cursor: pointer !important;
    transition: all .3s ease !important;
    box-shadow: 0 4px 20px rgba(0,0,0,.4), inset 0 1px 0 rgba(245,166,35,.08) !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, #3a2200 0%, #2e1800 50%, #321a00 100%) !important;
    border-color: rgba(245,166,35,.58) !important;
    color: var(--gold-pale) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,.5), 0 0 20px rgba(245,166,35,.12), inset 0 1px 0 rgba(245,166,35,.14) !important;
}
[data-testid="stSidebar"] .stButton > button:active { transform: translateY(0) !important; }
[data-testid="stSidebar"] hr { border-color: var(--line) !important; margin: 1.3rem 0 !important; }

/* Sidebar section labels */
.sb-label {
    font-family: 'DM Mono', monospace;
    font-size: .6rem; letter-spacing: .18em; text-transform: uppercase;
    color: var(--gold); margin: 1.4rem 0 .7rem;
    display: flex; align-items: center; gap: .6rem; opacity: .65;
}
.sb-label::before { content: ''; width: 14px; height: 1px; background: var(--gold); opacity: .5; flex-shrink: 0; }
.sb-label::after  { content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, rgba(255,112,67,.28), transparent); }

/* ── Divider & footer ── */
hr { border-color: var(--line) !important; margin: 2.5rem 0 1rem !important; }
[data-testid="stAppViewContainer"] > .main p:last-of-type {
    font-family: 'DM Mono', monospace !important; font-size: .65rem !important;
    color: var(--sub) !important; text-align: center !important;
    letter-spacing: .14em !important; text-transform: uppercase !important; opacity: .55 !important;
}

/* ─── Animate-in ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(22px); }
    to   { opacity: 1; transform: translateY(0); }
}
.page-hero    { animation: fadeUp .65s cubic-bezier(.16,1,.3,1) both; }
.metrics-grid,
.metrics-grid-3 { animation: fadeUp .65s .13s cubic-bezier(.16,1,.3,1) both; }
.how-strip    { animation: fadeUp .65s .24s cubic-bezier(.16,1,.3,1) both; }
.info-card    { animation: fadeUp .65s .15s cubic-bezier(.16,1,.3,1) both; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib global style — Warm Espresso · Amber & Orange ─────────────────
plt.rcParams.update({
    "figure.facecolor":  "#1e1208",
    "axes.facecolor":    "#1e1208",
    "axes.edgecolor":    "#3a2410",
    "axes.labelcolor":   "#8a6440",
    "axes.titlecolor":   "#ffc05a",
    "xtick.color":       "#8a6440",
    "ytick.color":       "#8a6440",
    "grid.color":        "#3a2410",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "text.color":        "#fff8f0",
    "figure.edgecolor":  "#1e1208",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})
ROSE   = "#f5a623"   # amber
GOLD   = "#ff7043"   # orange
ROSE2  = "#d4780a"   # deep amber
MUTED  = "#8a6440"   # espresso muted

# ── Load data & model ────────────────────────────────────────────────────────
df    = pd.read_csv("job_salary_prediction_dataset.csv")
model = joblib.load("model.pkl")

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.title("Navigation")
menu = st.sidebar.selectbox(
    "Go to",
    ["Salary Prediction", "Dashboard", "Market Insights", "Dataset Explorer", "About Project"]
)

# ══════════════════════════════════════════════════════════════════════════════
#  ① SALARY PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
if menu == "Salary Prediction":

    st.title("💼 Job Salary Prediction System")

    # Hero
    st.markdown("""
    <div class="page-hero">
        <div class="hero-flourish"><span></span><span></span><span></span></div>
        <div class="hero-diamond"></div>
        <div class="hero-badge"><span class="badge-gem"></span> ✦ ML-Powered · XGBoost · Real-Time</div>
        <div class="page-title">Job Salary<br>Prediction System</div>
        <div class="hero-divider"></div>
        <p class="page-sub">Enter your professional profile in the side panel to receive a precise, data-driven salary estimate — curated by machine learning.</p>
        <div class="hero-chips">
            <div class="hero-chip">✦ Instant Estimate</div>
            <div class="hero-chip">◈ Role Precision</div>
            <div class="hero-chip">⊕ Location Aware</div>
            <div class="hero-chip">▲ XGBoost Engine</div>
            <div class="hero-chip">⬡ Zero Data Stored</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Dataset metrics
    total_roles   = df["job_title"].nunique()
    total_locs    = df["location"].nunique()
    total_sectors = df["industry"].nunique()
    total_records = len(df)
    st.markdown(f"""
    <div class="metrics-grid">
        <div class="m-card"><div class="m-top-line"></div><span class="m-icon">💼</span><span class="m-val">{total_roles}</span><span class="m-lbl">Job Roles</span></div>
        <div class="m-card"><div class="m-top-line"></div><span class="m-icon">📍</span><span class="m-val">{total_locs}</span><span class="m-lbl">Locations</span></div>
        <div class="m-card"><div class="m-top-line"></div><span class="m-icon">🏭</span><span class="m-val">{total_sectors}</span><span class="m-lbl">Sectors</span></div>
        <div class="m-card"><div class="m-top-line"></div><span class="m-icon">🗃️</span><span class="m-val">{total_records:,}</span><span class="m-lbl">Records</span></div>
    </div>
    """, unsafe_allow_html=True)

    # How it works
    st.markdown("""
    <div class="how-strip">
        <div class="how-step"><span class="how-icon">🗂️</span><div class="how-num">01</div><div class="how-txt">Select Profile</div></div>
        <div class="how-step"><span class="how-icon">🎚️</span><div class="how-num">02</div><div class="how-txt">Set Experience</div></div>
        <div class="how-step"><span class="how-icon">🤖</span><div class="how-num">03</div><div class="how-txt">Run Model</div></div>
        <div class="how-step"><span class="how-icon">💰</span><div class="how-num">04</div><div class="how-txt">View Salary</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar inputs
    st.sidebar.header("⚙ Enter Job Details")

    st.sidebar.markdown('<div class="sb-label">Role & Education</div>', unsafe_allow_html=True)
    job_title = st.sidebar.selectbox("💼 Job Title", sorted(df["job_title"].unique()))
    education = st.sidebar.selectbox("🎓 Education Level", sorted(df["education_level"].unique()))

    st.sidebar.markdown('<div class="sb-label">Company & Location</div>', unsafe_allow_html=True)
    industry     = st.sidebar.selectbox("🏭 Industry", sorted(df["industry"].unique()))
    company_size = st.sidebar.selectbox("🏢 Company Size", sorted(df["company_size"].unique()))
    location     = st.sidebar.selectbox("📍 Location", sorted(df["location"].unique()))
    remote       = st.sidebar.selectbox("🌐 Remote Work", sorted(df["remote_work"].unique()))

    st.sidebar.markdown('<div class="sb-label">Skills & Experience</div>', unsafe_allow_html=True)
    experience     = st.sidebar.slider("⏳ Experience Years", 0, 30, 1)
    skills         = st.sidebar.slider("🛠 Skills Count", 0, 20, 1)
    certifications = st.sidebar.slider("📜 Certifications", 0, 10, 0)

    if st.sidebar.button("Predict Salary"):

        input_data = pd.DataFrame({
            'job_title':        [job_title],
            'experience_years': [experience],
            'education_level':  [education],
            'skills_count':     [skills],
            'industry':         [industry],
            'company_size':     [company_size],
            'location':         [location],
            'remote_work':      [remote],
            'certifications':   [certifications]
        })

        prediction = model.predict(input_data)
        salary = int(prediction[0])

        if experience == 0:
            salary = int(salary * 0.65)

        min_salary = int(salary * 0.85)
        max_salary = int(salary * 1.15)

        st.subheader("Predicted Salary")
        st.success(f"₹ {salary:,}")
        st.write(f"Estimated Salary Range: ₹ {min_salary:,} — ₹ {max_salary:,}")

        # Context card
        remote_label = "🏠 Remote" if str(remote).lower() in ["yes", "remote", "true"] else "🏢 On-site"
        st.markdown(f"""
        <div style="margin-top:1rem;padding:1.3rem 1.6rem;
            background:linear-gradient(145deg,rgba(245,166,35,.06),rgba(255,112,67,.04));
            border:1px solid rgba(255,112,67,.18);border-left:2px solid rgba(245,166,35,.5);
            border-radius:14px;font-family:'DM Mono',monospace;font-size:.73rem;
            color:#6b4e2a;line-height:2.1;letter-spacing:.06em;">
            <div style="color:#f5a623;font-size:.68rem;letter-spacing:.2em;margin-bottom:.5rem;text-transform:uppercase;">✦ Estimate Based On</div>
            <span style="color:#a07040;">Role</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{job_title}<br>
            <span style="color:#a07040;">Experience</span>&nbsp;{experience} Years<br>
            <span style="color:#a07040;">Education</span>&nbsp;&nbsp;{education}<br>
            <span style="color:#a07040;">Sector</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{industry}<br>
            <span style="color:#a07040;">Location</span>&nbsp;&nbsp;&nbsp;{location}<br>
            <span style="color:#a07040;">Mode</span>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{remote_label}
        </div>
        """, unsafe_allow_html=True)

        # Download prediction
        result_df = pd.DataFrame({
            "Job Title":        [job_title],
            "Experience":       [experience],
            "Education":        [education],
            "Predicted Salary": [salary]
        })
        st.download_button(
            label="Download Prediction Report",
            data=result_df.to_csv(index=False),
            file_name="salary_prediction.csv",
            mime="text/csv"
        )

# ══════════════════════════════════════════════════════════════════════════════
#  ② DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "Dashboard":

    st.title("📊 Salary Dashboard")

    st.markdown("""
    <div class="page-hero">
        <div class="hero-flourish"><span></span><span></span><span></span></div>
        <div class="hero-diamond"></div>
        <div class="hero-badge"><span class="badge-gem"></span> ✦ Data Visualisation · Live Charts</div>
        <div class="page-title">Salary<br>Dashboard</div>
        <div class="hero-divider"></div>
        <p class="page-sub">Explore interactive charts revealing how salary varies across experience, role, location, education and distribution.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Experience vs Salary ──
    st.subheader("Experience vs Salary")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.scatter(df["experience_years"], df["salary"],
               color=ROSE, alpha=.6, s=18, edgecolors="none")
    ax.set_xlabel("Experience (Years)", labelpad=8)
    ax.set_ylabel("Salary (₹)", labelpad=8)
    ax.set_title("Experience vs Salary", pad=12, fontsize=11, color="#ffc05a")
    ax.grid(True)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ── Top Paying Job Roles ──
    st.subheader("Top Paying Job Roles")
    top_jobs = df.groupby("job_title")["salary"].mean().sort_values(ascending=False).head(5)
    st.bar_chart(top_jobs)

    # ── Salary by Location ──
    st.subheader("Salary by Location")
    loc_salary = df.groupby("location")["salary"].mean()
    st.bar_chart(loc_salary)

    # ── Salary by Education ──
    st.subheader("Salary by Education")
    edu_salary = df.groupby("education_level")["salary"].mean()
    st.bar_chart(edu_salary)

    # ── Salary Distribution ──
    st.subheader("Salary Distribution")
    fig2, ax2 = plt.subplots(figsize=(8, 3.8))
    n, bins, patches = ax2.hist(df["salary"], bins=30, color=ROSE, edgecolor="#0f0a05", linewidth=.4)
    # Amber → Orange gradient coloring across bars
    for i, patch in enumerate(patches):
        patch.set_facecolor(plt.cm.YlOrRd(0.3 + 0.7 * i / len(patches)))
    ax2.set_xlabel("Salary (₹)", labelpad=8)
    ax2.set_ylabel("Frequency", labelpad=8)
    ax2.set_title("Salary Distribution", pad=12, fontsize=11, color="#ffc05a")
    ax2.grid(True, axis="y")
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

# ══════════════════════════════════════════════════════════════════════════════
#  ③ MARKET INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "Market Insights":

    st.title("📈 Market Insights")

    st.markdown("""
    <div class="page-hero">
        <div class="hero-flourish"><span></span><span></span><span></span></div>
        <div class="hero-diamond"></div>
        <div class="hero-badge"><span class="badge-gem"></span> ✦ Market Intelligence · Key Signals</div>
        <div class="page-title">Market<br>Insights</div>
        <div class="hero-divider"></div>
        <p class="page-sub">High-level salary signals distilled from the entire dataset — top roles, locations and industries at a glance.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Highest Salary", f"₹ {df['salary'].max():,}")

    with col2:
        st.metric("Average Salary", f"₹ {int(df['salary'].mean()):,}")

    with col3:
        st.metric("Total Records", f"{len(df):,}")

    # Insight pills
    best_role = df.groupby("job_title")["salary"].mean().idxmax()
    best_loc  = df.groupby("location")["salary"].mean().idxmax()
    best_ind  = df.groupby("industry")["salary"].mean().idxmax()

    st.markdown(f"""
    <div style="margin-top:1.8rem;">
        <div class="sec-head">Top Performing Segments</div>
        <div class="insight-row">
            <div class="insight-pill">
                <span class="pill-icon">💼</span>
                <div><span class="pill-label">Best Job Role</span><span class="pill-val">{best_role}</span></div>
            </div>
            <div class="insight-pill">
                <span class="pill-icon">📍</span>
                <div><span class="pill-label">Best Location</span><span class="pill-val">{best_loc}</span></div>
            </div>
            <div class="insight-pill">
                <span class="pill-icon">🏭</span>
                <div><span class="pill-label">Best Industry</span><span class="pill-val">{best_ind}</span></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("Best Paying Job Role:", df.groupby("job_title")["salary"].mean().idxmax())
    st.write("Best Location:",        df.groupby("location")["salary"].mean().idxmax())
    st.write("Best Industry:",        df.groupby("industry")["salary"].mean().idxmax())

# ══════════════════════════════════════════════════════════════════════════════
#  ④ DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "Dataset Explorer":

    st.title("📂 Dataset Explorer")

    st.markdown("""
    <div class="page-hero">
        <div class="hero-flourish"><span></span><span></span><span></span></div>
        <div class="hero-diamond"></div>
        <div class="hero-badge"><span class="badge-gem"></span> ✦ Raw Data · Filter & Explore</div>
        <div class="page-title">Dataset<br>Explorer</div>
        <div class="hero-divider"></div>
        <p class="page-sub">Browse and filter the underlying job salary dataset. Select a role to narrow results to that specific segment.</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("Preview Dataset")
    st.dataframe(df.head())

    job_filter  = st.selectbox("Filter by Job Title", df["job_title"].unique())
    filtered_df = df[df["job_title"] == job_filter]

    st.write("Filtered Data")
    st.dataframe(filtered_df)

# ══════════════════════════════════════════════════════════════════════════════
#  ⑤ ABOUT PROJECT
# ══════════════════════════════════════════════════════════════════════════════
elif menu == "About Project":

    st.title("ℹ About This Project")

    st.markdown("""
    <div class="page-hero">
        <div class="hero-flourish"><span></span><span></span><span></span></div>
        <div class="hero-diamond"></div>
        <div class="hero-badge"><span class="badge-gem"></span> ✦ XGBoost · Streamlit · ML Pipeline</div>
        <div class="page-title">About This<br>Project</div>
        <div class="hero-divider"></div>
        <p class="page-sub">A machine learning application that predicts job salaries using an XGBoost regressor — trained, evaluated and deployed end-to-end.</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("""
    This application predicts job salaries using Machine Learning.
    
    Model Used: XGBoost Regressor  
    Features Used:
    - Job Title
    - Experience Years
    - Education Level
    - Skills Count
    - Industry
    - Company Size
    - Location
    - Remote Work
    - Certifications
    
    The model was trained on a job salary dataset and deployed using Streamlit.
    """)

    st.write("### Machine Learning Workflow")
    st.write("""
    Data Collection → Data Cleaning → EDA → Feature Encoding → Scaling → Model Training → Model Evaluation → Model Deployment
    """)

    # Workflow visual
    steps = [
        ("📥", "Data Collection"),
        ("🧹", "Data Cleaning"),
        ("🔍", "EDA"),
        ("🔠", "Feature Encoding"),
        ("⚖️", "Scaling"),
        ("🤖", "Model Training"),
        ("📊", "Evaluation"),
        ("🚀", "Deployment"),
    ]
    st.markdown('<div class="sec-head">ML Pipeline</div>', unsafe_allow_html=True)
    arrow_html = '<span class="wf-arrow">›</span>'
    workflow_parts = []
    for i, (icon, label) in enumerate(steps):
        num   = str(i + 1).zfill(2)
        arr   = arrow_html if i < len(steps) - 1 else ""
        workflow_parts.append(
            '<div class="wf-step">'
            '<div class="wf-num">' + num + '</div>'
            '<span style="font-size:1.1rem">' + icon + '</span>'
            '<span class="wf-txt">' + label + '</span>'
            + arr +
            '</div>'
        )
    st.markdown('<div class="workflow">' + "".join(workflow_parts) + '</div>', unsafe_allow_html=True)

    # Feature tags
    features = ["💼 Job Title", "⏳ Experience Years", "🎓 Education Level",
                 "🛠️ Skills Count", "🏭 Industry", "🏢 Company Size",
                 "📍 Location", "🌐 Remote Work", "📜 Certifications"]
    st.markdown('<div class="sec-head" style="margin-top:2rem">Model Features</div>', unsafe_allow_html=True)
    feature_chips = "".join('<div class="hero-chip">' + f + '</div>' for f in features)
    st.markdown('<div class="hero-chips" style="margin-top:.5rem">' + feature_chips + '</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.write("---")
st.write("Machine Learning Salary Prediction Project | Streamlit App")
