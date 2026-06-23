import streamlit as st
import numpy as np
from scipy.stats import poisson
import plotly.graph_objects as go
import pandas as pd
import json
import requests

st.set_page_config(page_title="WC 2026 Predictor", page_icon="⚽", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0f1f3d 100%); }
    .title-block { text-align: center; padding: 2rem 0 1rem 0; border-bottom: 1px solid #1e3a5f; margin-bottom: 2rem; }
    .title-block h1 { font-size: 2.4rem; font-weight: 700; color: #ffffff; letter-spacing: -0.5px; }
    .title-block p  { color: #8899aa; font-size: 0.95rem; margin-top: 0.3rem; }
    .stat-card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 1rem; }
    .stat-card h4 { color: #8899aa; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 0.4rem 0; }
    .stat-card p  { color: #ffffff; font-size: 1.6rem; font-weight: 700; margin: 0; }
    .score-box { background: linear-gradient(135deg, #1e3a5f, #0f2d4a); border: 1px solid #2563eb; border-radius: 16px; padding: 2rem; text-align: center; }
    .score-box .teams { font-size: 1.1rem; color: #8899aa; margin-bottom: 0.5rem; }
    .score-box .score { font-size: 3rem; font-weight: 700; color: #ffffff; letter-spacing: 4px; }
    .score-box .label { font-size: 0.75rem; color: #2563eb; text-transform: uppercase; letter-spacing: 2px; margin-top: 0.5rem; }
    .confidence-high   { color: #22c55e; font-weight: 600; }
    .confidence-medium { color: #f59e0b; font-weight: 600; }
    .confidence-low    { color: #ef4444; font-weight: 600; }
    div[data-testid="stSelectbox"] label { color: #8899aa !important; font-size: 0.85rem; }
    .stButton > button { background: linear-gradient(135deg, #2563eb, #1d4ed8); color: white; border: none; border-radius: 8px; padding: 0.6rem 2rem; font-weight: 600; width: 100%; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-block">
    <h1>⚽ FIFA World Cup 2026 Predictor</h1>
    <p>Modelo Dixon-Coles + Distribución de Poisson · Datos: martj42 (2018–2026)</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Cargando modelo...")
def load_params():
    url = "https://raw.githubusercontent.com/Ac69K/wc2026-predictor/main/params.json"
    r = requests.get(url, timeout=15)
    return r.json()

data     = load_params()
attack   = data['attack']
defense  = data['defense']
home_adv = data['home_adv']
teams    = sorted(data['teams'])

def predict_match(team_a, team_b, neutral=True, max_goals=8):
    lam_a = np.exp(attack.get(team_a, 0) - defense.get(team_b, 0) + (0 if neutral else home_adv))
    lam_b = np.exp(attack.get(team_b, 0) - defense.get(team_a, 0))
    matrix = np.outer([poisson.pmf(i, lam_a) for i in range(max_goals+1)], [poisson.pmf(j, lam_b) for j in range(max_goals+1)])
    p_win  = np.sum(np.tril(matrix, -1))
    p_draw = np.sum(np.diag(matrix))
    p_lose = np.sum(np.triu(matrix, 1))
    idx = np.unravel_index(np.argmax(matrix), matrix.shape)
    return {'p_win': round(p_win*100,1), 'p_draw': round(p_draw*100,1), 'p_lose': round(p_lose*100,1), 'lam_a': round(lam_a,2), 'lam_b': round(lam_b,2), 'score': (idx[0], idx[1]), 'matrix': matrix}

def get_confidence(p_win, p_lose):
    diff = abs(p_win - p_lose)
    if diff > 40: return "Alto",   "confidence-high"
    if diff > 20: return "Medio",  "confidence-medium"
    return "Bajo", "confidence-low"

with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    st.markdown("---")
    team_a  = st.selectbox("🔵 Equipo A", teams, index=teams.index("Argentina") if "Argentina" in teams else 0)
    team_b  = st.selectbox("🔴 Equipo B", teams, index=teams.index("France")    if "France"    in teams else 1)
    neutral = st.checkbox("Sede neutral", value=True)
    predict_btn = st.button("⚡ Predecir", use_container_width=True)
    st.markdown("---")
    st.markdown(f"**Selecciones:** {len(teams)}")
    st.markdown("**Período:** 2018–2026 · **Modelo:** Dixon-Coles")

if not predict_btn:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown('<div style="text-align:center;padding:4rem 0;color:#8899aa"><div style="font-size:4rem;margin-bottom:1rem">⚽</div><p style="font-size:1.1rem">Selecciona dos equipos en el panel izquierdo<br>y presiona <strong style="color:#2563eb">Predecir</strong></p></div>', unsafe_allow_html=True)
else:
    if team_a == team_b:
        st.error("Selecciona dos equipos diferentes.")
        st.stop()

    result = predict_match(team_a, team_b, neutral)
    conf_label, conf_class = get_confidence(result['p_win'], result['p_lose'])

    col_score, col_probs = st.columns([1, 1.5], gap="large")
    with col_score:
        st.markdown(f'<div class="score-box"><div class="teams">{team_a} vs {team_b}</div><div class="score">{result["score"][0]} – {result["score"][1]}</div><div class="label">Marcador más probable</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: st.markdown(f'<div class="stat-card"><h4>xG {team_a}</h4><p>{result["lam_a"]}</p></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="stat-card"><h4>xG {team_b}</h4><p>{result["lam_b"]}</p></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stat-card"><h4>Confianza</h4><p class="{conf_class}">{conf_label}</p></div>', unsafe_allow_html=True)

    with col_probs:
        st.markdown("#### Probabilidades de resultado")
        fig_bar = go.Figure(go.Bar(x=[f"Gana {team_a}", "Empate", f"Gana {team_b}"], y=[result['p_win'], result['p_draw'], result['p_lose']], marker_color=['#22c55e','#f59e0b','#ef4444'], text=[f"{v}%" for v in [result['p_win'], result['p_draw'], result['p_lose']]], textposition='outside', textfont=dict(color='white', size=14)))
        fig_bar.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', yaxis=dict(range=[0, max(result['p_win'], result['p_draw'], result['p_lose'])+15], showgrid=False, showticklabels=False), xaxis=dict(tickfont=dict(color='white', size=12)), margin=dict(t=20,b=10,l=0,r=0), height=280)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔢 Matriz de probabilidades de marcador")
    max_show = 6
    matrix = result['matrix'][:max_show+1, :max_show+1] * 100
    fig_heat = go.Figure(go.Heatmap(z=matrix, x=[str(i) for i in range(max_show+1)], y=[str(i) for i in range(max_show+1)], colorscale=[[0.0,'#0a0e1a'],[0.3,'#1e3a5f'],[0.7,'#2563eb'],[1.0,'#22c55e']], text=[[f"{matrix[i][j]:.1f}%" for j in range(max_show+1)] for i in range(max_show+1)], texttemplate="%{text}", textfont=dict(size=11, color='white'), showscale=False))
    fig_heat.update_layout(xaxis=dict(title=f"Goles {team_b}", tickfont=dict(color='white'), titlefont=dict(color='#8899aa')), yaxis=dict(title=f"Goles {team_a}", tickfont=dict(color='white'), titlefont=dict(color='#8899aa')), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=10,b=10), height=380)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("#### 🏆 Top 10 marcadores más probables")
    scores = [{'Marcador': f"{i}–{j}", 'Probabilidad': round(result['matrix'][i][j]*100, 2)} for i in range(max_show+1) for j in range(max_show+1)]
    scores_df = pd.DataFrame(scores).sort_values('Probabilidad', ascending=False).head(10)
    fig_top = go.Figure(go.Bar(x=scores_df['Probabilidad'], y=scores_df['Marcador'], orientation='h', marker_color='#2563eb', text=[f"{v}%" for v in scores_df['Probabilidad']], textposition='outside', textfont=dict(color='white', size=11)))
    fig_top.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', xaxis=dict(showgrid=False, showticklabels=False, range=[0, scores_df['Probabilidad'].max()+3]), yaxis=dict(tickfont=dict(color='white', size=12), autorange='reversed'), margin=dict(t=10,b=10,l=80,r=60), height=320)
    st.plotly_chart(fig_top, use_container_width=True)
