import streamlit as st
import numpy as np
from scipy.stats import poisson
import plotly.graph_objects as go
import pandas as pd
import json
import os
import xgboost as xgb

st.set_page_config(page_title="WC 2026 Predictor", page_icon="⚽", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0f1f3d 100%); }
    .title-block { text-align: center; padding: 2rem 0 1rem 0; border-bottom: 1px solid #1e3a5f; margin-bottom: 1.5rem; }
    .title-block h1 { font-size: 2.4rem; font-weight: 700; color: #ffffff; letter-spacing: -0.5px; }
    .title-block p  { color: #8899aa; font-size: 0.95rem; margin-top: 0.3rem; }
    .model-tag { display:inline-block; font-size:0.7rem; text-transform:uppercase; letter-spacing:1px; padding:0.2rem 0.7rem; border-radius:20px; margin-bottom:0.6rem; font-weight:600; }
    .tag-dc  { background:rgba(37,99,235,0.2); color:#60a5fa; border:1px solid #2563eb; }
    .tag-xgb { background:rgba(34,197,94,0.2); color:#4ade80; border:1px solid #22c55e; }
    .stat-card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 1.2rem 1.5rem; margin-bottom: 1rem; }
    .stat-card h4 { color: #8899aa; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 0.4rem 0; }
    .stat-card p  { color: #ffffff; font-size: 1.6rem; font-weight: 700; margin: 0; }
    .score-box { background: linear-gradient(135deg, #1e3a5f, #0f2d4a); border: 1px solid #2563eb; border-radius: 16px; padding: 1.5rem; text-align: center; }
    .score-box.xgb { border-color:#22c55e; background:linear-gradient(135deg,#163a2a,#0f2d22); }
    .score-box .teams { font-size: 1rem; color: #8899aa; margin-bottom: 0.4rem; }
    .score-box .score { font-size: 2.6rem; font-weight: 700; color: #ffffff; letter-spacing: 3px; }
    .score-box .label { font-size: 0.7rem; color: #8899aa; text-transform: uppercase; letter-spacing: 2px; margin-top: 0.4rem; }
    div[data-testid="stSelectbox"] label { color: #8899aa !important; font-size: 0.85rem; }
    .stButton > button { background: linear-gradient(135deg, #2563eb, #1d4ed8); color: white; border: none; border-radius: 8px; padding: 0.6rem 2rem; font-weight: 600; width: 100%; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-block">
    <h1>⚽ FIFA World Cup 2026 Predictor</h1>
    <p>Dixon-Coles + XGBoost · Datos: martj42 + Ranking FIFA (2018–2026)</p>
</div>
""", unsafe_allow_html=True)

BASE = os.path.dirname(__file__)

@st.cache_data
def load_dc():
    with open(os.path.join(BASE, "params.json"), encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def load_xgb():
    mh = xgb.XGBRegressor(); mh.load_model(os.path.join(BASE, "xgb_home.json"))
    ma = xgb.XGBRegressor(); ma.load_model(os.path.join(BASE, "xgb_away.json"))
    with open(os.path.join(BASE, "xgb_snapshot.json"), encoding="utf-8") as f:
        snap = json.load(f)
    return mh, ma, snap

try:
    dc = load_dc()
    model_home, model_away, snap = load_xgb()
except Exception as e:
    st.error(f"Error cargando modelos: {e}")
    st.stop()

attack, defense, home_adv = dc['attack'], dc['defense'], dc['home_adv']
teams = sorted(dc['teams'])
rankings, form = snap['rankings'], snap['form']
FEATURES = snap['features']

def matrix_from_lambdas(lam_a, lam_b, max_goals=8):
    return np.outer([poisson.pmf(i, lam_a) for i in range(max_goals+1)],
                    [poisson.pmf(j, lam_b) for j in range(max_goals+1)])

def summarize(matrix):
    p_win  = float(np.sum(np.tril(matrix, -1)))
    p_draw = float(np.sum(np.diag(matrix)))
    p_lose = float(np.sum(np.triu(matrix, 1)))
    idx = np.unravel_index(np.argmax(matrix), matrix.shape)
    return {'p_win': round(p_win*100,1), 'p_draw': round(p_draw*100,1), 'p_lose': round(p_lose*100,1),
            'score': (int(idx[0]), int(idx[1])), 'matrix': matrix}

def predict_dc(a, b, neutral=True):
    lam_a = np.exp(attack.get(a,0) - defense.get(b,0) + (0 if neutral else home_adv))
    lam_b = np.exp(attack.get(b,0) - defense.get(a,0))
    r = summarize(matrix_from_lambdas(lam_a, lam_b))
    r['lam_a'], r['lam_b'] = round(float(lam_a),2), round(float(lam_b),2)
    return r

def predict_xgb(a, b, neutral=True):
    ra, rb = rankings.get(a,1000), rankings.get(b,1000)
    fa, fb = form.get(a,{'gf':1,'ga':1}), form.get(b,{'gf':1,'ga':1})
    x = pd.DataFrame([{
        'home_rank': ra, 'away_rank': rb, 'rank_diff': ra-rb,
        'home_gf': fa['gf'], 'home_ga': fa['ga'],
        'away_gf': fb['gf'], 'away_ga': fb['ga'],
        'neutral': 1 if neutral else 0
    }])[FEATURES]
    lam_a = float(model_home.predict(x)[0])
    lam_b = float(model_away.predict(x)[0])
    lam_a, lam_b = max(lam_a, 0.05), max(lam_b, 0.05)
    r = summarize(matrix_from_lambdas(lam_a, lam_b))
    r['lam_a'], r['lam_b'] = round(lam_a,2), round(lam_b,2)
    return r

def conf(pw, pl):
    d = abs(pw-pl)
    if d>40: return "Alto","#22c55e"
    if d>20: return "Medio","#f59e0b"
    return "Bajo","#ef4444"

with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    st.markdown("---")
    team_a  = st.selectbox("🔵 Equipo A", teams, index=teams.index("Argentina") if "Argentina" in teams else 0)
    team_b  = st.selectbox("🔴 Equipo B", teams, index=teams.index("France") if "France" in teams else 1)
    neutral = st.checkbox("Sede neutral", value=True)
    go_btn  = st.button("⚡ Predecir", use_container_width=True)
    st.markdown("---")
    st.markdown(f"**Ranking {team_a}:** {rankings.get(team_a,'N/A')}")
    st.markdown(f"**Ranking {team_b}:** {rankings.get(team_b,'N/A')}")

def render_model(col, res, team_a, team_b, tag_class, tag_text, box_class=""):
    with col:
        st.markdown(f'<span class="model-tag {tag_class}">{tag_text}</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="score-box {box_class}"><div class="teams">{team_a} vs {team_b}</div><div class="score">{res["score"][0]} – {res["score"][1]}</div><div class="label">Marcador más probable</div></div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="stat-card"><h4>Gana {team_a[:3]}</h4><p>{res["p_win"]}%</p></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="stat-card"><h4>Empate</h4><p>{res["p_draw"]}%</p></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="stat-card"><h4>Gana {team_b[:3]}</h4><p>{res["p_lose"]}%</p></div>', unsafe_allow_html=True)
        cl, cv = conf(res['p_win'], res['p_lose'])
        st.markdown(f'<div class="stat-card"><h4>xG · Confianza</h4><p style="font-size:1.1rem">{res["lam_a"]} – {res["lam_b"]} · <span style="color:{cv}">{cl}</span></p></div>', unsafe_allow_html=True)

if not go_btn:
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        st.markdown('<div style="text-align:center;padding:4rem 0;color:#8899aa"><div style="font-size:4rem;margin-bottom:1rem">⚽</div><p style="font-size:1.1rem">Selecciona dos equipos y presiona <strong style="color:#2563eb">Predecir</strong><br>para comparar ambos modelos</p></div>', unsafe_allow_html=True)
else:
    if team_a == team_b:
        st.error("Selecciona dos equipos diferentes.")
        st.stop()

    res_dc  = predict_dc(team_a, team_b, neutral)
    res_xgb = predict_xgb(team_a, team_b, neutral)

    col_dc, col_xgb = st.columns(2, gap="large")
    render_model(col_dc,  res_dc,  team_a, team_b, "tag-dc",  "📐 Dixon-Coles")
    render_model(col_xgb, res_xgb, team_a, team_b, "tag-xgb", "🌳 XGBoost", "xgb")

    # Consenso
    st.markdown("---")
    avg_a = round((res_dc['score'][0] + res_xgb['score'][0]) / 2)
    avg_b = round((res_dc['score'][1] + res_xgb['score'][1]) / 2)
    same = res_dc['score'] == res_xgb['score']
    msg = "✅ Ambos modelos coinciden" if same else "⚠️ Los modelos difieren — partido incierto"
    st.markdown(f"#### 🎯 Consenso: {team_a} {avg_a} – {avg_b} {team_b}")
    st.caption(msg)

    # Matriz XGBoost (la mas informativa con features)
    st.markdown("---")
    st.markdown("#### 🔢 Matriz de probabilidades (XGBoost)")
    ms = 6
    matrix = res_xgb['matrix'][:ms+1,:ms+1]*100
    fig = go.Figure(go.Heatmap(z=matrix, x=[str(i) for i in range(ms+1)], y=[str(i) for i in range(ms+1)],
        colorscale=[[0,'#0a0e1a'],[0.3,'#163a2a'],[0.7,'#22c55e'],[1,'#4ade80']],
        text=[[f"{matrix[i][j]:.1f}%" for j in range(ms+1)] for i in range(ms+1)],
        texttemplate="%{text}", textfont=dict(size=11,color='white'), showscale=False))
    fig.update_layout(xaxis=dict(title=f"Goles {team_b}", tickfont=dict(color='white'), titlefont=dict(color='#8899aa')),
        yaxis=dict(title=f"Goles {team_a}", tickfont=dict(color='white'), titlefont=dict(color='#8899aa')),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=10,b=10), height=380)
    st.plotly_chart(fig, use_container_width=True)
