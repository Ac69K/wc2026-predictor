import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize
import plotly.graph_objects as go
import plotly.express as px
import requests
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# ── CONFIG ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WC 2026 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── STYLE ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    
    .main { background-color: #0a0e1a; }
    
    .stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0f1f3d 100%); }
    
    .title-block {
        text-align: center;
        padding: 2rem 0 1rem 0;
        border-bottom: 1px solid #1e3a5f;
        margin-bottom: 2rem;
    }
    .title-block h1 {
        font-size: 2.4rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.5px;
    }
    .title-block p {
        color: #8899aa;
        font-size: 0.95rem;
        margin-top: 0.3rem;
    }
    
    .stat-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .stat-card h4 { color: #8899aa; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; margin: 0 0 0.4rem 0; }
    .stat-card p  { color: #ffffff; font-size: 1.6rem; font-weight: 700; margin: 0; }
    
    .prob-bar-win  { background: #22c55e; height: 8px; border-radius: 4px; }
    .prob-bar-draw { background: #f59e0b; height: 8px; border-radius: 4px; }
    .prob-bar-lose { background: #ef4444; height: 8px; border-radius: 4px; }
    
    .score-box {
        background: linear-gradient(135deg, #1e3a5f, #0f2d4a);
        border: 1px solid #2563eb;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .score-box .teams { font-size: 1.1rem; color: #8899aa; margin-bottom: 0.5rem; }
    .score-box .score { font-size: 3rem; font-weight: 700; color: #ffffff; letter-spacing: 4px; }
    .score-box .label { font-size: 0.75rem; color: #2563eb; text-transform: uppercase; letter-spacing: 2px; margin-top: 0.5rem; }
    
    .confidence-high   { color: #22c55e; font-weight: 600; }
    .confidence-medium { color: #f59e0b; font-weight: 600; }
    .confidence-low    { color: #ef4444; font-weight: 600; }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label { color: #8899aa !important; font-size: 0.85rem; }
    
    .stButton > button {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover { background: linear-gradient(135deg, #1d4ed8, #1e40af); }
</style>
""", unsafe_allow_html=True)

# ── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
    <h1>⚽ FIFA World Cup 2026 Predictor</h1>
    <p>Modelo Dixon-Coles + Distribución de Poisson · Datos: martj42 (2018–2026)</p>
</div>
""", unsafe_allow_html=True)

# ── SELECCIONES DEL MUNDIAL 2026 ─────────────────────────────────────────────
WC2026_TEAMS = [
    "Mexico", "South Africa", "Czech Republic", "South Korea", "Switzerland",
    "Bosnia and Herzegovina", "Canada", "Qatar", "United States", "Paraguay",
    "Brazil", "Morocco", "Haiti", "Scotland", "Turkey", "Australia", "Germany",
    "Curaçao", "Ivory Coast", "Ecuador", "Netherlands", "Japan", "Sweden",
    "Tunisia", "Spain", "Cape Verde", "Belgium", "Egypt", "Iran", "New Zealand",
    "France", "Senegal", "Norway", "Iraq", "Argentina", "Algeria", "Jordan",
    "Austria", "Uzbekistan", "Colombia", "England", "Croatia", "Ghana", "Panama",
    "Uruguay", "Saudi Arabia", "Portugal", "DR Congo"
]

# ── DATOS ────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Descargando historial de partidos...")
def load_data():
    url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df['date'] = pd.to_datetime(df['date'])
        # Filtrar desde 2018
        df = df[df['date'] >= '2018-01-01'].copy()
        df = df.dropna(subset=['home_score', 'away_score'])
        df['home_score'] = df['home_score'].astype(int)
        df['away_score'] = df['away_score'].astype(int)
        # Solo partidos donde participe al menos una de las 48 selecciones del Mundial
        mask = df['home_team'].isin(WC2026_TEAMS) | df['away_team'].isin(WC2026_TEAMS)
        df = df[mask].copy()
        return df
    except Exception as e:
        st.error(f"Error descargando datos: {e}")
        return None

# ── MODELO DIXON-COLES ───────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Entrenando modelo Dixon-Coles...")
def train_dixon_coles(_df):
    teams = sorted(set(_df['home_team'].unique()) | set(_df['away_team'].unique()))
    team_idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    # Pre-computar índices y goles como arrays (vectorización)
    hi = _df['home_team'].map(team_idx).values
    ai = _df['away_team'].map(team_idx).values
    hs = _df['home_score'].values
    as_ = _df['away_score'].values

    def neg_log_likelihood(params):
        attack   = params[:n]
        defense  = params[n:2*n]
        home_adv = params[2*n]
        lam_h = np.exp(attack[hi] - defense[ai] + home_adv)
        lam_a = np.exp(attack[ai] - defense[hi])
        ll = poisson.logpmf(hs, lam_h).sum() + poisson.logpmf(as_, lam_a).sum()
        return -ll

    x0 = np.zeros(2 * n + 1)
    x0[2*n] = 0.2  # home advantage prior

    result = minimize(neg_log_likelihood, x0,
                      method='L-BFGS-B',
                      options={'maxiter': 50, 'ftol': 1e-4})

    params = result.x
    attack  = {t: params[i]     for t, i in team_idx.items()}
    defense = {t: params[n + i] for t, i in team_idx.items()}
    home_adv = params[2*n]

    return attack, defense, home_adv, teams

# ── PREDICCIÓN ────────────────────────────────────────────────────────────────
def predict_match(team_a, team_b, attack, defense, home_adv, neutral=True, max_goals=8):
    lam_a = np.exp(attack.get(team_a, 0) - defense.get(team_b, 0) + (0 if neutral else home_adv))
    lam_b = np.exp(attack.get(team_b, 0) - defense.get(team_a, 0))

    matrix = np.outer(
        [poisson.pmf(i, lam_a) for i in range(max_goals + 1)],
        [poisson.pmf(j, lam_b) for j in range(max_goals + 1)]
    )

    p_win  = np.sum(np.tril(matrix, -1))
    p_draw = np.sum(np.diag(matrix))
    p_lose = np.sum(np.triu(matrix, 1))

    # Marcador más probable
    idx = np.unravel_index(np.argmax(matrix), matrix.shape)
    most_likely = (idx[0], idx[1])

    return {
        'p_win':  round(p_win  * 100, 1),
        'p_draw': round(p_draw * 100, 1),
        'p_lose': round(p_lose * 100, 1),
        'lam_a':  round(lam_a, 2),
        'lam_b':  round(lam_b, 2),
        'score':  most_likely,
        'matrix': matrix
    }

def get_confidence(p_win, p_lose):
    diff = abs(p_win - p_lose)
    if diff > 40:   return "Alto",   "confidence-high"
    if diff > 20:   return "Medio",  "confidence-medium"
    return "Bajo", "confidence-low"

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    st.markdown("---")

    df = load_data()

    if df is not None:
        attack, defense, home_adv, teams = train_dixon_coles(df)

        team_a = st.selectbox("🔵 Equipo A", teams, index=teams.index("Argentina") if "Argentina" in teams else 0)
        team_b = st.selectbox("🔴 Equipo B", teams, index=teams.index("France") if "France" in teams else 1)
        neutral = st.checkbox("Sede neutral", value=True)
        predict_btn = st.button("⚡ Predecir", use_container_width=True)

        st.markdown("---")
        st.markdown("### 📊 Datos")
        st.markdown(f"**Partidos analizados:** {len(df):,}")
        st.markdown(f"**Selecciones:** {len(teams)}")
        st.markdown(f"**Período:** 2018 – 2026")

# ── MAIN ──────────────────────────────────────────────────────────────────────
if df is None:
    st.warning("No se pudieron cargar los datos. Verifica tu conexión.")
    st.stop()

attack, defense, home_adv, teams = train_dixon_coles(df)

if 'predict_btn' not in dir() or not predict_btn:
    # Estado inicial
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("""
        <div style="text-align:center; padding: 4rem 0; color: #8899aa;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">⚽</div>
            <p style="font-size: 1.1rem;">Selecciona dos equipos en el panel izquierdo<br>y presiona <strong style="color:#2563eb">Predecir</strong></p>
        </div>
        """, unsafe_allow_html=True)
else:
    if team_a == team_b:
        st.error("Selecciona dos equipos diferentes.")
        st.stop()

    result = predict_match(team_a, team_b, attack, defense, home_adv, neutral)

    conf_label, conf_class = get_confidence(result['p_win'], result['p_lose'])

    # ── FILA 1: marcador + probabilidades ─────────────────────────────────
    col_score, col_probs = st.columns([1, 1.5], gap="large")

    with col_score:
        st.markdown(f"""
        <div class="score-box">
            <div class="teams">{team_a} vs {team_b}</div>
            <div class="score">{result['score'][0]} – {result['score'][1]}</div>
            <div class="label">Marcador más probable</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""<div class="stat-card"><h4>xG {team_a}</h4><p>{result['lam_a']}</p></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="stat-card"><h4>xG {team_b}</h4><p>{result['lam_b']}</p></div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="stat-card"><h4>Nivel de confianza</h4><p class="{conf_class}">{conf_label}</p></div>""", unsafe_allow_html=True)

    with col_probs:
        st.markdown("#### Probabilidades de resultado")

        labels = [f"Gana {team_a}", "Empate", f"Gana {team_b}"]
        values = [result['p_win'], result['p_draw'], result['p_lose']]
        colors = ['#22c55e', '#f59e0b', '#ef4444']

        fig_bar = go.Figure(go.Bar(
            x=labels, y=values,
            marker_color=colors,
            text=[f"{v}%" for v in values],
            textposition='outside',
            textfont=dict(color='white', size=14)
        ))
        fig_bar.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(range=[0, max(values) + 15], showgrid=False, showticklabels=False),
            xaxis=dict(tickfont=dict(color='white', size=12)),
            margin=dict(t=20, b=10, l=0, r=0),
            height=280
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── FILA 2: Matriz de probabilidades ──────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔢 Matriz de probabilidades de marcador")

    max_show = 6
    matrix = result['matrix'][:max_show+1, :max_show+1] * 100

    fig_heat = go.Figure(go.Heatmap(
        z=matrix,
        x=[str(i) for i in range(max_show+1)],
        y=[str(i) for i in range(max_show+1)],
        colorscale=[
            [0.0, '#0a0e1a'],
            [0.3, '#1e3a5f'],
            [0.7, '#2563eb'],
            [1.0, '#22c55e']
        ],
        text=[[f"{matrix[i][j]:.1f}%" for j in range(max_show+1)] for i in range(max_show+1)],
        texttemplate="%{text}",
        textfont=dict(size=11, color='white'),
        showscale=False
    ))
    fig_heat.update_layout(
        xaxis=dict(title=f"Goles {team_b}", tickfont=dict(color='white'), titlefont=dict(color='#8899aa')),
        yaxis=dict(title=f"Goles {team_a}", tickfont=dict(color='white'), titlefont=dict(color='#8899aa')),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=10, b=10),
        height=380
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── FILA 3: Top 10 marcadores ─────────────────────────────────────────
    st.markdown("#### 🏆 Top 10 marcadores más probables")

    scores = []
    for i in range(max_show+1):
        for j in range(max_show+1):
            scores.append({'Marcador': f"{i}–{j}", 'Probabilidad': round(result['matrix'][i][j]*100, 2)})

    scores_df = pd.DataFrame(scores).sort_values('Probabilidad', ascending=False).head(10)

    fig_top = go.Figure(go.Bar(
        x=scores_df['Probabilidad'],
        y=scores_df['Marcador'],
        orientation='h',
        marker_color='#2563eb',
        text=[f"{v}%" for v in scores_df['Probabilidad']],
        textposition='outside',
        textfont=dict(color='white', size=11)
    ))
    fig_top.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, showticklabels=False, range=[0, scores_df['Probabilidad'].max()+3]),
        yaxis=dict(tickfont=dict(color='white', size=12), autorange='reversed'),
        margin=dict(t=10, b=10, l=80, r=60),
        height=320
    )
    st.plotly_chart(fig_top, use_container_width=True)
