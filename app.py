import streamlit as st
import numpy as np
from scipy.stats import poisson
import plotly.graph_objects as go
import pandas as pd
import json, os, re
import xgboost as xgb
import requests

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
    .ctx-note { background:rgba(34,197,94,0.07); border:1px solid rgba(34,197,94,0.25); border-radius:10px; padding:0.8rem 1rem; color:#4ade80; font-size:0.85rem; margin-bottom:1rem; line-height:1.6; }
    .ctx-warn { background:rgba(245,158,11,0.07); border:1px solid rgba(245,158,11,0.25); border-radius:10px; padding:0.6rem 1rem; color:#fbbf24; font-size:0.8rem; margin-bottom:1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="title-block">
    <h1>⚽ FIFA World Cup 2026 Predictor</h1>
    <p>Dixon-Coles + XGBoost · Contexto automático · Datos: martj42 + Ranking FIFA (2018–2026)</p>
</div>
""", unsafe_allow_html=True)

BASE = os.path.dirname(__file__)

@st.cache_data
def load_dc():
    with open(os.path.join(BASE, "params.json"), encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource
def load_xgb_models():
    mh = xgb.XGBRegressor(); mh.load_model(os.path.join(BASE, "xgb_home.json"))
    ma = xgb.XGBRegressor(); ma.load_model(os.path.join(BASE, "xgb_away.json"))
    with open(os.path.join(BASE, "xgb_snapshot.json"), encoding="utf-8") as f:
        snap = json.load(f)
    return mh, ma, snap

try:
    dc = load_dc()
    model_home, model_away, snap = load_xgb_models()
except Exception as e:
    st.error(f"Error cargando modelos: {e}")
    st.stop()

attack, defense, home_adv = dc['attack'], dc['defense'], dc['home_adv']
teams    = sorted(dc['teams'])
rankings = snap['rankings']
form     = snap['form']
FEATURES = snap['features']

# ── CORE MATH ────────────────────────────────────────────────────────────────
def matrix_from_lambdas(lam_a, lam_b, max_goals=8):
    return np.outer(
        [poisson.pmf(i, lam_a) for i in range(max_goals+1)],
        [poisson.pmf(j, lam_b) for j in range(max_goals+1)]
    )

def summarize(matrix):
    p_win  = float(np.sum(np.tril(matrix, -1)))
    p_draw = float(np.sum(np.diag(matrix)))
    p_lose = float(np.sum(np.triu(matrix, 1)))
    idx    = np.unravel_index(np.argmax(matrix), matrix.shape)
    return {'p_win':  round(p_win*100, 1), 'p_draw': round(p_draw*100, 1),
            'p_lose': round(p_lose*100, 1), 'score': (int(idx[0]), int(idx[1])), 'matrix': matrix}

def predict_dc(a, b, neutral=True, adj_a=1.0, adj_b=1.0):
    lam_a = np.exp(attack.get(a,0) - defense.get(b,0) + (0 if neutral else home_adv)) * adj_a
    lam_b = np.exp(attack.get(b,0) - defense.get(a,0)) * adj_b
    lam_a, lam_b = max(lam_a, 0.05), max(lam_b, 0.05)
    r = summarize(matrix_from_lambdas(lam_a, lam_b))
    r['lam_a'], r['lam_b'] = round(float(lam_a), 2), round(float(lam_b), 2)
    return r

def predict_xgb(a, b, neutral=True, adj_a=1.0, adj_b=1.0):
    ra, rb = rankings.get(a, 1000), rankings.get(b, 1000)
    fa, fb = form.get(a, {'gf':1,'ga':1}), form.get(b, {'gf':1,'ga':1})
    x = pd.DataFrame([{
        'home_rank': ra, 'away_rank': rb, 'rank_diff': ra-rb,
        'home_gf': fa['gf'], 'home_ga': fa['ga'],
        'away_gf': fb['gf'], 'away_ga': fb['ga'],
        'neutral': 1 if neutral else 0
    }])[FEATURES]
    lam_a = float(model_home.predict(x)[0]) * adj_a
    lam_b = float(model_away.predict(x)[0]) * adj_b
    lam_a, lam_b = max(lam_a, 0.05), max(lam_b, 0.05)
    r = summarize(matrix_from_lambdas(lam_a, lam_b))
    r['lam_a'], r['lam_b'] = round(lam_a, 2), round(lam_b, 2)
    return r

def conf(pw, pl):
    d = abs(pw - pl)
    if d > 40: return "Alto",  "#22c55e"
    if d > 20: return "Medio", "#f59e0b"
    return "Bajo", "#ef4444"

def best_quiniela_bet(matrix, team_a, team_b, pts_tendency=3, pts_exact=5, max_goals=6):
    candidates = []
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p_exact = float(matrix[i][j])
            if i > j:    tendency = "home"; p_tend = float(np.sum(np.tril(matrix, -1)))
            elif i == j: tendency = "draw"; p_tend = float(np.sum(np.diag(matrix)))
            else:        tendency = "away"; p_tend = float(np.sum(np.triu(matrix,  1)))
            candidates.append((i, j, p_exact*pts_exact + p_tend*pts_tendency, tendency, p_exact, p_tend))
    candidates.sort(key=lambda x: -x[2])
    i, j, ev, tendency, p_exact, p_tend = candidates[0]
    label = f"Gana {team_a}" if tendency=="home" else ("Empate" if tendency=="draw" else f"Gana {team_b}")
    return {
        'score': (i, j), 'ev': round(ev*100, 2), 'p_exact': round(p_exact*100, 2),
        'p_tend': round(p_tend*100, 1), 'tendency_label': label,
        'top3': [(c[0], c[1], round(c[2]*100, 2)) for c in candidates[:3]]
    }

# ── AUTO CONTEXT (Anthropic API + web_search) ─────────────────────────────────
def fetch_wc_context(team_a, team_b):
    """
    Calls claude-sonnet-4-6 with web_search to auto-fetch WC2026 injury/form context.
    Returns {adj_a, adj_b, context_a, context_b} or None.
    Results cached in session_state per team pair.
    """
    cache_key = f"ctx__{team_a}__{team_b}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return None

    if not api_key:
        return None

    hdrs = {
        "content-type":    "application/json",
        "x-api-key":       api_key,
        "anthropic-version": "2023-06-01",
        "anthropic-beta":  "web-search-2025-03-05",
    }

    prompt = (
        f"FIFA World Cup 2026 is happening right now (June 2026). "
        f"Search for the very latest news about {team_a} and {team_b} at this tournament.\n\n"
        f"Find for each team: confirmed injuries, suspensions, recent match results, "
        f"squad issues, and motivation level (already through? must-win?).\n\n"
        f"Respond with ONLY a JSON object — no markdown fences, no other text:\n"
        f'{{"adj_a":<0.70-1.30>,"adj_b":<0.70-1.30>,'
        f'"context_a":"<{team_a} one-sentence summary>",'
        f'"context_b":"<{team_b} one-sentence summary>"}}\n\n'
        f"Multiplier reference: 1.00=normal squad, 1.10=full squad+strong form, "
        f"0.92=minor injury, 0.85=key player out, 0.75=multiple absences, "
        f"0.88=low motivation (eliminated/already qualified and rotating)"
    )

    messages = [{"role": "user", "content": prompt}]

    try:
        for _ in range(8):
            body = {
                "model":     "claude-sonnet-4-6",
                "max_tokens": 500,
                "tools":     [{"type": "web_search_20250305", "name": "web_search"}],
                "messages":  messages,
            }
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=hdrs, json=body, timeout=45
            )
            if resp.status_code != 200:
                return None

            data    = resp.json()
            content = data.get("content", [])
            stop    = data.get("stop_reason", "end_turn")

            if stop == "end_turn":
                for block in content:
                    if block.get("type") == "text":
                        m = re.search(r'\{[^{}]*"adj_a"[^{}]*\}', block["text"], re.DOTALL)
                        if m:
                            result = json.loads(m.group())
                            result["adj_a"] = round(max(0.60, min(1.40, float(result.get("adj_a", 1.0)))), 2)
                            result["adj_b"] = round(max(0.60, min(1.40, float(result.get("adj_b", 1.0)))), 2)
                            st.session_state[cache_key] = result
                            return result
                return None

            elif stop == "tool_use":
                messages.append({"role": "assistant", "content": content})
                tool_results = [
                    {"type": "tool_result", "tool_use_id": b["id"], "content": ""}
                    for b in content if b.get("type") == "tool_use"
                ]
                if tool_results:
                    messages.append({"role": "user", "content": tool_results})
            else:
                break

    except Exception:
        pass

    return None

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    st.markdown("---")
    team_a  = st.selectbox("🔵 Equipo A", teams, index=teams.index("Argentina") if "Argentina" in teams else 0)
    team_b  = st.selectbox("🔴 Equipo B", teams, index=teams.index("France")    if "France"    in teams else 1)
    neutral = st.checkbox("Sede neutral", value=True)
    go_btn  = st.button("⚡ Predecir", use_container_width=True)
    st.markdown("---")
    st.markdown(f"**Ranking {team_a}:** {rankings.get(team_a,'N/A')}")
    st.markdown(f"**Ranking {team_b}:** {rankings.get(team_b,'N/A')}")
    st.markdown("---")
    try:
        has_key = bool(st.secrets.get("ANTHROPIC_API_KEY",""))
    except Exception:
        has_key = False
    if has_key:
        st.success("🟢 Contexto automático activo")
    else:
        st.info("⚪ Sin contexto automático\nAgrega ANTHROPIC_API_KEY en Streamlit Secrets")

# ── RENDER MODEL COLUMN ───────────────────────────────────────────────────────
def render_model(col, res, ta, tb, tag_class, tag_text, box_class=""):
    with col:
        st.markdown(f'<span class="model-tag {tag_class}">{tag_text}</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="score-box {box_class}"><div class="teams">{ta} vs {tb}</div><div class="score">{res["score"][0]} – {res["score"][1]}</div><div class="label">Marcador más probable</div></div>', unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        c1.markdown(f'<div class="stat-card"><h4>Gana {ta[:3]}</h4><p>{res["p_win"]}%</p></div>',  unsafe_allow_html=True)
        c2.markdown(f'<div class="stat-card"><h4>Empate</h4><p>{res["p_draw"]}%</p></div>',          unsafe_allow_html=True)
        c3.markdown(f'<div class="stat-card"><h4>Gana {tb[:3]}</h4><p>{res["p_lose"]}%</p></div>', unsafe_allow_html=True)
        cl, cv = conf(res['p_win'], res['p_lose'])
        st.markdown(f'<div class="stat-card"><h4>xG · Confianza</h4><p style="font-size:1.1rem">{res["lam_a"]} – {res["lam_b"]} · <span style="color:{cv}">{cl}</span></p></div>', unsafe_allow_html=True)

# ── MAIN ──────────────────────────────────────────────────────────────────────
if not go_btn:
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        st.markdown('<div style="text-align:center;padding:4rem 0;color:#8899aa"><div style="font-size:4rem;margin-bottom:1rem">⚽</div><p style="font-size:1.1rem">Selecciona dos equipos y presiona<br><strong style="color:#2563eb">⚡ Predecir</strong></p></div>', unsafe_allow_html=True)
else:
    if team_a == team_b:
        st.error("Selecciona dos equipos diferentes.")
        st.stop()

    # Auto context fetch
    ctx = None
    if has_key:
        with st.spinner("🔍 Buscando contexto del partido..."):
            ctx = fetch_wc_context(team_a, team_b)

    adj_a = ctx["adj_a"] if ctx else 1.0
    adj_b = ctx["adj_b"] if ctx else 1.0

    # Context display
    if ctx:
        st.markdown(
            f'<div class="ctx-note">'
            f'🩺 <strong>{team_a}:</strong> {ctx["context_a"]}<br>'
            f'🩺 <strong>{team_b}:</strong> {ctx["context_b"]}<br>'
            f'<span style="font-size:0.75rem;opacity:0.6">'
            f'Ajustes · {team_a}: ×{adj_a} · {team_b}: ×{adj_b}'
            f'</span></div>',
            unsafe_allow_html=True
        )
    elif has_key:
        st.markdown('<div class="ctx-warn">⚠️ No se pudo obtener contexto — predicción con parámetros base</div>', unsafe_allow_html=True)

    # Predictions
    res_dc  = predict_dc(team_a,  team_b, neutral, adj_a, adj_b)
    res_xgb = predict_xgb(team_a, team_b, neutral, adj_a, adj_b)

    col_dc, col_xgb = st.columns(2, gap="large")
    render_model(col_dc,  res_dc,  team_a, team_b, "tag-dc",  "📐 Dixon-Coles")
    render_model(col_xgb, res_xgb, team_a, team_b, "tag-xgb", "🌳 XGBoost", "xgb")

    # Consensus + surprise alert
    st.markdown("---")
    avg_a = round((res_dc['score'][0] + res_xgb['score'][0]) / 2)
    avg_b = round((res_dc['score'][1] + res_xgb['score'][1]) / 2)
    same  = res_dc['score'] == res_xgb['score']
    st.markdown(f"#### 🎯 Consenso: {team_a} {avg_a} – {avg_b} {team_b}")
    st.caption("✅ Ambos modelos coinciden" if same else "⚠️ Los modelos difieren — partido incierto")

    if max(res_xgb['p_win'], res_xgb['p_lose']) < 55:
        st.warning("🚨 Alerta de sorpresa: El favorito es vulnerable, ideal para apostar al empate.")

    # Quiniela optimizer
    st.markdown("---")
    st.markdown("#### 🏆 Optimizador de Quiniela")
    bet = best_quiniela_bet(res_xgb['matrix'], team_a, team_b)
    c1,c2,c3 = st.columns(3)
    c1.markdown(f'<div class="stat-card"><h4>Apuesta óptima</h4><p style="font-size:1.4rem">{bet["score"][0]}–{bet["score"][1]}</p></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="stat-card"><h4>Valor esperado</h4><p style="font-size:1.4rem;color:#f59e0b">{bet["ev"]} pts</p></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="stat-card"><h4>Tendencia</h4><p style="font-size:1.1rem">{bet["tendency_label"]}<br><span style="font-size:0.9rem;color:#8899aa">{bet["p_tend"]}% prob</span></p></div>', unsafe_allow_html=True)
    st.caption(f"Marcador exacto: {bet['p_exact']}% · Top 3 por EV: " +
               " | ".join([f"{s[0]}–{s[1]} ({s[2]} pts)" for s in bet['top3']]))

    # XGBoost heatmap
    st.markdown("---")
    st.markdown("#### 🔢 Matriz de probabilidades (XGBoost)")
    ms = 6
    matrix = res_xgb['matrix'][:ms+1,:ms+1]*100
    fig = go.Figure(go.Heatmap(
        z=matrix, x=[str(i) for i in range(ms+1)], y=[str(i) for i in range(ms+1)],
        colorscale=[[0,'#0a0e1a'],[0.3,'#163a2a'],[0.7,'#22c55e'],[1,'#4ade80']],
        text=[[f"{matrix[i][j]:.1f}%" for j in range(ms+1)] for i in range(ms+1)],
        texttemplate="%{text}", textfont=dict(size=11,color='white'), showscale=False
    ))
    fig.update_layout(
        xaxis=dict(title=f"Goles {team_b}", tickfont=dict(color='white'), title_font=dict(color='#8899aa')),
        yaxis=dict(title=f"Goles {team_a}", tickfont=dict(color='white'), title_font=dict(color='#8899aa')),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=10,b=10), height=380
    )
    st.plotly_chart(fig, use_container_width=True)
