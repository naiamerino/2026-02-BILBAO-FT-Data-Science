import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import date

# ─── Configuración de la página ───────────────────────────────────────────────
st.set_page_config(
    page_title="WTA Match Predictor",
    page_icon="🎾",
    layout="centered"
)

# ─── Estilos ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0f0f0f;
        color: #f0f0f0;
    }
    .titulo {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 3.5rem;
        letter-spacing: 0.08em;
        color: #c8f000;
        margin-bottom: 0;
        line-height: 1;
    }
    .subtitulo {
        font-size: 0.9rem;
        color: #888;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 2rem;
    }
    .vs {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 2rem;
        color: #444;
        text-align: center;
        padding-top: 1.5rem;
    }
    .prob-box {
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    .prob-ganadora {
        background: linear-gradient(135deg, #c8f000 0%, #8ab800 100%);
        color: #0f0f0f;
    }
    .prob-perdedora {
        background: #1e1e1e;
        color: #888;
        border: 1px solid #333;
    }
    .prob-nombre {
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    .prob-numero {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 3rem;
        line-height: 1;
    }
    .prob-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 0.3rem;
    }
    .divider {
        border: none;
        border-top: 1px solid #222;
        margin: 2rem 0;
    }
    .info-chip {
        display: inline-block;
        background: #1e1e1e;
        border: 1px solid #333;
        border-radius: 20px;
        padding: 0.3rem 0.8rem;
        font-size: 0.8rem;
        color: #aaa;
        margin: 0.2rem;
    }
    .stSelectbox label, .stDateInput label {
        color: #aaa !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .stSelectbox > div > div {
        background-color: #1e1e1e !important;
        border-color: #333 !important;
        color: #f0f0f0 !important;
    }
    .stButton > button {
        background: #c8f000;
        color: #0f0f0f;
        font-weight: 600;
        font-size: 1rem;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 2rem;
        width: 100%;
        letter-spacing: 0.05em;
        transition: opacity 0.2s;
    }
    .stButton > button:hover {
        opacity: 0.85;
        color: #0f0f0f;
    }
    .warning-box {
        background: #1e1e1e;
        border-left: 3px solid #c8f000;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
        color: #aaa;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Cargar datos y modelo ─────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    with open('gbx_red.model', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def cargar_historico():
    return pd.read_csv('historico_partidos.csv', parse_dates=['date'])

modelo = cargar_modelo()
df = cargar_historico()

# ─── Jugadoras activas (con partidos en el último año del dataset) ─────────────
@st.cache_data
def jugadoras_activas(df):
    fecha_max = df['date'].max()
    fecha_corte = fecha_max - pd.DateOffset(months=12)
    recientes = df[df['date'] >= fecha_corte]
    # Recuperar nombres originales desde el histórico usando match_id
    # Cargamos wta para sacar nombres
    try:
        wta = pd.read_csv('wta_copia_mod.csv', low_memory=False)
        ids_recientes = recientes['match_id'].values
        jugadoras = pd.concat([
            wta.loc[wta.index.isin(ids_recientes), 'Player_1'],
            wta.loc[wta.index.isin(ids_recientes), 'Player_2']
        ]).unique()
        return sorted(jugadoras)
    except:
        return []

jugadoras = jugadoras_activas(df)

# ─── Funciones de features ────────────────────────────────────────────────────
@st.cache_data
def cargar_wta():
    wta = pd.read_csv('wta_copia_mod.csv', low_memory=False)
    wta['Date'] = pd.to_datetime(wta['Date'], errors='coerce')
    wta = wta.dropna(subset=['Date'])
    return wta

wta = cargar_wta()

def forma_reciente(df_wta, jugadora, fecha_limite, meses=2, superficie=None):
    fecha_inicio = fecha_limite - pd.DateOffset(months=meses)
    mask = (
        ((df_wta['Player_1'] == jugadora) | (df_wta['Player_2'] == jugadora)) &
        (df_wta['Date'] < fecha_limite) &
        (df_wta['Date'] >= fecha_inicio)
    )
    if superficie:
        mask &= (df_wta['Surface'] == superficie)
    partidos = df_wta[mask]
    if len(partidos) == 0:
        return 0
    victorias = (
        ((partidos['Player_1'] == jugadora) & (partidos['Winner'] == 1)) |
        ((partidos['Player_2'] == jugadora) & (partidos['Winner'] == 2))
    ).sum()
    return victorias/len(partidos)

def winrate(df_wta, jugadora, fecha_limite, superficie=None, ronda=None):
    mask = (
        ((df_wta['Player_1'] == jugadora) | (df_wta['Player_2'] == jugadora)) &
        (df_wta['Date'] < fecha_limite)
    )
    if superficie:
        mask &= (df_wta['Surface'] == superficie)
    if ronda:
        mask &= (df_wta['Round'] == ronda)
    partidos = df_wta[mask]
    if len(partidos) == 0:
        return 0.4
    victorias = (
        ((partidos['Player_1'] == jugadora) & (partidos['Winner'] == 1)) |
        ((partidos['Player_2'] == jugadora) & (partidos['Winner'] == 2))
    ).sum()
    return victorias/len(partidos)

def headtohead(df_wta, p1, p2, fecha_limite):
    mask = (
        ((df_wta['Player_1'] == p1) & (df_wta['Player_2'] == p2)) |
        ((df_wta['Player_1'] == p2) & (df_wta['Player_2'] == p1))
    ) & (df_wta['Date'] < fecha_limite)
    partidos = df_wta[mask]
    if len(partidos) == 0:
        return 0.5
    victorias_p1 = (
        ((partidos['Player_1'] == p1) & (partidos['Winner'] == 1)) |
        ((partidos['Player_2'] == p1) & (partidos['Winner'] == 2))
    ).sum()
    return victorias_p1/len(partidos)

def experiencia(df_wta, jugadora, fecha_limite):
    mask = (
        ((df_wta['Player_1'] == jugadora) | (df_wta['Player_2'] == jugadora)) &
        (df_wta['Date'] < fecha_limite)
    )
    partidos = partidos = df[mask]
    return len(partidos)

def get_ranking(df_wta, jugadora, fecha_limite):
    mask = (
        ((df_wta['Player_1'] == jugadora) | (df_wta['Player_2'] == jugadora)) &
        (df_wta['Date'] < fecha_limite)
    )
    partidos = df_wta[mask].sort_values('Date', ascending=False)
    if len(partidos) == 0:
        return 500
    ultimo = partidos.iloc[0]
    if ultimo['Player_1'] == jugadora:
        return ultimo['Rank_1']
    return ultimo['Rank_2']

def construir_features(p1, p2, superficie, ronda, fecha):
    rank_p1 = get_ranking(wta, p1, fecha)
    rank_p2 = get_ranking(wta, p2, fecha)
    row = {
        'surface': superficie,
        'round': ronda,
        'rank_diff': rank_p1 - rank_p2,
        'wins2meses_p1': forma_reciente(wta, p1, fecha),
        'wins2meses_p2': forma_reciente(wta, p2, fecha),
        'ratio_superficie_p1': winrate(wta, p1, fecha, superficie=superficie),
        'ratio_superficie_p2': winrate(wta, p2, fecha, superficie=superficie),
        'h2h': headtohead(wta, p1, p2, fecha),
        'ratio_ronda_p1': winrate(wta, p1, fecha, ronda=ronda),
        'ratio_ronda_p2': winrate(wta, p2, fecha, ronda=ronda),
        'experiencia_p1': experiencia(wta, p1, fecha),
        'experiencia_p2': experiencia(wta, p2, fecha),
        'tournament_type': 'GS'  # por defecto
    }
    return pd.DataFrame([row])

# ─── UI ───────────────────────────────────────────────────────────────────────
st.markdown('<p class="titulo">WTA Match<br>Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitulo">Machine Learning · WTA 2007–2026</p>', unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# Selectores de jugadoras
col1, col_vs, col2 = st.columns([5, 1, 5])

with col1:
    p1 = st.selectbox("Jugadora 1", jugadoras, index=0)

with col_vs:
    st.markdown('<p class="vs">VS</p>', unsafe_allow_html=True)

with col2:
    p2 = st.selectbox("Jugadora 2", jugadoras, index=1)

# Configuración del partido
col3, col4, col5 = st.columns(3)

with col3:
    superficie = st.selectbox("Superficie", ['Clay', 'Hard', 'Grass'])

with col4:
    ronda = st.selectbox("Ronda", [
        '1st Round', '2nd Round', '3rd Round', '4th Round',
        'Quarterfinals', 'Semifinals', 'The Final'
    ])

with col5:
    fecha = st.date_input("Fecha", value=date.today())

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# Botón de predicción
if st.button("🎾 Predecir resultado"):

    if p1 == p2:
        st.error("Selecciona dos jugadoras distintas.")
    else:
        with st.spinner("Calculando..."):
            fecha_pd = pd.Timestamp(fecha)
            X = construir_features(p1, p2, superficie, ronda, fecha_pd)

            proba = modelo.predict_proba(X)[0]
            prob_p1 = proba[1]
            prob_p2 = proba[0]

            ganadora = p1 if prob_p1 > prob_p2 else p2
            prob_ganadora = max(prob_p1, prob_p2)
            prob_perdedora = min(prob_p1, prob_p2)
            perdedora = p2 if ganadora == p1 else p1

        # Resultados
        col_g, col_p = st.columns(2)

        with col_g:
            st.markdown(f"""
            <div class="prob-box prob-ganadora">
                <div class="prob-nombre">🏆 {ganadora}</div>
                <div class="prob-numero">{prob_ganadora:.0%}</div>
                <div class="prob-label">Probabilidad de victoria</div>
            </div>
            """, unsafe_allow_html=True)

        with col_p:
            st.markdown(f"""
            <div class="prob-box prob-perdedora">
                <div class="prob-nombre">{perdedora}</div>
                <div class="prob-numero">{prob_perdedora:.0%}</div>
                <div class="prob-label">Probabilidad de victoria</div>
            </div>
            """, unsafe_allow_html=True)

        # Info adicional
        h2h_val = headtohead(wta, p1, p2, fecha_pd)
        forma_p1 = forma_reciente(wta, p1, fecha_pd)
        forma_p2 = forma_reciente(wta, p2, fecha_pd)

        st.markdown(f"""
        <div style="margin-top: 1.5rem; text-align: center">
            <span class="info-chip">H2H {p1.split()[0]}: {h2h_val:.0%}</span>
            <span class="info-chip">Forma {p1.split()[0]}: {forma_p1:.0%}</span>
            <span class="info-chip">Forma {p2.split()[0]}: {forma_p2:.0%}</span>
            <span class="info-chip">Superficie: {superficie}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="warning-box">
            Predicción basada en {experiencia(wta, p1, fecha_pd)} partidos históricos de {p1.split()[0]} 
            y {experiencia(wta, p2, fecha_pd)} de {p2.split()[0]} hasta {fecha}.
        </div>
        """, unsafe_allow_html=True)
