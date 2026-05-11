import streamlit as st
import pandas as pd
import numpy as np
import pickle
import random
from datetime import date
from features import (forma_reciente, winrate, headtohead, experiencia,
                      get_ranking, get_elo)

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
    .ronda-header {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.3rem;
        color: #c8f000;
        letter-spacing: 0.1em;
        margin-bottom: 0.4rem;
        margin-top: 0.5rem;
    }
    .partido-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #1e1e1e;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin-bottom: 0.3rem;
        font-size: 0.88rem;
    }
    .partido-ganadora { color: #f0f0f0; font-weight: 500; }
    .partido-perdedora { color: #555; }
    .partido-prob {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.15rem;
        color: #c8f000;
    }
    .mc-row {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 0.45rem;
        background: #1a1a1a;
        padding: 0.35rem 0.8rem;
        border-radius: 6px;
    }
    .mc-nombre {
        width: 160px;
        font-size: 0.88rem;
        color: #f0f0f0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .mc-bar-wrap {
        flex: 1;
        background: #2a2a2a;
        border-radius: 4px;
        height: 18px;
        overflow: hidden;
    }
    .mc-bar-fill {
        height: 100%;
        background: linear-gradient(90deg, #c8f000, #8ab800);
        border-radius: 4px;
    }
    .mc-pct {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.1rem;
        color: #c8f000;
        width: 52px;
        text-align: right;
    }
    .bracket-cell {
        background: #1e1e1e;
        border: 1px solid #2a2a2a;
        border-radius: 6px;
        padding: 0.4rem 0.7rem;
        font-size: 0.82rem;
        color: #f0f0f0;
        margin-bottom: 2px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .bracket-cell.ganadora {
        border-color: #c8f000;
        color: #c8f000;
        font-weight: 600;
    }
    .bracket-cell.perdedora {
        color: #555;
        border-color: #222;
    }
    .bracket-match {
        margin-bottom: 1rem;
    }
    .bracket-prob {
        font-size: 0.7rem;
        color: #666;
        text-align: right;
        margin-top: 2px;
        font-family: 'Bebas Neue', sans-serif;
        letter-spacing: 0.05em;
    }
    .campeonas-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 1.8rem;
        color: #c8f000;
        letter-spacing: 0.08em;
        margin-bottom: 0.8rem;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)
#----

# ─── Definición de funciones ────────────────────────────────────────────────────
#
# Dadas dos jugadoras calcular las features a fecha dada. Para el predictor de 1 partido
def construir_features(df_wta, p1, p2, superficie, ronda, fecha):
    rank_p1 = get_ranking(df_wta, p1, fecha)
    rank_p2 = get_ranking(df_wta, p2, fecha)
    elo_p1_global = get_elo (df_wta, p1, fecha)
    elo_p2_global = get_elo (df_wta, p2, fecha)
    elo_p1_superficie = get_elo (df_wta,p1,fecha,superficie)
    elo_p2_superficie = get_elo (df_wta,p2,fecha,superficie)

    row = {
        'surface': superficie,
        'round': ronda,
        'rank_diff': rank_p1 - rank_p2,
        'wins2meses_p1': forma_reciente(df_wta, p1, fecha),
        'wins2meses_p2': forma_reciente(df_wta, p2, fecha),
        'ratio_superficie_p1': winrate(df_wta, p1, fecha, superficie=superficie),
        'ratio_superficie_p2': winrate(df_wta, p2, fecha, superficie=superficie),
        'h2h': headtohead(df_wta, p1, p2, fecha),
        'ratio_ronda_p1': winrate(df_wta, p1, fecha, ronda=ronda),
        'ratio_ronda_p2': winrate(df_wta, p2, fecha, ronda=ronda),
        'experiencia_p1': experiencia(df_wta, p1, fecha),
        'experiencia_p2': experiencia(df_wta, p2, fecha),
        'tournament_type': 'GS',  # por defecto
        'elo_p1':          elo_p1_superficie,
        'elo_p2':          elo_p2_superficie,
        'elo_diff':        elo_p1_superficie - elo_p2_superficie,
        'elo_global_p1':   elo_p1_global,
        'elo_global_p2':   elo_p2_global,
        'elo_global_diff': elo_p1_global - elo_p2_global,
        'is_new_p1': int(experiencia(df_wta, p1, fecha) < 10),
        'is_new_p2': int(experiencia(df_wta, p2, fecha) < 10),
    }
    return pd.DataFrame([row])

# Para el cuadro de un torneo construir las features de todas las jugadoras y cachearlo
def calculo_features_jugadoras_torneo (df, cuadro):
    fecha = pd.to_datetime('2026-05-05')
    superficie = 'Clay'
    rondas = ['1st Round', '2nd Round', '3rd Round', '4th Round', 
              'Quarterfinals', 'Semifinals', 'The Final']
    
    cache_features = {}
    for jugadora in cuadro:

        # Calcular los valores
        wins2meses_val = forma_reciente(df, jugadora, fecha)
        ratio_superficie_val = winrate(df, jugadora, fecha, superficie=superficie)
        experiencia_val = experiencia(df, jugadora, fecha)
        ranking_val = get_ranking(df, jugadora, fecha)
        elo_global = get_elo (df, jugadora, fecha)
        elo_superficie = get_elo (df,jugadora,fecha,superficie)
        
        # Calcular winrate por ronda
        winrate_por_ronda = {}
        for ronda in rondas:
            winrate_por_ronda[ronda] = winrate(df, jugadora, fecha, ronda=ronda)
    
        cache_features[jugadora] = {
            'wins2meses': wins2meses_val,
            'ratio_superficie': ratio_superficie_val,
            'experiencia': experiencia_val,
            'ranking': ranking_val,
            'winrate_por_ronda': winrate_por_ronda,
            'elo_global':   elo_global,
            'elo_superficie': elo_superficie
        }

    # Y una caché para los h2h entre cada par
    cache_h2h = {}
    for i, p1 in enumerate(cuadro):
        for p2 in cuadro[i+1:]:
            cache_h2h[(p1, p2)] = headtohead(df, p1, p2, fecha)
            cache_h2h[(p2, p1)] = 1 - cache_h2h[(p1, p2)]
    
    return cache_features, cache_h2h

def construir_features_desde_cache (cache_features, cache_h2h, p1, p2, superficie, ronda):
    f1 = cache_features[p1]
    f2 = cache_features[p2]

    row = {
        'surface':             superficie,
        'round':               ronda,
        'rank_diff':           f1['ranking'] - f2['ranking'],
        'wins2meses_p1':       f1['wins2meses'],
        'wins2meses_p2':       f2['wins2meses'],
        'ratio_superficie_p1': f1['ratio_superficie'],
        'ratio_superficie_p2': f2['ratio_superficie'],
        'h2h':                 cache_h2h.get((p1, p2), 0.5),
        'ratio_ronda_p1':      f1['winrate_por_ronda'][ronda],
        'ratio_ronda_p2':      f2['winrate_por_ronda'][ronda],
        'experiencia_p1':      f1['experiencia'],
        'experiencia_p2':      f2['experiencia'],
        'tournament_type':     'GS',
        'elo_p1':          f1['elo_superficie'],
        'elo_p2':          f2['elo_superficie'],
        'elo_diff':        f1['elo_superficie'] - f2['elo_superficie'],
        'elo_global_p1':   f1['elo_global'],
        'elo_global_p2':   f2['elo_global'],
        'elo_global_diff': f1['elo_global'] - f2['elo_global'],
        'is_new_p1': int(f1['experiencia'] < 10),
        'is_new_p2': int(f2['experiencia'] < 10),

    }
    return pd.DataFrame([row])

# ─── Funciones de simulacion de partido y torneo ────────────────────────────────────────────────────
def simular_partido(prob_a):
  # lanzamos un dado cargado
  return random.random() < prob_a

def simular_torneo(cache_features, cache_h2h, cuadro, modelo):
    fecha = pd.to_datetime('2026-05-05')
    superficie = 'Clay'
    rondas = ['1st Round', '2nd Round', '3rd Round', '4th Round', 
              'Quarterfinals', 'Semifinals', 'The Final']
    
    jugadoras = cuadro.copy()
    indice_ronda = 0 

    while len(jugadoras) > 1:
        siguiente_ronda = []
        ronda_actual = rondas[indice_ronda]  # Nombre de la ronda actual
   
        # Emparejar jugadoras
        for i in range(0, len(jugadoras), 2):
            a, b = jugadoras[i], jugadoras[i+1]
            X = construir_features_desde_cache (cache_features, cache_h2h, a, b, superficie, ronda_actual)
            prob_a = modelo.predict_proba(X)[0][1]  # Clase 1: probabilidad de que gane p1(a)
            
            ganadora = a if simular_partido(prob_a) else b
            siguiente_ronda.append(ganadora)
        
        # Actualizar para la siguiente ronda
        jugadoras = siguiente_ronda
        indice_ronda += 1  
        

    return jugadoras[0]
# ---- Simulación Monte Carlo
# SIMULACIÓN MONTE CARLO
def simulacion_montecarlo(cache_features, cacheh2h, cuadro, modelo, n_simulaciones=10000):

    victorias = {}  # Diccionario vacío
    
    for _ in range(n_simulaciones):
        campeona = simular_torneo(cache_features, cacheh2h, cuadro, modelo)

        if campeona in victorias:
            victorias[campeona] += 1
        else:
            victorias[campeona] = 1
    
    # Calcular probabilidades
    probabilidades = {}
    for jugadora, wins in victorias.items():
        probabilidades[jugadora] = wins / n_simulaciones
    
    return probabilidades, victorias

# ─── Cargar datos, modelo y cuadro del torneo ─────────────────────────────────────────────────────
@st.cache_resource
def cargar_modelo():
    with open('gbx_v3.model', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def cargar_wta():
    return pd.read_csv('wta_limpio.csv', low_memory=False, parse_dates=['Date'])
#@st.cache_data
# def cargar_wta():
#     wta = pd.read_csv('wta_limpio.csv', low_memory=False)
#     wta['Date'] = pd.to_datetime(wta['Date'], errors='coerce')
#     wta = wta.dropna(subset=['Date'])
#     return wta

# wta = cargar_wta()

@st.cache_data
def cargar_historico():
    return pd.read_csv('historico_partidos.csv', parse_dates=['date'])

@st.cache_resource
def cargar_cache_torneo(wta):
    with open("cuadro_python_list.txt", "r", encoding="utf-8") as f:
        codigo = f.read()
    local_vars = {}
    exec(codigo, {}, local_vars)
    cuadro = local_vars['cuadro']
    cache_features, cache_h2h = calculo_features_jugadoras_torneo(wta, cuadro)
    return cuadro, cache_features, cache_h2h


modelo = cargar_modelo()
df = cargar_historico()
wta = cargar_wta()
cuadro, cache_features, cache_h2h = cargar_cache_torneo(wta)

# ─── Jugadoras activas (con partidos en el último año del dataset) ─────────────
@st.cache_data
def jugadoras_activas(df):
    fecha_max = df['date'].max()
    fecha_corte = fecha_max - pd.DateOffset(months=12)
    recientes = df[df['date'] >= fecha_corte]
    # Recuperar nombres originales desde el histórico usando match_id
    # Cargamos wta para sacar nombres
    try:
        wta = pd.read_csv('wta_limpio.csv', low_memory=False)
        ids_recientes = recientes['match_id'].values
        jugadoras = pd.concat([
            wta.loc[wta.index.isin(ids_recientes), 'Player_1'],
            wta.loc[wta.index.isin(ids_recientes), 'Player_2']
        ]).unique()
        return sorted(jugadoras)
    except:
        return []

jugadoras = jugadoras_activas(df)


# ─── UI ───────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Predictor de partido", "Simulador del torneo"])
with tab1:
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
                X = construir_features(wta, p1, p2, superficie, ronda, fecha_pd)

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
with tab2:
    st.markdown('<div class="titulo">Simulador del torneo</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitulo">Roland Garros 2025 · Cuadro femenino · Monte Carlo</div>', unsafe_allow_html=True)
 

    n_sims = st.selectbox(
        "Simulaciones Monte Carlo",
        options=[100, 500, 1000, 5000, 10000],
        index=2
    )

    lanzar = st.button("⚡ Simular torneo")
 
    if lanzar:
        rondas_nombres = ['1st Round', '2nd Round', '3rd Round', '4th Round',
                          'Quarterfinals', 'Semifinals', 'The Final']
        rondas_es = {
            '1st Round': 'Primera ronda',
            '2nd Round': 'Segunda ronda',
            '3rd Round': 'Tercera ronda',
            '4th Round': 'Cuarta ronda',
            'Quarterfinals': 'Cuartos de final',
            'Semifinals': 'Semifinales',
            'The Final': 'Final',
        }
        rondas_bracket = ['Quarterfinals', 'Semifinals', 'The Final']
        rondas_expander = ['1st Round', '2nd Round', '3rd Round', '4th Round']
        fecha = pd.to_datetime('2026-05-05')
        superficie = 'Clay'
 
        ## ── SIMULACIÓN RONDA A RONDA ─────────────────────────────────────────
        jugadoras = cuadro.copy()
        resultados_por_ronda = {}
 
        for ronda_actual in rondas_nombres:
            if len(jugadoras) <= 1:
                break
            siguiente_ronda = []
            partidos_ronda = []
 
            for i in range(0, len(jugadoras), 2):
                a, b = jugadoras[i], jugadoras[i + 1]
                X = construir_features_desde_cache(
                    cache_features, cache_h2h, a, b, superficie, ronda_actual
                )
                prob_a = modelo.predict_proba(X)[0][1]
                prob_b = 1 - prob_a
                ganadora = a if simular_partido(prob_a) else b
                perdedora = b if ganadora == a else a
                prob_gan = prob_a if ganadora == a else prob_b
 
                siguiente_ronda.append(ganadora)
                partidos_ronda.append({
                    'a': a, 'b': b,
                    'ganadora': ganadora,
                    'perdedora': perdedora,
                    'prob': prob_gan
                })
 
            resultados_por_ronda[ronda_actual] = partidos_ronda
            jugadoras = siguiente_ronda
 
        campeona_sim = jugadoras[0] if jugadoras else '—'
 
        ## ── RONDAS PREVIAS (expanders) ───────────────────────────────────────
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="campeonas-title">Desarrollo del torneo</div>', unsafe_allow_html=True)
 
        for ronda in rondas_expander:
            if ronda not in resultados_por_ronda:
                continue
            with st.expander(rondas_es[ronda], expanded=False):
                html = ""
                for p in resultados_por_ronda[ronda]:
                    html += f"""
                    <div class="partido-row">
                        <span class="partido-ganadora">🏆 {p['ganadora']}</span>
                        <span class="partido-perdedora">{p['perdedora']}</span>
                        <span class="partido-prob">{p['prob']*100:.0f}%</span>
                    </div>"""
                st.markdown(html, unsafe_allow_html=True)
 
        ## ── BRACKET QF / SF / FINAL ──────────────────────────────────────────
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
 
        rondas_bracket_presentes = [r for r in rondas_bracket if r in resultados_por_ronda]
        if rondas_bracket_presentes:
            cols = st.columns(len(rondas_bracket_presentes))
            for idx, ronda in enumerate(rondas_bracket_presentes):
                with cols[idx]:
                    st.markdown(
                        f'<div class="ronda-header">{rondas_es[ronda]}</div>',
                        unsafe_allow_html=True
                    )
                    for p in resultados_por_ronda[ronda]:
                        st.markdown(f"""
                        <div class="bracket-match">
                            <div class="bracket-cell ganadora">🏆 {p['ganadora']}</div>
                            <div class="bracket-cell perdedora">{p['perdedora']}</div>
                            <div class="bracket-prob">{p['prob']*100:.0f}% de victoria</div>
                        </div>""", unsafe_allow_html=True)
 
        ## ── CAMPEONA ─────────────────────────────────────────────────────────
        st.markdown(
            f'<div class="prob-box prob-ganadora" style="margin-top:1.5rem;">'
            f'<div class="prob-label">Campeona simulada</div>'
            f'<div class="prob-numero">{campeona_sim}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
 
        ## ── MONTE CARLO ──────────────────────────────────────────────────────
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown('<div class="campeonas-title">Probabilidades Monte Carlo</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="subtitulo">{n_sims:,} simulaciones · % de veces que ganó el torneo</div>',
            unsafe_allow_html=True
        )
 
        with st.spinner("Simulando..."):
            _, victorias = simulacion_montecarlo(
                cache_features, cache_h2h, cuadro, modelo, n_sims
            )
 
        top10 = sorted(victorias.items(), key=lambda x: x[1], reverse=True)[:10]
        max_wins = top10[0][1] if top10 else 1
        medallas = ["🥇", "🥈", "🥉"]
 
        bars_html = ""
        for i, (jugadora, wins) in enumerate(top10):
            pct = wins / n_sims * 100
            bar_pct = wins / max_wins * 100
            medal = medallas[i] if i < 3 else f"{i+1}."
            bars_html += f"""
            <div class="mc-row">
                <div class="mc-nombre">{medal} {jugadora}</div>
                <div class="mc-bar-wrap">
                    <div class="mc-bar-fill" style="width:{bar_pct:.1f}%"></div>
                </div>
                <div class="mc-pct">{pct:.1f}%</div>
            </div>"""
 
        st.markdown(bars_html, unsafe_allow_html=True)
