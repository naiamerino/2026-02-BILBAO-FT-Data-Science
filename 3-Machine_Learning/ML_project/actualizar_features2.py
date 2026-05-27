import pandas as pd
import numpy as np
from features import forma_reciente, winrate, headtohead, experiencia, get_elo

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
HISTORICO_RAW_PATH = 'wta_limpio.csv'
FEATURES_PATH      = 'historico_partidos.csv'
NUEVOS_PATH        = 'wta_matches_may2026.csv'
OUTPUT_PATH        = 'features_actualizado.csv'
K_FACTOR           = 32
MIN_PARTIDOS       = 10
# =============================================================================

# -----------------------------------------------------------------------------
# 1. CARGA
# -----------------------------------------------------------------------------
raw      = pd.read_csv(HISTORICO_RAW_PATH, parse_dates=['Date'])
features = pd.read_csv(FEATURES_PATH, parse_dates=['date'])
nuevos   = pd.read_csv(NUEVOS_PATH, parse_dates=['Date'])
nuevos   = nuevos.sort_values('Date').reset_index(drop=True)

# -----------------------------------------------------------------------------
# 2. HISTÓRICO ACUMULADO
# El punto de partida para las consultas es el raw completo + el dataset
# de features (que tiene los ELOs calculados). Las funciones de features.py
# leen siempre de este dataframe, que se va ampliando partido a partido.
# -----------------------------------------------------------------------------
# Para get_elo necesitamos el df de features (tiene columnas elo_p1 etc.)
# Para el resto de funciones usamos el raw (tiene todos los partidos)
historico_raw      = raw.copy()
historico_features = features.copy()

# -----------------------------------------------------------------------------
# 3. FUNCIONES ELO (no están en tu features.py, las definimos aquí)
# -----------------------------------------------------------------------------
def expected_score(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

def update_elo(elo_ganador, elo_perdedor, k=K_FACTOR):
    exp_g = expected_score(elo_ganador, elo_perdedor)
    exp_p = expected_score(elo_perdedor, elo_ganador)
    return elo_ganador + k * (1 - exp_g), elo_perdedor + k * (0 - exp_p)

# -----------------------------------------------------------------------------
# 4. BUCLE PRINCIPAL
# -----------------------------------------------------------------------------
nuevas_filas = []

for _, partido in nuevos.iterrows():
    p1      = partido['Player_1']
    p2      = partido['Player_2']
    winner  = partido['Winner']
    fecha   = partido['Date']
    surface = partido['Surface']
    ronda   = partido['Round']
    rank_1  = partido['Rank_1']
    rank_2  = partido['Rank_2']

    # ELO antes del partido — se lee del historico_features igual que en tu app
    elo_p1        = get_elo(historico_features, p1, fecha, tipo='superficie')
    elo_p2        = get_elo(historico_features, p2, fecha, tipo='superficie')
    elo_global_p1 = get_elo(historico_features, p1, fecha, tipo='global')
    elo_global_p2 = get_elo(historico_features, p2, fecha, tipo='global')

    # Resto de features — se leen del historico_raw igual que en tu notebook
    rank_diff      = (rank_1 - rank_2) if pd.notna(rank_1) and pd.notna(rank_2) else 0.0
    w2m_p1         = forma_reciente(historico_raw, p1, fecha)
    w2m_p2         = forma_reciente(historico_raw, p2, fecha)
    rat_sup_p1     = winrate(historico_raw, p1, fecha, superficie=surface)
    rat_sup_p2     = winrate(historico_raw, p2, fecha, superficie=surface)
    h2h            = headtohead(historico_raw, p1, p2, fecha)
    rat_ronda_p1   = winrate(historico_raw, p1, fecha, ronda=ronda)
    rat_ronda_p2   = winrate(historico_raw, p2, fecha, ronda=ronda)
    exp_p1         = experiencia(historico_raw, p1, fecha)
    exp_p2         = experiencia(historico_raw, p2, fecha)
    isnew_p1       = int(exp_p1 < MIN_PARTIDOS)
    isnew_p2       = int(exp_p2 < MIN_PARTIDOS)
    target         = 1 if winner == p1 else 0

    nuevas_filas.append({
        'Tournament':          partido['Tournament'],
        'Date':                fecha,
        'Surface':             surface,
        'Round':               ronda,
        'Player_1':            p1,
        'Player_2':            p2,
        'Winner':              winner,
        'Rank_1':              rank_1,
        'Rank_2':              rank_2,
        'rank_diff':           rank_diff,
        'wins2meses_p1':       w2m_p1,
        'wins2meses_p2':       w2m_p2,
        'ratio_superficie_p1': rat_sup_p1,
        'ratio_superficie_p2': rat_sup_p2,
        'h2h':                 h2h,
        'ratio_ronda_p1':      rat_ronda_p1,
        'ratio_ronda_p2':      rat_ronda_p2,
        'experiencia_p1':      exp_p1,
        'experiencia_p2':      exp_p2,
        'target':              target,
        'is_new_p1':           isnew_p1,
        'is_new_p2':           isnew_p2,
        'elo_p1':              elo_p1,
        'elo_p2':              elo_p2,
        'elo_diff':            elo_p1 - elo_p2,
        'elo_global_p1':       elo_global_p1,
        'elo_global_p2':       elo_global_p2,
        'elo_global_diff':     elo_global_p1 - elo_global_p2,
    })

# Actualizar ELO después del partido
    ganador  = p1 if winner == p1 else p2
    perdedor = p2 if winner == p1 else p1
    elo_g_s, elo_p_s = update_elo(get_elo(historico_features, ganador, fecha, 'superficie'),
                                   get_elo(historico_features, perdedor, fecha, 'superficie'))
    elo_g_g, elo_p_g = update_elo(get_elo(historico_features, ganador, fecha, 'global'),
                                   get_elo(historico_features, perdedor, fecha, 'global'))

    # Asignar correctamente a p1/p2 
    elo_p1_post        = elo_g_s if winner == p1 else elo_p_s
    elo_p2_post        = elo_p_s if winner == p1 else elo_g_s
    elo_global_p1_post = elo_g_g if winner == p1 else elo_p_g
    elo_global_p2_post = elo_p_g if winner == p1 else elo_g_g


    # Añadir la fila nueva a ambos históricos para que los siguientes partidos la vean
    fila_raw = pd.DataFrame([{
        'Date': fecha, 'Player_1': p1, 'Player_2': p2,
        'Winner': winner, 'Surface': surface, 'Round': ronda,
        'Rank_1': rank_1, 'Rank_2': rank_2, 'Tournament': partido['Tournament']
    }])

    fila_features = pd.DataFrame([{
        'Date':          fecha,
        'Player_1':      p1,
        'Player_2':      p2,
        'Winner':        winner,
        'Surface':       surface,
        'elo_p1':        elo_p1_post,
        'elo_p2':        elo_p2_post,
        'elo_global_p1': elo_global_p1_post,
        'elo_global_p2': elo_global_p2_post,
    }])

    historico_raw      = pd.concat([historico_raw, fila_raw], ignore_index=True)
    historico_features = pd.concat([historico_features, fila_features], ignore_index=True)

# -----------------------------------------------------------------------------
# 5. GUARDAR
# -----------------------------------------------------------------------------
df_nuevos   = pd.DataFrame(nuevas_filas)
df_completo = pd.concat([features, df_nuevos], ignore_index=True)
df_completo.to_csv(OUTPUT_PATH, index=False)
historico_raw.to_csv('wta_limpio2', index=False)

print(f"Listo: {len(features)} filas originales + {len(df_nuevos)} nuevas = {len(df_completo)} total")