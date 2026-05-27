import pandas as pd
import numpy as np
from datetime import timedelta
from features import (forma_reciente, winrate, headtohead, experiencia,
                      get_ranking, get_elo)

# =============================================================================
# CONFIGURACIÓN - ajusta estas rutas
# =============================================================================
HISTORICO_RAW_PATH   = 'wta_limpio.csv'        # tu CSV raw desde 2007
FEATURES_PATH        = 'historico_partidos.csv'  # tu dataset de features
NUEVOS_PATH          = 'wta_matches_may2026.csv'   # el CSV que acabamos de generar
OUTPUT_PATH          = 'features_actualizado.csv'

K_FACTOR             = 32    # ajusta al K que uses tú
ELO_INICIAL          = 1500
VENTANA_DIAS         = 60    # "últimos 2 meses"
MIN_PARTIDOS_IS_NEW  = 10
# =============================================================================


# -----------------------------------------------------------------------------
# 1. CARGA DE DATOS
# -----------------------------------------------------------------------------
print("Cargando datos...")
raw      = pd.read_csv(HISTORICO_RAW_PATH, parse_dates=['Date'])
features = pd.read_csv(FEATURES_PATH)
nuevos   = pd.read_csv(NUEVOS_PATH, parse_dates=['Date'])

# Normalizar nombres de columnas del raw al esquema que usa el script
# Ajusta estos nombres si en tu raw son distintos
raw = raw.rename(columns={
    'Player_1': 'player_1',
    'Player_2': 'player_2',
    'Winner':   'winner',
    'Surface':  'surface',
    'Round':    'round',
    'Rank_1':   'rank_1',
    'Rank_2':   'rank_2',
    'Date':     'date',
    'Tournament': 'tournament',
})
nuevos = nuevos.rename(columns={
    'Player_1':   'player_1',
    'Player_2':   'player_2',
    'Winner':     'winner',
    'Surface':    'surface',
    'Round':      'round',
    'Rank_1':     'rank_1',
    'Rank_2':     'rank_2',
    'Date':       'date',
    'Tournament': 'tournament',
})

raw['date']    = pd.to_datetime(raw['date'])
nuevos['date'] = pd.to_datetime(nuevos['date'])
raw = raw.sort_values('date').reset_index(drop=True)
nuevos = nuevos.sort_values('date').reset_index(drop=True)

CUTOFF = pd.Timestamp('2026-05-02')


# -----------------------------------------------------------------------------
# 2. RECONSTRUIR ELO FINAL A FECHA CUTOFF
#    Partimos del ELO implícito en la última aparición de cada jugadora
#    en el dataset de features, que ya está calculado correctamente.
# -----------------------------------------------------------------------------
print("Reconstruyendo estado ELO al 2 de mayo...")

# El dataset de features tiene player_1/player_2 — necesitamos el nombre
# tal como aparece en el raw para hacer el join
# Asumimos que features tiene columnas 'Player_1','Player_2','elo_p1','elo_p2',
# 'elo_global_p1','elo_global_p2' y 'Surface'
# Ajusta si tus columnas se llaman distinto

feat_cols_needed = ['Player_1','Player_2','Surface','Date',
                    'elo_p1','elo_p2','elo_global_p1','elo_global_p2']

# Construimos diccionarios {jugadora: {superficie: elo}} desde features
elo_superficie = {}   # {player: {surface: elo}}
elo_global     = {}   # {player: elo_global}

features_sorted = features.sort_values('Date') if 'Date' in features.columns else features

for _, row in features_sorted.iterrows():
    p1, p2 = row['Player_1'], row['Player_2']
    surf   = row['Surface']

    if p1 not in elo_superficie:
        elo_superficie[p1] = {}
    if p2 not in elo_superficie:
        elo_superficie[p2] = {}

    elo_superficie[p1][surf]  = row['elo_p1']
    elo_superficie[p2][surf]  = row['elo_p2']
    elo_global[p1]            = row['elo_global_p1']
    elo_global[p2]            = row['elo_global_p2']


def get_elo(player, surface):
    return elo_superficie.get(player, {}).get(surface, ELO_INICIAL)

def get_elo_global(player):
    return elo_global.get(player, ELO_INICIAL)


# -----------------------------------------------------------------------------
# 3. FUNCIONES DE FEATURES
# -----------------------------------------------------------------------------

def expected_score(elo_a, elo_b):
    return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

def update_elo(elo_ganador, elo_perdedor, k=K_FACTOR):
    exp_g = expected_score(elo_ganador, elo_perdedor)
    exp_p = expected_score(elo_perdedor, elo_ganador)
    nuevo_g = elo_ganador + k * (1 - exp_g)
    nuevo_p = elo_perdedor + k * (0 - exp_p)
    return nuevo_g, nuevo_p

def wins_ultimos_2meses(player, fecha, historico):
    """Win rate en los últimos VENTANA_DIAS días antes del partido."""
    ventana = historico[
        (historico['date'] < fecha) &
        (historico['date'] >= fecha - timedelta(days=VENTANA_DIAS)) &
        ((historico['player_1'] == player) | (historico['player_2'] == player))
    ]
    if len(ventana) == 0:
        return 0.0
    wins = (ventana['winner'] == player).sum()
    return wins / len(ventana)

def ratio_superficie(player, surface, historico):
    """Win rate histórico total en esa superficie."""
    partidos = historico[
        (historico['date'] < historico['date'].max()) &  # todo el histórico previo
        (historico['surface'] == surface) &
        ((historico['player_1'] == player) | (historico['player_2'] == player))
    ]
    if len(partidos) == 0:
        return 0.5
    wins = (partidos['winner'] == player).sum()
    return wins / len(partidos)

def ratio_superficie_hasta(player, surface, fecha, historico):
    """Win rate histórico en esa superficie hasta fecha."""
    partidos = historico[
        (historico['date'] < fecha) &
        (historico['surface'] == surface) &
        ((historico['player_1'] == player) | (historico['player_2'] == player))
    ]
    if len(partidos) == 0:
        return 0.5
    wins = (partidos['winner'] == player).sum()
    return wins / len(partidos)

def ratio_ronda_hasta(player, ronda, fecha, historico):
    """Win rate histórico en esa ronda específica hasta fecha."""
    partidos = historico[
        (historico['date'] < fecha) &
        (historico['round'] == ronda) &
        ((historico['player_1'] == player) | (historico['player_2'] == player))
    ]
    if len(partidos) == 0:
        return 0.5
    wins = (partidos['winner'] == player).sum()
    return wins / len(partidos)

def calc_h2h(p1, p2, fecha, historico):
    """
    H2H de p1 frente a p2 hasta fecha.
    Retorna fracción de victorias de p1 (0.5 si no hay histórico).
    """
    enfrentamientos = historico[
        (historico['date'] < fecha) &
        (
            ((historico['player_1'] == p1) & (historico['player_2'] == p2)) |
            ((historico['player_1'] == p2) & (historico['player_2'] == p1))
        )
    ]
    if len(enfrentamientos) == 0:
        return 0.5
    wins_p1 = (enfrentamientos['winner'] == p1).sum()
    return wins_p1 / len(enfrentamientos)

def experiencia_hasta(player, fecha, historico):
    """Número total de partidos jugados hasta fecha."""
    return int(((historico['date'] < fecha) &
                ((historico['player_1'] == player) | (historico['player_2'] == player))).sum())

def is_new(player, fecha, historico):
    return int(experiencia_hasta(player, fecha, historico) < MIN_PARTIDOS_IS_NEW)


# -----------------------------------------------------------------------------
# 4. COMBINAR HISTÓRICO RAW + NUEVOS PARA LAS CONSULTAS
#    Los nuevos partidos se van incorporando al histórico conforme se procesan,
#    para que las features de partidos posteriores los vean.
# -----------------------------------------------------------------------------
print("Calculando features para los 149 partidos nuevos...")

# Histórico acumulado: empieza con el raw completo hasta cutoff
historico_acumulado = raw[raw['date'] <= CUTOFF].copy()

nuevas_filas = []

for i, partido in nuevos.iterrows():
    p1      = partido['player_1']
    p2      = partido['player_2']
    winner  = partido['winner']
    fecha   = partido['date']
    surface = partido['surface']
    ronda   = partido['round']
    rank_1  = partido['rank_1']
    rank_2  = partido['rank_2']

    # --- ELO antes del partido ---
    elo_p1_pre        = get_elo(p1, surface)
    elo_p2_pre        = get_elo(p2, surface)
    elo_global_p1_pre = get_elo_global(p1)
    elo_global_p2_pre = get_elo_global(p2)

    # --- Resto de features ---
    rank_diff       = (rank_1 - rank_2) if pd.notna(rank_1) and pd.notna(rank_2) else 0.0
    w2m_p1          = wins_ultimos_2meses(p1, fecha, historico_acumulado)
    w2m_p2          = wins_ultimos_2meses(p2, fecha, historico_acumulado)
    rat_sup_p1      = ratio_superficie_hasta(p1, surface, fecha, historico_acumulado)
    rat_sup_p2      = ratio_superficie_hasta(p2, surface, fecha, historico_acumulado)
    h2h             = calc_h2h(p1, p2, fecha, historico_acumulado)
    rat_ronda_p1    = ratio_ronda_hasta(p1, ronda, fecha, historico_acumulado)
    rat_ronda_p2    = ratio_ronda_hasta(p2, ronda, fecha, historico_acumulado)
    exp_p1          = experiencia_hasta(p1, fecha, historico_acumulado)
    exp_p2          = experiencia_hasta(p2, fecha, historico_acumulado)
    isnew_p1        = is_new(p1, fecha, historico_acumulado)
    isnew_p2        = is_new(p2, fecha, historico_acumulado)
    target          = 1 if winner == p1 else 0

    nuevas_filas.append({
        'Tournament':       partido['tournament'],
        'Date':             fecha,
        'Surface':          surface,
        'Round':            ronda,
        'Player_1':         p1,
        'Player_2':         p2,
        'Winner':           winner,
        'Rank_1':           rank_1,
        'Rank_2':           rank_2,
        'rank_diff':        rank_diff,
        'wins2meses_p1':    w2m_p1,
        'wins2meses_p2':    w2m_p2,
        'ratio_superficie_p1': rat_sup_p1,
        'ratio_superficie_p2': rat_sup_p2,
        'h2h':              h2h,
        'ratio_ronda_p1':   rat_ronda_p1,
        'ratio_ronda_p2':   rat_ronda_p2,
        'experiencia_p1':   exp_p1,
        'experiencia_p2':   exp_p2,
        'target':           target,
        'is_new_p1':        isnew_p1,
        'is_new_p2':        isnew_p2,
        'elo_p1':           elo_p1_pre,
        'elo_p2':           elo_p2_pre,
        'elo_diff':         elo_p1_pre - elo_p2_pre,
        'elo_global_p1':    elo_global_p1_pre,
        'elo_global_p2':    elo_global_p2_pre,
        'elo_global_diff':  elo_global_p1_pre - elo_global_p2_pre,
    })

    # --- Actualizar ELO DESPUÉS del partido ---
    ganador  = p1 if winner == p1 else p2
    perdedor = p2 if winner == p1 else p1

    elo_g_surf, elo_p_surf = update_elo(get_elo(ganador, surface), get_elo(perdedor, surface))
    elo_g_glob, elo_p_glob = update_elo(get_elo_global(ganador), get_elo_global(perdedor))

    if ganador not in elo_superficie: elo_superficie[ganador] = {}
    if perdedor not in elo_superficie: elo_superficie[perdedor] = {}

    elo_superficie[ganador][surface] = elo_g_surf
    elo_superficie[perdedor][surface] = elo_p_surf
    elo_global[ganador] = elo_g_glob
    elo_global[perdedor] = elo_p_glob

    # Incorporar partido al histórico acumulado para los siguientes
    historico_acumulado = pd.concat([
        historico_acumulado,
        pd.DataFrame([{
            'date': fecha, 'player_1': p1, 'player_2': p2,
            'winner': winner, 'surface': surface, 'round': ronda,
            'rank_1': rank_1, 'rank_2': rank_2, 'tournament': partido['tournament']
        }])
    ], ignore_index=True)

    if i % 25 == 0:
        print(f"  Procesados {i+1}/{len(nuevos)} partidos...")


# -----------------------------------------------------------------------------
# 5. CONCATENAR Y GUARDAR
# -----------------------------------------------------------------------------
print("Concatenando y guardando...")

df_nuevos   = pd.DataFrame(nuevas_filas)
df_completo = pd.concat([features, df_nuevos], ignore_index=True)
df_completo.to_csv(OUTPUT_PATH, index=False)

print(f"\nListo. Dataset actualizado: {len(df_completo)} filas -> '{OUTPUT_PATH}'")
print(f"  Filas originales : {len(features)}")
print(f"  Filas nuevas     : {len(df_nuevos)}")
