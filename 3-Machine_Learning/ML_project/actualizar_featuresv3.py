import pandas as pd
import numpy as np

# ── Rutas ──────────────────────────────────────────────────────────────────
WTA_LIMPIO_PATH = 'wta_limpio.csv'
NUEVOS_PATH     = 'wta_matches_may2026.csv'
OUTPUT_PATH     = 'wta_limpio_actualizado.csv'
K_FACTOR        = 32
# ───────────────────────────────────────────────────────────────────────────

wta     = pd.read_csv(WTA_LIMPIO_PATH, parse_dates=['Date'])
nuevos  = pd.read_csv(NUEVOS_PATH, parse_dates=['Date'])
nuevos  = nuevos.sort_values('Date').reset_index(drop=True)

# ELO helpers
def expected_score(a, b):
    return 1 / (1 + 10 ** ((b - a) / 400))

def update_elo(elo_g, elo_p, k=K_FACTOR):
    e = expected_score(elo_g, elo_p)
    return elo_g + k * (1 - e), elo_p + k * (0 - (1 - e))

def get_elo_actual(df, jugadora, superficie=None):
    """Última fila donde aparece la jugadora, devuelve su ELO."""
    col_p1 = 'elo_p1' if superficie else 'elo_global_p1'
    col_p2 = 'elo_p2' if superficie else 'elo_global_p2'
    if superficie:
        mask = (df['Surface'] == superficie)
    else:
        mask = pd.Series(True, index=df.index)
    mask &= ((df['Player_1'] == jugadora) | (df['Player_2'] == jugadora))
    partidos = df[mask].sort_values('Date', ascending=False)
    if len(partidos) == 0:
        return 1500.0
    ultimo = partidos.iloc[0]
    return ultimo[col_p1] if ultimo['Player_1'] == jugadora else ultimo[col_p2]

# Acumulado en memoria
historico = wta.copy()
nuevas_filas = []

for _, partido in nuevos.iterrows():
    p1      = partido['Player_1']
    p2      = partido['Player_2']
    winner  = partido['Winner']
    fecha   = partido['Date']
    surface = partido['Surface']

    # ELO antes del partido
    elo_p1_surf   = get_elo_actual(historico, p1, superficie=surface)
    elo_p2_surf   = get_elo_actual(historico, p2, superficie=surface)
    elo_p1_global = get_elo_actual(historico, p1)
    elo_p2_global = get_elo_actual(historico, p2)

    # Actualizar ELO tras el partido
    ganador, perdedor = (p1, p2) if winner == p1 else (p2, p1)
    elo_g_s, elo_p_s = update_elo(get_elo_actual(historico, ganador, surface),
                                   get_elo_actual(historico, perdedor, surface))
    elo_g_g, elo_p_g = update_elo(get_elo_actual(historico, ganador),
                                   get_elo_actual(historico, perdedor))

    # Asignar correctamente a p1/p2
    elo_p1_surf_post   = elo_g_s if winner == p1 else elo_p_s
    elo_p2_surf_post   = elo_p_s if winner == p1 else elo_g_s
    elo_p1_global_post = elo_g_g if winner == p1 else elo_p_g
    elo_p2_global_post = elo_p_g if winner == p1 else elo_g_g

    nuevas_filas.append({
        'Date':             fecha,
        'Surface':          surface,
        'Round':            partido['Round'],
        'Player_1':         p1,
        'Player_2':         p2,
        'Winner':           winner,
        'Rank_1':           partido['Rank_1'],
        'Rank_2':           partido['Rank_2'],
        'Pts_1':            None,
        'Pts_2':            None,
        'Odd_1':            None,
        'Odd_2':            None,
        'Score':            partido.get('Score', None),
        'tournament_type':  'GS' if partido['Tournament'] == 'Roland Garros' else 'WTA250',
        'prob_1':           None,
        'prob_2':           None,
        'target':           1 if winner == p1 else 0,
        'elo_p1':           elo_p1_surf_post,
        'elo_p2':           elo_p2_surf_post,
        'elo_diff':         elo_p1_surf_post - elo_p2_surf_post,
        'elo_global_p1':    elo_p1_global_post,
        'elo_global_p2':    elo_p2_global_post,
        'elo_global_diff':  elo_p1_global_post - elo_p2_global_post,
    })

    # Añadir al histórico en memoria
    historico = pd.concat([historico, pd.DataFrame([nuevas_filas[-1]])], ignore_index=True)

df_nuevos   = pd.DataFrame(nuevas_filas)
df_completo = pd.concat([wta, df_nuevos], ignore_index=True)
df_completo.to_csv(OUTPUT_PATH, index=False)

print(f"Listo: {len(wta)} filas originales + {len(df_nuevos)} nuevas = {len(df_completo)} total")