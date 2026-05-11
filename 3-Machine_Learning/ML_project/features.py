import pandas as pd


# ─────────────────────────────────────────────
# FUNCIONES DE FEATURE ENGINEERING
# Usadas en: 01_EDA, 02_Modelo_ML, 03_SimulacionTorneo, Streamlit
# ─────────────────────────────────────────────

def forma_reciente(df, jugadora, fecha_limite, meses=2):
    """Win rate de una jugadora en los N meses anteriores a la fecha.
    Devuelve 0 si no tiene partidos (lesión o pausa larga)"""
    fecha_inicio = fecha_limite - pd.DateOffset(months=meses)
    mask = (
        ((df['Player_1'] == jugadora) | (df['Player_2'] == jugadora)) &
        (df['Date'] < fecha_limite) &
        (df['Date'] >= fecha_inicio)
    )
    partidos = df[mask]
    if len(partidos) == 0:
        return 0
    victorias = (partidos['Winner'] == jugadora).sum()
    return victorias / len(partidos)


def winrate(df, jugadora, fecha_limite, superficie=None, ronda=None):
    """Win rate histórico de una jugadora, filtrable por superficie y/o ronda.
    Devuelve 0.4 si no tiene historial (ligeramente por debajo de neutro = novata en esa condición)"""
    mask = (
        ((df['Player_1'] == jugadora) | (df['Player_2'] == jugadora)) &
        (df['Date'] < fecha_limite)
    )
    if superficie:
        mask &= (df['Surface'] == superficie)
    if ronda:
        mask &= (df['Round'] == ronda)
    partidos = df[mask]
    if len(partidos) == 0:
        return 0.4
    victorias = (partidos['Winner'] == jugadora).sum()
    return victorias / len(partidos)


def headtohead(df, p1, p2, fecha_limite):
    """% de victorias de p1 sobre p2 en enfrentamientos directos previos a la fecha.
    Devuelve 0.5 si nunca se han enfrentado (neutro)"""
    mask = (
        (
            ((df['Player_1'] == p1) & (df['Player_2'] == p2)) |
            ((df['Player_1'] == p2) & (df['Player_2'] == p1))
        ) & (df['Date'] < fecha_limite)
    )
    partidos = df[mask]
    if len(partidos) == 0:
        return 0.5
    victorias_p1 = (partidos['Winner'] == p1).sum()
    return victorias_p1 / len(partidos)


def experiencia(df, jugadora, fecha_limite):
    """Número total de partidos jugados por la jugadora antes de la fecha"""
    mask = (
        ((df['Player_1'] == jugadora) | (df['Player_2'] == jugadora)) &
        (df['Date'] < fecha_limite)
    )
    return df[mask].shape[0]


def get_ranking(df_wta, jugadora, fecha_limite):
    """Ranking más reciente de una jugadora antes de la fecha límite.
    Devuelve 500 si no tiene historial."""
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


def get_elo(df_wta, jugadora, fecha_limite, tipo='superficie'):
    """ELO más reciente de una jugadora antes de la fecha límite.
    tipo='superficie' → elo_p1/elo_p2
    tipo='global'     → elo_global_p1/elo_global_p2
    Devuelve 1500 (ELO inicial) si no tiene historial."""
    col_p1 = 'elo_p1' if tipo == 'superficie' else 'elo_global_p1'
    col_p2 = 'elo_p2' if tipo == 'superficie' else 'elo_global_p2'

    mask = (
        ((df_wta['Player_1'] == jugadora) | (df_wta['Player_2'] == jugadora)) &
        (df_wta['Date'] < fecha_limite)
    )
    partidos = df_wta[mask].sort_values('Date', ascending=False)
    if len(partidos) == 0:
        return 1500
    ultimo = partidos.iloc[0]
    if ultimo['Player_1'] == jugadora:
        return ultimo[col_p1]
    return ultimo[col_p2]


# ─────────────────────────────────────────────
# FUNCIONES DE CÁLCULO DE ELO
# Usadas en: 01_EDA (para construir historico_partidos)
# NO llamar en simulación — usar get_elo() en su lugar
# ─────────────────────────────────────────────

def calcular_elo_global(df, k=32, elo_inicial=1500):
    """
    Precalcula ELO global (independiente de superficie) para cada partido.
    Añade elo_global_p1, elo_global_p2, elo_global_diff al dataframe.

    IMPORTANTE: df debe estar ordenado por fecha ascendente antes de llamar esto.
    """
    elo_ratings = {}

    def get_elo_interno(jugadora):
        return elo_ratings.get(jugadora, elo_inicial)

    def expected(elo_a, elo_b):
        return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

    elo_p1_list, elo_p2_list = [], []

    for _, row in df.iterrows():
        p1        = row['Player_1']
        p2        = row['Player_2']
        resultado = row['target']

        elo_p1 = get_elo_interno(p1)
        elo_p2 = get_elo_interno(p2)

        elo_p1_list.append(elo_p1)
        elo_p2_list.append(elo_p2)

        exp_p1 = expected(elo_p1, elo_p2)

        elo_ratings[p1] = elo_p1 + k * (resultado - exp_p1)
        elo_ratings[p2] = elo_p2 + k * ((1 - resultado) - (1 - exp_p1))

    df = df.copy()
    df['elo_global_p1']   = elo_p1_list
    df['elo_global_p2']   = elo_p2_list
    df['elo_global_diff'] = df['elo_global_p1'] - df['elo_global_p2']

    return df


def calcular_elo_superficie(df, k=32, elo_inicial=1500):
    """
    Precalcula ELO por superficie para cada partido.
    Añade elo_p1, elo_p2, elo_diff al dataframe.

    IMPORTANTE: df debe estar ordenado por fecha ascendente antes de llamar esto.
    El ELO de cada partido = ELO ANTES de jugarlo (sin leakage).
    """
    elo_ratings = {}

    def get_elo_interno(jugadora, superficie):
        return elo_ratings.get(jugadora, {}).get(superficie, elo_inicial)

    def update_elo(jugadora, superficie, nuevo_valor):
        if jugadora not in elo_ratings:
            elo_ratings[jugadora] = {}
        elo_ratings[jugadora][superficie] = nuevo_valor

    def expected(elo_a, elo_b):
        return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

    elo_p1_list, elo_p2_list = [], []

    for _, row in df.iterrows():
        p1         = row['Player_1']
        p2         = row['Player_2']
        superficie = row['Surface']
        resultado  = row['target']

        elo_p1 = get_elo_interno(p1, superficie)
        elo_p2 = get_elo_interno(p2, superficie)

        elo_p1_list.append(elo_p1)
        elo_p2_list.append(elo_p2)

        exp_p1 = expected(elo_p1, elo_p2)
        exp_p2 = 1 - exp_p1

        update_elo(p1, superficie, elo_p1 + k * (resultado - exp_p1))
        update_elo(p2, superficie, elo_p2 + k * ((1 - resultado) - exp_p2))

    df = df.copy()
    df['elo_p1']   = elo_p1_list
    df['elo_p2']   = elo_p2_list
    df['elo_diff'] = df['elo_p1'] - df['elo_p2']

    return df
