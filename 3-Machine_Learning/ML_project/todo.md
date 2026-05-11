# Notas preliminares

Dataset base: resultados de partidos WTA desde 2007 hasta ahora
Entrenar modelo partido a partido (quien gana y con qué porcentaje)
Validar simulando el último grand slam y los dos torneos previos de tierra por ejemplo?
Simular el próximo Roland Garros

Tengo datos desde 2007 hasta 2026 (actualizados diariamente)
Entrenar con todos los datos desde 2007
Ponderar más los últimos 3-5 años?




```python
# Ejemplo simple de peso temporal. Por si hay que aplicar a alguna feature
import numpy as np

años_atras = 2024 - año_partido
peso = np.exp(-0.1 * años_atras)  # decae exponencialmente
```

Para cada partido del cuadro → predice probabilidad A gana B
→ avanza el ganador (o haz 10.000 sorteos ponderados)
→ cuenta cuántas veces gana cada jugadora el torneo

La simulación de Monte Carlo suena intimidante pero con el modelo ya entrenado es sorprendentemente sencilla — básicamente un bucle que repite el torneo miles de veces usando las probabilidades que predice tu modelo en cada partido.

Montecarlo
![image.png](000_Notas_limpio_files/image.png)

Simulación del cuadro, llamo al predictor en cada partido, cargo la predicción obtenida

Clasificador  →  predict_proba()  →  probabilidad  →  dado cargado  →  ganadora
  (entrena)        (infiere)          [0.73, 0.27]     random()         Swiatek

┌─────────────────────────────────────────┐
│           SIMULACIÓN MONTE CARLO        │  ← repite 10.000 veces
│                                         │
│   Cuartos → Semis → Final               │
│   para cada partido llama a...          │
│                                         │
│   ┌─────────────────────────────────┐   │
│   │         MODELO ML               │   │  ← entrenado con datos históricos
│   │  input: jugadora A vs B         │   │
│   │  output: probabilidad 0..1      │   │
│   └─────────────────────────────────┘   │
│                                         │
└─────────────────────────────────────────┘
         ↓ resultado final
   {"Swiatek": 38%, "Sabalenka": 22%, ...}

1. PREPARAR DATASET
   wta.csv → calcular features → df_modelo (guardar como CSV)

2. ENTRENAR
   df_modelo[2007:2023] → train → Random Forest / XGBoost
   df_modelo[2024:2025] → test  → accuracy + comparar con odds

3. SIMULAR ROLAND GARROS
   cuadro Australian Open 2025 (adaptado)
   + ranking WTA actual
   + forma_reciente calculada con fecha de hoy
   → Monte Carlo 10.000 veces
   → probabilidades por jugadora

4. (OPCIONAL) STREAMLIT
   Input: dos jugadoras + superficie
   Output: probabilidades + simulación torneo

Fase 1 — entrenamiento: aprende de 44.000 partidos históricos. Esto pasa una vez.
Fase 2 — inferencia: dado un cuadro real, preguntas al modelo miles de veces. Esto pasa cuando quieres predecir.

Cuotas de apuestas: (ej: 1.33 vs 3.00)
Es un predictor genial pero claro, no voy a tenerlo antes de los partidos en el simulador de Roland Garros.
Solo existen cuando el partido ya está anunciado y las casas las publican
Conversor de cuota a porcentaje: 1/cuota. 1/1.33 = 75%
Creo que lo quitaré del entrenamiento y de alguna forma puedo usarlo como medida de cuanto de bueno es mi predictor vs casas de apuestas

Valoraré si incluir estadísticas adicionales. Por el momento buscar cómo inferir datos adicionales de los datos que hay.
Estado de forma:
% partidos ganados en los últimos 2-3 meses?
% sets ganados en el último 2-3 meses?

Histórico:
Head to head histórico
Win rate de cada jugadora en esa superficie concreta
Win rate en esa ronda concreta (hay jugadoras que rinden mejor en finales)
Ratio Puntos Totales Ganados/Perdidos 

Tendría que buscar:
% Primer Servicio
% Puntos con 2do Servicio
% Juegos Ganados al Saque: Solidez del servicio. 
% Puntos de Break Points) Convertir oportunidades de quiebre y salvar las propias es crucial para ganar partidos ajustados.
Errores No Forzados vs. Golpes Ganadores:  

Simulador de partido. Solo actual? Entiendo que es mejor. No tiene sentido simular partidos pasados

Simulador de torneo. Cosas a tener en cuenta:
- Sacar ranking

# Metricas


```python
#Accuracy y AUC-ROC
# es más correcto conceptualmente usar AUC-ROC para optimizar los modelos
#  porque el objetivo del modelo es generar probabilidades para la simulación,
#  no clasificar con un umbral fijo

from sklearn.metrics import roc_auc_score

# AUC de tu modelo
y_proba_modelo = xgb_best.predict_proba(X_test)[:, 1]
auc_modelo = roc_auc_score(y_test, y_proba_modelo)

# AUC de las casas — usas sus probabilidades directamente
prob_casas = df.loc[df_test['match_id'], 'odd_1'].values
mask = ~np.isnan(prob_casas)

auc_casas = roc_auc_score(y_test[mask], prob_casas[mask])

print(f'AUC modelo: {auc_modelo:.4f}')
print(f'AUC casas:  {auc_casas:.4f}')
```

# Otras features

- Variación del ranking (si viene en tendencia positiva o negativa. coger otro margen de tiempo para que capte algo diferente del winrate2meses)
- Inactividad: Días de descanso desde último partido (limitar a maximo - un año igual)
- Capar más la diferencia de ranking? rank_diff_clipped = min(max(rank_diff, -50), 50)
- Ponderación por años (dar menos peso a los más antiguos)
- ELO! 

El sistema de coeficiente Elo en el tenis es un método estadístico diseñado para medir la habilidad relativa de los jugadores, ajustando su puntuación tras cada partido en función del resultado y la fuerza del oponente. A diferencia del ranking oficial ATP/WTA, que se basa en la acumulación de puntos por rondas alcanzadas en 52 semanas, el Elo es dinámico y evalúa la calidad de cada victoria individual  
La idea es simple: cada jugadora tiene una puntuación numérica que sube cuando gana y baja cuando pierde, pero la cantidad que sube o baja depende de la dificultad del rival:

Se puede mejorar el modelo incluyendo  perfiles de la jugadora? 

Resumen del pipeline completo  

- Calcular medias por jugadora (seleccionando bien qué stats usar)
- Generar dataset jugadora → medias
- Escalar con StandardScaler
- KMeans — prueba con k=3 o k=4, usa el método del codo para decidir
- Merge al dataset principal por nombre de jugadora
- Añadir cluster como feature categórica para p1 y p2
# WTA Match Predictor — EDA y Feature Engineering
Pipeline completo: carga, limpieza, clasificación de torneos, cálculo de features derivadas y generación del dataset de entrenamiento.

## 1. Imports y carga de datos


```python
import pandas as pd
import numpy as np
import os
from ydata_profiling import ProfileReport
```

    c:\Users\NaiaJon\AppData\Local\Programs\Python\Python311\Lib\site-packages\tqdm\auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm
    C:\Users\NaiaJon\AppData\Local\Temp\ipykernel_22896\1618932038.py:4: DeprecationWarning: 
        `import ydata_profiling` is deprecated and will not receive more updates. 
        Please install fg-data-profiling via `pip install fg-data-profiling` and use `import data_profiling` instead.
        
      from ydata_profiling import ProfileReport
    


```python
#!pip install kagglehub
```

    Collecting kagglehub
      Downloading kagglehub-1.0.1-py3-none-any.whl.metadata (40 kB)
         ---------------------------------------- 0.0/40.1 kB ? eta -:--:--
         ------------------- ------------------ 20.5/40.1 kB 217.9 kB/s eta 0:00:01
         -------------------------------------- 40.1/40.1 kB 317.6 kB/s eta 0:00:00
    Collecting kagglesdk<1.0,>=0.1.22 (from kagglehub)
      Downloading kagglesdk-0.1.23-py3-none-any.whl.metadata (13 kB)
    Requirement already satisfied: packaging in c:\users\naiajon\appdata\roaming\python\python311\site-packages (from kagglehub) (26.0)
    Requirement already satisfied: pyyaml in c:\users\naiajon\appdata\local\programs\python\python311\lib\site-packages (from kagglehub) (6.0.3)
    Requirement already satisfied: requests in c:\users\naiajon\appdata\local\programs\python\python311\lib\site-packages (from kagglehub) (2.32.5)
    Requirement already satisfied: tqdm in c:\users\naiajon\appdata\local\programs\python\python311\lib\site-packages (from kagglehub) (4.67.3)
    Requirement already satisfied: protobuf in c:\users\naiajon\appdata\local\programs\python\python311\lib\site-packages (from kagglesdk<1.0,>=0.1.22->kagglehub) (7.34.1)
    Requirement already satisfied: charset_normalizer<4,>=2 in c:\users\naiajon\appdata\local\programs\python\python311\lib\site-packages (from requests->kagglehub) (3.4.6)
    Requirement already satisfied: idna<4,>=2.5 in c:\users\naiajon\appdata\local\programs\python\python311\lib\site-packages (from requests->kagglehub) (3.11)
    Requirement already satisfied: urllib3<3,>=1.21.1 in c:\users\naiajon\appdata\local\programs\python\python311\lib\site-packages (from requests->kagglehub) (2.6.3)
    Requirement already satisfied: certifi>=2017.4.17 in c:\users\naiajon\appdata\local\programs\python\python311\lib\site-packages (from requests->kagglehub) (2026.2.25)
    Requirement already satisfied: colorama in c:\users\naiajon\appdata\roaming\python\python311\site-packages (from tqdm->kagglehub) (0.4.6)
    Downloading kagglehub-1.0.1-py3-none-any.whl (70 kB)
       ---------------------------------------- 0.0/70.6 kB ? eta -:--:--
       ---------------------------------------- 70.6/70.6 kB 1.9 MB/s eta 0:00:00
    Downloading kagglesdk-0.1.23-py3-none-any.whl (217 kB)
       ---------------------------------------- 0.0/217.8 kB ? eta -:--:--
       ---------------------------------------  215.0/217.8 kB 4.4 MB/s eta 0:00:01
       ---------------------------------------- 217.8/217.8 kB 3.3 MB/s eta 0:00:00
    Installing collected packages: kagglesdk, kagglehub
    Successfully installed kagglehub-1.0.1 kagglesdk-0.1.23
    

    
    [notice] A new release of pip is available: 24.0 -> 26.1
    [notice] To update, run: python.exe -m pip install --upgrade pip
    


```python
# Descarga de Kaggle actualizado
import kagglehub

# Download latest version
path = kagglehub.dataset_download("dissfya/wta-tennis-2007-2023-daily-update")

print("Path to dataset files:", path)
```

    Downloading to C:\Users\NaiaJon\.cache\kagglehub\datasets\dissfya\wta-tennis-2007-2023-daily-update\1062.archive...
    

    100%|██████████| 1.03M/1.03M [00:00<00:00, 2.34MB/s]

    Extracting files...
    Path to dataset files: C:\Users\NaiaJon\.cache\kagglehub\datasets\dissfya\wta-tennis-2007-2023-daily-update\versions\1062
    

    
    


```python
# Cuidado con las rutas al copiarlo al gihub
file_path = os.path.join(path, "wta.csv")
wta = pd.read_csv(file_path, low_memory=False)
print(f'Partidos cargados: {len(wta)}')
wta.tail()
```

    Partidos cargados: 44446
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tournament</th>
      <th>Date</th>
      <th>Court</th>
      <th>Surface</th>
      <th>Round</th>
      <th>Best of</th>
      <th>Player_1</th>
      <th>Player_2</th>
      <th>Winner</th>
      <th>Rank_1</th>
      <th>Rank_2</th>
      <th>Pts_1</th>
      <th>Pts_2</th>
      <th>Odd_1</th>
      <th>Odd_2</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>44441</th>
      <td>Mutua Madrid Open</td>
      <td>2026-04-29 00:00:00</td>
      <td>Outdoor</td>
      <td>Clay</td>
      <td>Quarterfinals</td>
      <td>3</td>
      <td>Pliskova Ka.</td>
      <td>Potapova A.</td>
      <td>Potapova A.</td>
      <td>197</td>
      <td>56</td>
      <td>375</td>
      <td>1065</td>
      <td>2.5</td>
      <td>1.53</td>
      <td>1-6 7-6 3-6</td>
    </tr>
    <tr>
      <th>44442</th>
      <td>Mutua Madrid Open</td>
      <td>2026-04-29 00:00:00</td>
      <td>Outdoor</td>
      <td>Clay</td>
      <td>Quarterfinals</td>
      <td>3</td>
      <td>Kostyuk M.</td>
      <td>Noskova L.</td>
      <td>Kostyuk M.</td>
      <td>23</td>
      <td>13</td>
      <td>1722</td>
      <td>2849</td>
      <td>1.5</td>
      <td>2.63</td>
      <td>7-6 6-0</td>
    </tr>
    <tr>
      <th>44443</th>
      <td>Mutua Madrid Open</td>
      <td>2026-04-30 00:00:00</td>
      <td>Outdoor</td>
      <td>Clay</td>
      <td>Semifinals</td>
      <td>3</td>
      <td>Baptiste H.</td>
      <td>Andreeva M.</td>
      <td>Andreeva M.</td>
      <td>32</td>
      <td>8</td>
      <td>1392</td>
      <td>3746</td>
      <td>2.63</td>
      <td>1.5</td>
      <td>4-6 6-7</td>
    </tr>
    <tr>
      <th>44444</th>
      <td>Mutua Madrid Open</td>
      <td>2026-04-30 00:00:00</td>
      <td>Outdoor</td>
      <td>Clay</td>
      <td>Semifinals</td>
      <td>3</td>
      <td>Kostyuk M.</td>
      <td>Potapova A.</td>
      <td>Kostyuk M.</td>
      <td>23</td>
      <td>56</td>
      <td>1722</td>
      <td>1065</td>
      <td>1.33</td>
      <td>3.4</td>
      <td>6-2 1-6 6-1</td>
    </tr>
    <tr>
      <th>44445</th>
      <td>Mutua Madrid Open</td>
      <td>2026-05-02 00:00:00</td>
      <td>Outdoor</td>
      <td>Clay</td>
      <td>The Final</td>
      <td>3</td>
      <td>Andreeva M.</td>
      <td>Kostyuk M.</td>
      <td>Kostyuk M.</td>
      <td>8</td>
      <td>23</td>
      <td>3746</td>
      <td>1722</td>
      <td>1.67</td>
      <td>2.2</td>
      <td>3-6 5-7</td>
    </tr>
  </tbody>
</table>
</div>



## 2. Exploración inicial


```python
wta.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 44446 entries, 0 to 44445
    Data columns (total 16 columns):
     #   Column      Non-Null Count  Dtype 
    ---  ------      --------------  ----- 
     0   Tournament  44446 non-null  object
     1   Date        44446 non-null  object
     2   Court       44446 non-null  object
     3   Surface     44446 non-null  object
     4   Round       44446 non-null  object
     5   Best of     44446 non-null  int64 
     6   Player_1    44446 non-null  object
     7   Player_2    44446 non-null  object
     8   Winner      44446 non-null  object
     9   Rank_1      44446 non-null  int64 
     10  Rank_2      44446 non-null  int64 
     11  Pts_1       44446 non-null  int64 
     12  Pts_2       44446 non-null  int64 
     13  Odd_1       44446 non-null  object
     14  Odd_2       44446 non-null  object
     15  Score       44446 non-null  object
    dtypes: int64(5), object(11)
    memory usage: 5.4+ MB
    


```python
# Profiling completo — genera wta_report.html
profile = ProfileReport(wta, title='WTA Profiling Report')
profile.to_file('wta_report.html')
```

## 3. Limpieza y transformaciones


```python
# Eliminar columnas no útiles para el modelo
# Court: muy desbalanceado (40k outdoor vs 400 indoor), la superficie ya lo captura
# Best of: en WTA siempre es al mejor de 3 sets
wta = wta.drop(columns=['Court', 'Best of'])
```


```python
# Unificar superficies: Greenset y Carpet tienen muy pocas entradas y son pistas rápidas → Hard
wta['Surface'] = wta['Surface'].replace({'Greenset': 'Hard', 'Carpet': 'Hard'})
print(wta['Surface'].value_counts())
```

    Surface
    Hard     27290
    Clay     12347
    Grass     4809
    Name: count, dtype: int64
    


```python
# Clasificar torneos por categoría usando palabras clave
# Los nombres cambian con patrocinadores pero la ciudad/torneo es estable

def clasificar_torneo(nombre):
    n = nombre.lower()
    
    if any(x in n for x in ['australian open', 'roland garros', 'french open',
                              'wimbledon', 'us open']):
        return 'GS'
    
    if any(x in n for x in ['wta finals', 'wta elite trophy', 'championships',
                              'tournament of champions', 'riyadh finals', 'wta tour championships']):
        return 'WTA_Finals'
    
    if any(x in n for x in ['indian wells', 'miami', 'madrid', 'roma', 'rome',
                              'internazionali', 'canada', 'toronto', 'rogers',
                              'canadian', 'cincinnati', 'beijing', 'china open',
                              'wuhan', 'doha', 'qatar', 'dubai', 'bnp paribas open',
                              'national bank open', 'sony ericsson open',
                              'western & southern financial group', 'shenzhen']):
        return 'WTA1000'
    
    if any(x in n for x in ['abu dhabi', 'abu dabi', 'adelaide', 'berlin', 'bett1open',
                              'eastbourne', 'rothesay', 'bad homburg', 'san jose',
                              'silicon valley', 'guangzhou', 'osaka', 'tokyo',
                              'toray pan pacific', 'seoul', 'strasbourg', 'birmingham',
                              'nottingham', 'brisbane', 'linz', 'merida', 'monterrey',
                              'charleston', 'stuttgart', 'porsche', 'washington',
                              'mubadala', 'guadalajara', 'gdl open akron',
                              'family circle cup', 'aegon', 'kremlin', 'pilot pen',
                              'new haven', 'bank of the west', 'sydney international',
                              'stanford', 'luxembourg', 'belgium', 'diamond games',
                              'fortis', 'san diego', 'zhengzhou', 'german']):
        return 'WTA500'
    
    return 'WTA250_o_menor'

wta['tournament_type'] = wta['Tournament'].map(clasificar_torneo)
print(wta['tournament_type'].value_counts())
```

    tournament_type
    WTA250_o_menor    14363
    WTA1000           10354
    GS                 9455
    WTA500             8343
    WTA_Finals         1931
    Name: count, dtype: int64
    


```python
# Comprobación: torneos sin clasificar
sin_clasificar = wta[wta['tournament_type'] == 'WTA250_o_menor']['Tournament'].nunique()
print(f'Torneos únicos sin clasificar: {sin_clasificar}')
```

    Torneos únicos sin clasificar: 145
    


```python
# Eliminar Tournament (ya tenemos tournament_type) y convertir tipos
wta = wta.drop(columns='Tournament')

wta['Date']  = pd.to_datetime(wta['Date'], errors='coerce')
wta['Odd_1'] = pd.to_numeric(wta['Odd_1'], errors='coerce')
wta['Odd_2'] = pd.to_numeric(wta['Odd_2'], errors='coerce')

print('Nulos tras conversión:')
print(wta[['Date', 'Odd_1', 'Odd_2']].isna().sum())
```

    Nulos tras conversión:
    Date     2
    Odd_1    1
    Odd_2    1
    dtype: int64
    


```python
# Eliminar partidos sin fecha 
wta = wta.dropna(subset=['Date'])
```


```python
# Limpiar odds: valores < 1 son códigos de sin datos (equivalente a -1)
# Los convertimos a NaN — se usarán para comparar con casas pero no para entrenar
wta.loc[wta['Odd_1'] < 1, 'Odd_1'] = None
wta.loc[wta['Odd_2'] < 1, 'Odd_2'] = None
```


```python
# Calcular probabilidades implícitas de las casas de apuestas
# Indican lo que se paga por euro la victoria (2.5, 3.5... etc). Lo paso a probabilidades
# 1/odd y normalizar para que sumen 1 (elimina el margen de la casa: suman un poco más de uno)

wta['prob_1'] = 1 / wta['Odd_1']
wta['prob_2'] = 1 / wta['Odd_2']
total = wta['prob_1'] + wta['prob_2']
wta['prob_1'] = wta['prob_1'] / total
wta['prob_2'] = wta['prob_2'] / total

print(wta['prob_1'].describe())
```

    count    44328.000000
    mean         0.498446
    std          0.214081
    min          0.019231
    25%          0.337349
    50%          0.500000
    75%          0.662651
    max          0.980769
    Name: prob_1, dtype: float64
    


```python
# Convertir tipos de columnas categóricas y de texto
wta['Surface']  = wta['Surface'].astype(str)
wta['Round']    = wta['Round'].astype(str)
wta['Score']    = wta['Score'].astype(str)
wta['Player_1'] = wta['Player_1'].astype(str)
wta['Player_2'] = wta['Player_2'].astype(str)
wta['Winner']   = wta['Winner'].astype(str)
```


```python
# Crear target: 1 si gana Player_1, 2 si gana Player_2
def asignar_target(fila):
    if fila['Winner'] == fila['Player_1']:
        return 1
    else:
        return 0

wta['target'] = wta.apply(asignar_target, axis=1)
print(wta['target'].value_counts())
```

    target
    1    22222
    0    22222
    Name: count, dtype: int64
    


```python
# Guardar wta limpio para uso en app.py y consultas. 
wta.to_csv('wta_limpio.csv', index=False)
print('wta_limpio.csv guardado')
```

    wta_limpio.csv guardado
    


```python
wta = pd.read_csv ('wta_limpio.csv', parse_dates= ['Date'])
```

## 4. Feature Engineering

Calculamos features derivadas para cada partido usando solo información disponible **antes** de ese partido (corte temporal para evitar data leakage).


```python
wta = calcular_elo_superficie(wta)
wta = calcular_elo_global(wta)
```


```python
# Incluye el ELO que necesitaremos en simulación
wta.to_csv('wta_limpio.csv', index=False)
print('wta_limpio.csv guardado')
```

    wta_limpio.csv guardado
    


```python
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
    return victorias/len(partidos)
```


```python
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
    return victorias/len(partidos)
```


```python
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
    return victorias_p1/len(partidos)
```


```python
def experiencia(df, jugadora, fecha_limite):
    """Número total de partidos jugados por la jugadora antes de la fecha"""
    mask = (
        ((df['Player_1'] == jugadora) | (df['Player_2'] == jugadora)) &
        (df['Date'] < fecha_limite)
    )
    return df[mask].shape[0]
```


```python
def inactividad (df, jugadora, fecha_limite):
    """ Días desde su último partido"""
    mask = (
        ((df['Player_1'] == jugadora) | (df['Player_2'] == jugadora)) &
        (df['Date'] < fecha_limite)
    )
    partidos = df[mask].sort_values('Date', ascending=False)
    if len(partidos) == 0:
        return 1000 # sin historial reciente
    ultimo = partidos.iloc[0]['Date']
    dias = (fecha_limite - ultimo).days
    return dias
```


```python
def tendencia_ranking(df, jugadora, fecha_limite, meses=6):
    """Diferencia de ranking entre hace 6 meses y ahora: """
    
    fecha_inicio = fecha_limite - pd.DateOffset(months=meses)
    
    mask = (
        ((df['Player_1'] == jugadora) | (df['Player_2'] == jugadora)) &
        (df['Date'] > fecha_inicio) & 
        (df['Date'] <= fecha_limite)  
    )
    
    partidos = df[mask].sort_values('Date', ascending=True)  
    
    if len(partidos) == 0:
        return 0  # o podrías return None para indicar sin datos
    
    primero = partidos.iloc[0]   
    ultimo = partidos.iloc[-1]   
    
    def get_ranking(partido, jugadora):
        if partido['Player_1'] == jugadora:
            return partido['Rank_1']
        else:
            return partido['Rank_2']
    
    ranking_antes = get_ranking(primero, jugadora)
    ranking_ahora = get_ranking(ultimo, jugadora)
    
    if pd.isna(ranking_antes) or pd.isna(ranking_ahora) or ranking_antes == 0 or ranking_ahora == 0:
        return None
    
    diferencia = ranking_antes - ranking_ahora
    return diferencia
```


```python
def calcular_elo_global(df, k=32, elo_inicial=1500):
    """
    Precalcula ELO global (independiente de superficie) para cada partido.
    El elo_global_p1, elo_global_p2, elo_global_diff al dataframe.
    
    IMPORTANTE: df debe estar ordenado por fecha ascendente antes de llamar esto.
    """
    elo_ratings = {}  # {jugadora: elo}

    def get_elo(jugadora):
        return elo_ratings.get(jugadora, elo_inicial)

    def expected(elo_a, elo_b):
        return 1 / (1 + 10 ** ((elo_b - elo_a) / 400))

    elo_p1_list, elo_p2_list = [], []

    for _, row in df.iterrows():
        p1        = row['Player_1']
        p2        = row['Player_2']
        resultado = row['target']

        elo_p1 = get_elo(p1)
        elo_p2 = get_elo(p2)

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
```


```python
def calcular_elo_superficie(df, k=32, elo_inicial=1500):
    """
    Precalcula ELO por superficie para cada partido.
    Añade columnas elo_p1, elo_p2, elo_diff al dataframe.
    
    IMPORTANTE: df debe estar ordenado por fecha ascendente antes de llamar esto.
    El ELO de cada partido = ELO ANTES de jugarlo (sin leakage).
    """
    elo_ratings = {}  # {jugadora: {superficie: elo}}

    def get_elo(jugadora, superficie):
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
        resultado  = row['target']  # 1 = gana p1, 0 = gana p2

        elo_p1 = get_elo(p1, superficie)
        elo_p2 = get_elo(p2, superficie)

        # Guardamos ANTES de actualizar → sin leakage
        elo_p1_list.append(elo_p1)
        elo_p2_list.append(elo_p2)

        # Actualización
        exp_p1 = expected(elo_p1, elo_p2)
        exp_p2 = 1 - exp_p1

        update_elo(p1, superficie, elo_p1 + k * (resultado - exp_p1))
        update_elo(p2, superficie, elo_p2 + k * ((1 - resultado) - exp_p2))

    df = df.copy()
    df['elo_p1']  = elo_p1_list
    df['elo_p2']  = elo_p2_list
    df['elo_diff'] = df['elo_p1'] - df['elo_p2']

    return df


```


```python
# En el flujo final lo quitaríamos porque iría en la función generar_features
#wta = calcular_elo_superficie(wta)
#wta = calcular_elo_global(wta)
```


```python
#historico_partidos = pd.read_csv ('historico_partidos.csv', parse_dates= ['date'])
```


```python
# Merge por match_id (que es el índice original de wta)
# historico_partidos = historico_partidos.merge(
#     wta[['elo_p1', 'elo_p2', 'elo_diff',
#          'elo_global_p1', 'elo_global_p2', 'elo_global_diff']].rename_axis('match_id').reset_index(),
#     on='match_id',
#     how='left'
# )




```


```python
historico_partidos.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>match_id</th>
      <th>date</th>
      <th>odd_1</th>
      <th>odd_2</th>
      <th>surface</th>
      <th>round</th>
      <th>tournament_type</th>
      <th>rank_diff</th>
      <th>wins2meses_p1</th>
      <th>wins2meses_p2</th>
      <th>...</th>
      <th>experiencia_p2</th>
      <th>target</th>
      <th>is_new_p1</th>
      <th>is_new_p2</th>
      <th>elo_p1</th>
      <th>elo_p2</th>
      <th>elo_diff</th>
      <th>elo_global_p1</th>
      <th>elo_global_p2</th>
      <th>elo_global_diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>42041</th>
      <td>44439</td>
      <td>2026-04-29</td>
      <td>0.379653</td>
      <td>0.620347</td>
      <td>Clay</td>
      <td>Quarterfinals</td>
      <td>WTA1000</td>
      <td>141.0</td>
      <td>0.857143</td>
      <td>0.800000</td>
      <td>...</td>
      <td>268</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1679.197340</td>
      <td>1745.745804</td>
      <td>-66.548464</td>
      <td>1726.747695</td>
      <td>1783.622488</td>
      <td>-56.874793</td>
    </tr>
    <tr>
      <th>42042</th>
      <td>44440</td>
      <td>2026-04-29</td>
      <td>0.636804</td>
      <td>0.363196</td>
      <td>Clay</td>
      <td>Quarterfinals</td>
      <td>WTA1000</td>
      <td>10.0</td>
      <td>0.833333</td>
      <td>0.727273</td>
      <td>...</td>
      <td>163</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1715.277499</td>
      <td>1595.556868</td>
      <td>119.720632</td>
      <td>1889.971589</td>
      <td>1818.420717</td>
      <td>71.550872</td>
    </tr>
    <tr>
      <th>42043</th>
      <td>44441</td>
      <td>2026-04-30</td>
      <td>0.363196</td>
      <td>0.636804</td>
      <td>Clay</td>
      <td>Semifinals</td>
      <td>WTA1000</td>
      <td>24.0</td>
      <td>0.714286</td>
      <td>0.823529</td>
      <td>...</td>
      <td>155</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1631.501843</td>
      <td>1836.954165</td>
      <td>-205.452323</td>
      <td>1768.212330</td>
      <td>1926.385894</td>
      <td>-158.173565</td>
    </tr>
    <tr>
      <th>42044</th>
      <td>44442</td>
      <td>2026-04-30</td>
      <td>0.718816</td>
      <td>0.281184</td>
      <td>Clay</td>
      <td>Semifinals</td>
      <td>WTA1000</td>
      <td>-33.0</td>
      <td>0.846154</td>
      <td>0.818182</td>
      <td>...</td>
      <td>269</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1725.972486</td>
      <td>1758.718071</td>
      <td>-32.745585</td>
      <td>1902.722355</td>
      <td>1797.026455</td>
      <td>105.695900</td>
    </tr>
    <tr>
      <th>42045</th>
      <td>44443</td>
      <td>2026-05-02</td>
      <td>0.568475</td>
      <td>0.431525</td>
      <td>Clay</td>
      <td>The Final</td>
      <td>WTA1000</td>
      <td>-15.0</td>
      <td>0.833333</td>
      <td>0.857143</td>
      <td>...</td>
      <td>243</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1844.460435</td>
      <td>1743.476026</td>
      <td>100.984408</td>
      <td>1935.566483</td>
      <td>1913.999679</td>
      <td>21.566804</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python
# Comprobación de las funciones antes de ejecutar el bucle completo
jugadora = 'Badosa P.'
fecha_limite = pd.to_datetime('2026-01-01')
print(f"Experiencia: {experiencia(wta, jugadora, fecha_limite)} partidos")
print(f"H2H vs {jugadora}: {headtohead(wta, jugadora, 'Sabalenka A.', fecha_limite):.2%}")
print(f"Win rate Clay: {winrate(wta, jugadora, fecha_limite, superficie='Clay'):.2%}")
print(f"Forma reciente: {forma_reciente(wta, jugadora, fecha_limite):.2%}")
# Revisar!
print(f'Tendencia ranking: {tendencia_ranking(wta, jugadora, fecha_limite):}')
print(f'Días inactiva: {inactividad (wta, jugadora, fecha_limite):}')

```

    Experiencia: 225 partidos
    H2H vs Badosa P.: 28.57%
    Win rate Clay: 70.67%
    Forma reciente: 0.00%
    Tendencia ranking: 0
    Días inactiva: 97
    

### Construcción del dataset de features

⚠️ Este bucle tarda ~30 minutos. El resultado se guarda en CSV para no tener que repetirlo.


```python
# Reset index para usar como match_id
wta = wta.reset_index()

features = []
for i, partido in wta.iterrows():
    p1    = partido['Player_1']
    p2    = partido['Player_2']
    fecha = partido['Date']
    
    row = {
        # Identificador y metadata
        'match_id': i,
        'date':     partido['Date'],
        'odd_1':    partido['prob_1'],
        'odd_2':    partido['prob_2'],

        # Features del partido
        'surface':         partido['Surface'],
        'round':           partido['Round'],
        'tournament_type': partido['tournament_type'],

        # Features calculadas (solo info anterior al partido)
        'rank_diff':           float(partido['Rank_1']) - float(partido['Rank_2']),
        'wins2meses_p1':       forma_reciente(wta, p1, fecha),
        'wins2meses_p2':       forma_reciente(wta, p2, fecha),
        'ratio_superficie_p1': winrate(wta, p1, fecha, superficie=partido['Surface']),
        'ratio_superficie_p2': winrate(wta, p2, fecha, superficie=partido['Surface']),
        'h2h':                 headtohead(wta, p1, p2, fecha),
        'ratio_ronda_p1':      winrate(wta, p1, fecha, ronda=partido['Round']),
        'ratio_ronda_p2':      winrate(wta, p2, fecha, ronda=partido['Round']),
        'experiencia_p1':      experiencia(wta, p1, fecha),
        'experiencia_p2':      experiencia(wta, p2, fecha),
        # Añadido 
        # 'inactividad_p1':      inactividad(wta, p1, fecha),
        # 'inactividad_p2':      inactividad(wta, p2, fecha),
        'elo_p1':          partido['elo_p1'],
        'elo_p2':          partido['elo_p2'],
        'elo_diff':        partido['elo_diff'],
        'elo_global_p1':   partido['elo_global_p1'],
        'elo_global_p2':   partido['elo_global_p2'],
        'elo_global_diff': partido['elo_global_diff'],

        # Target
        'target': partido['target']
    }
    features.append(row)

historico_partidos = pd.DataFrame(features)

# Feature de inexperiencia
historico_partidos['is_new_p1'] = (historico_partidos['experiencia_p1'] < 10).astype(int)
historico_partidos['is_new_p2'] = (historico_partidos['experiencia_p2'] < 10).astype(int)

print(f'Dataset construido: {len(historico_partidos)} partidos')
print(historico_partidos.info())
```

    Dataset construido: 44444 partidos
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 44444 entries, 0 to 44443
    Data columns (total 20 columns):
     #   Column               Non-Null Count  Dtype         
    ---  ------               --------------  -----         
     0   match_id             44444 non-null  int64         
     1   date                 44444 non-null  datetime64[ns]
     2   odd_1                44328 non-null  float64       
     3   odd_2                44328 non-null  float64       
     4   surface              44444 non-null  object        
     5   round                44444 non-null  object        
     6   tournament_type      44444 non-null  object        
     7   rank_diff            44444 non-null  float64       
     8   wins2meses_p1        44444 non-null  float64       
     9   wins2meses_p2        44444 non-null  float64       
     10  ratio_superficie_p1  44444 non-null  float64       
     11  ratio_superficie_p2  44444 non-null  float64       
     12  h2h                  44444 non-null  float64       
     13  ratio_ronda_p1       44444 non-null  float64       
     14  ratio_ronda_p2       44444 non-null  float64       
     15  experiencia_p1       44444 non-null  int64         
     16  experiencia_p2       44444 non-null  int64         
     17  target               44444 non-null  int64         
     18  is_new_p1            44444 non-null  int64         
     19  is_new_p2            44444 non-null  int64         
    dtypes: datetime64[ns](1), float64(10), int64(6), object(3)
    memory usage: 6.8+ MB
    None
    

## 5. Limpieza del dataset de features


```python
# Eliminar anteriores a 2007: primer año del dataset, las features derivadas están muy incompletas
# porque no hay historial previo para calcularlas
historico_partidos = historico_partidos[historico_partidos['date'].dt.year > 2007]
print(f'Partidos tras eliminar 2007: {len(historico_partidos)}')
```

    Partidos tras eliminar 2007: 42046
    


```python
# Verificar nulos
print(historico_partidos.isnull().sum())
```

    match_id                0
    date                    0
    odd_1                  88
    odd_2                  88
    surface                 0
    round                   0
    tournament_type         0
    rank_diff               0
    wins2meses_p1           0
    wins2meses_p2           0
    ratio_superficie_p1     0
    ratio_superficie_p2     0
    h2h                     0
    ratio_ronda_p1          0
    ratio_ronda_p2          0
    experiencia_p1          0
    experiencia_p2          0
    target                  0
    is_new_p1               0
    is_new_p2               0
    elo_p1                  0
    elo_p2                  0
    elo_diff                0
    elo_global_p1           0
    elo_global_p2           0
    elo_global_diff         0
    dtype: int64
    


```python
# Guardar — carga posterior con pd.read_csv('historico_partidos.csv', parse_dates=['date'])
historico_partidos.to_csv('historico_partidos.csv', index=False)
print('✓ historico_partidos.csv guardado')
```

    ✓ historico_partidos.csv guardado
    


```python
#hist = pd.read_csv ('historico_partidos.csv', parse_dates= ['date'])
```

# Clustering. Perfiles de jugadoras


```python
import os
print(os.listdir(r'C:\Users\NaiaJon\Documents\Naia\BootcampDataScience\Datos ML\stats'))
```

    ['wta_matches_2007.csv', 'wta_matches_2008.csv', 'wta_matches_2009.csv', 'wta_matches_2010.csv', 'wta_matches_2011.csv', 'wta_matches_2012.csv', 'wta_matches_2013.csv', 'wta_matches_2014.csv', 'wta_matches_2015.csv', 'wta_matches_2016.csv', 'wta_matches_2017.csv', 'wta_matches_2018.csv', 'wta_matches_2019.csv', 'wta_matches_2020.csv', 'wta_matches_2021.csv', 'wta_matches_2022.csv', 'wta_matches_2023.csv', 'wta_matches_2024.csv', 'wta_matches_2025.csv', 'wta_matches_2026.csv']
    


```python
stats_2024 = pd.read_csv(r'C:\Users\NaiaJon\Documents\Naia\BootcampDataScience\Datos ML\wta_matches_2024.csv', low_memory=False)
stats_2024.head()
```


```python
import glob
# Cargar todos los archivos desde 
archivos = glob.glob(r'C:\Users\NaiaJon\Documents\Naia\BootcampDataScience\Datos ML\stats\*.csv')  # ajusta la ruta

```


```python
dfs = []
for f in archivos:
    dfs.append(pd.read_csv(f)) 
df_stats = pd.concat(dfs, ignore_index=True)
print(df_stats.shape)
```

    (51935, 49)
    


```python
df_stats.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tourney_id</th>
      <th>tourney_name</th>
      <th>surface</th>
      <th>draw_size</th>
      <th>tourney_level</th>
      <th>tourney_date</th>
      <th>match_num</th>
      <th>winner_id</th>
      <th>winner_seed</th>
      <th>winner_entry</th>
      <th>...</th>
      <th>l_1stIn</th>
      <th>l_1stWon</th>
      <th>l_2ndWon</th>
      <th>l_SvGms</th>
      <th>l_bpSaved</th>
      <th>l_bpFaced</th>
      <th>winner_rank</th>
      <th>winner_rank_points</th>
      <th>loser_rank</th>
      <th>loser_rank_points</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>51930</th>
      <td>2026-W-FC-2026-QUA</td>
      <td>BJK Cup Qualifiers</td>
      <td>NaN</td>
      <td>36</td>
      <td>D</td>
      <td>20260410</td>
      <td>118</td>
      <td>259733</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>127.0</td>
      <td>600.0</td>
      <td>128.0</td>
      <td>597.0</td>
    </tr>
    <tr>
      <th>51931</th>
      <td>2026-W-FC-2026-QUA</td>
      <td>BJK Cup Qualifiers</td>
      <td>NaN</td>
      <td>36</td>
      <td>D</td>
      <td>20260410</td>
      <td>119</td>
      <td>220714</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>42.0</td>
      <td>1310.0</td>
      <td>84.0</td>
      <td>863.0</td>
    </tr>
    <tr>
      <th>51932</th>
      <td>2026-W-FC-2026-QUA</td>
      <td>BJK Cup Qualifiers</td>
      <td>NaN</td>
      <td>36</td>
      <td>D</td>
      <td>20260410</td>
      <td>120</td>
      <td>201709</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>74.0</td>
      <td>893.0</td>
      <td>198.0</td>
      <td>365.0</td>
    </tr>
    <tr>
      <th>51933</th>
      <td>2026-W-FC-2026-QUA</td>
      <td>BJK Cup Qualifiers</td>
      <td>NaN</td>
      <td>36</td>
      <td>D</td>
      <td>20260410</td>
      <td>121</td>
      <td>211279</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>173.0</td>
      <td>420.0</td>
      <td>80.0</td>
      <td>880.0</td>
    </tr>
    <tr>
      <th>51934</th>
      <td>2026-W-FC-2026-QUA</td>
      <td>BJK Cup Qualifiers</td>
      <td>NaN</td>
      <td>36</td>
      <td>D</td>
      <td>20260410</td>
      <td>122</td>
      <td>261963</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>275.0</td>
      <td>243.0</td>
      <td>56.0</td>
      <td>1050.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 49 columns</p>
</div>




```python
df_stats.columns
```




    Index(['tourney_id', 'tourney_name', 'surface', 'draw_size', 'tourney_level',
           'tourney_date', 'match_num', 'winner_id', 'winner_seed', 'winner_entry',
           'winner_name', 'winner_hand', 'winner_ht', 'winner_ioc', 'winner_age',
           'loser_id', 'loser_seed', 'loser_entry', 'loser_name', 'loser_hand',
           'loser_ht', 'loser_ioc', 'loser_age', 'score', 'best_of', 'round',
           'minutes', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
           'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'l_ace', 'l_df', 'l_svpt',
           'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced',
           'winner_rank', 'winner_rank_points', 'loser_rank', 'loser_rank_points'],
          dtype='object')




```python
df = df_stats [['winner_name', 'loser_name', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn', 'w_1stWon', 'w_2ndWon',
       'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'l_ace', 'l_df', 'l_svpt',
       'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms', 'l_bpSaved', 'l_bpFaced']]
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 51935 entries, 0 to 51934
    Data columns (total 18 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   winner_name  51935 non-null  object 
     1   loser_name   51935 non-null  object 
     2   w_ace        45005 non-null  float64
     3   w_df         44972 non-null  float64
     4   w_svpt       45006 non-null  float64
     5   w_1stIn      45006 non-null  float64
     6   w_1stWon     45006 non-null  float64
     7   w_2ndWon     45006 non-null  float64
     8   w_bpSaved    45002 non-null  float64
     9   w_bpFaced    45002 non-null  float64
     10  l_ace        45000 non-null  float64
     11  l_df         44972 non-null  float64
     12  l_svpt       45006 non-null  float64
     13  l_1stIn      45006 non-null  float64
     14  l_1stWon     45006 non-null  float64
     15  l_2ndWon     45006 non-null  float64
     16  l_bpSaved    45004 non-null  float64
     17  l_bpFaced    45004 non-null  float64
    dtypes: float64(16), object(2)
    memory usage: 7.1+ MB
    


```python
# l_SvGms	Number of service games won by the loser. (Integer)
#w_SvGms     Number of service games won by the winer. (Integer) 
# 27830 non-null  float64. Los voy a quitar
```


```python
df = df.drop (['l_SvGms','w_SvGms'], axis=1)
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 51935 entries, 0 to 51934
    Data columns (total 18 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   winner_name  51935 non-null  object 
     1   loser_name   51935 non-null  object 
     2   w_ace        45005 non-null  float64
     3   w_df         44972 non-null  float64
     4   w_svpt       45006 non-null  float64
     5   w_1stIn      45006 non-null  float64
     6   w_1stWon     45006 non-null  float64
     7   w_2ndWon     45006 non-null  float64
     8   w_bpSaved    45002 non-null  float64
     9   w_bpFaced    45002 non-null  float64
     10  l_ace        45000 non-null  float64
     11  l_df         44972 non-null  float64
     12  l_svpt       45006 non-null  float64
     13  l_1stIn      45006 non-null  float64
     14  l_1stWon     45006 non-null  float64
     15  l_2ndWon     45006 non-null  float64
     16  l_bpSaved    45004 non-null  float64
     17  l_bpFaced    45004 non-null  float64
    dtypes: float64(16), object(2)
    memory usage: 7.1+ MB
    


```python
df.columns
```




    Index(['winner_name', 'loser_name', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn',
           'w_1stWon', 'w_2ndWon', 'w_bpSaved', 'w_bpFaced', 'l_ace', 'l_df',
           'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_bpSaved', 'l_bpFaced'],
          dtype='object')




```python
# Medias de estadísticos de cada jugadora.

# La misma jugadora tiene partidos como winner y como loser
# Separar estadísticas para sumar luego

df_winner = df_stats[['winner_name','w_ace', 'w_df', 'w_svpt', 'w_1stIn',
       'w_1stWon', 'w_2ndWon', 'w_bpSaved', 'w_bpFaced']].rename(columns={
    'winner_name': 'jugadora', 'w_ace': 'ace', 'w_df': 'df', 'w_svpt': 'svpt', 'w_1stIn' : '1stIn',
       'w_1stWon': '1stWon', 'w_2ndWon':'2ndWon', 'w_bpSaved':'bpSaved' , 'w_bpFaced': 'bpFaced'
})

# 2. Extraer stats como perdedora
df_loser = df_stats[['loser_name', 'l_ace', 'l_df', 'l_svpt', 'l_1stIn', 
                     'l_1stWon', 'l_2ndWon', 'l_bpSaved', 'l_bpFaced']].rename(columns={
    'loser_name': 'jugadora', 'l_ace': 'ace', 'l_df': 'df', 'l_svpt': 'svpt', 'l_1stIn' : '1stIn',
       'l_1stWon': '1stWon', 'l_2ndWon':'2ndWon', 'l_bpSaved':'bpSaved' , 'l_bpFaced': 'bpFaced'
})
```


```python
from difflib import get_close_matches
def nombre_stats_a_wta(nombre_completo, candidatos_wta):
    """'Caroline Garcia' → 'Garcia C.'"""
    if pd.isna(nombre_completo):
        return None
    partes = nombre_completo.strip().split(' ')
    inicial = partes[0][0]
    apellido = ' '.join(partes[1:])
    nombre_convertido = f"{apellido} {inicial}."
    
    # Primero buscar exacto
    if nombre_convertido in candidatos_wta:
        return nombre_convertido
    
    # Si no, buscar el más parecido
    matches = get_close_matches(nombre_convertido, candidatos_wta, n=1, cutoff=0.6)
    return matches[0] if matches else None

candidatos_wta = list(set(wta['Player_1'].unique()) | set(wta['Player_2'].unique()))

```


```python
# Agrupamos todos los partidos
df_all = pd.concat([df_winner, df_loser], ignore_index=True)

# Voy a incluir también % de break points 
df_all ['bpsaved_per'] = df_all ['bpSaved'] / df_all ['bpFaced']

# Normalizo nombres al formato de mi otro dataset (Aprox porque... vaya cristo)
df_all['jugadora_wta'] = df_all['jugadora'].apply(lambda x: nombre_stats_a_wta(x, candidatos_wta))

# Me hago un df con las medias de los estadísticos para cada jugadora 
df_cluster = df_all.drop(columns=['jugadora_wta']).groupby('jugadora').mean()

# Vuelvo a añadir nombre_wta mapeado
mapping = df_all[['jugadora', 'jugadora_wta']].drop_duplicates().set_index('jugadora')
df_cluster['nombre_wta'] = mapping['jugadora_wta']

```


```python
df_cluster.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 1999 entries, Abigail Guthrie to Zuzana Ondraskova
    Data columns (total 10 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   ace          1519 non-null   float64
     1   df           1519 non-null   float64
     2   svpt         1519 non-null   float64
     3   1stIn        1519 non-null   float64
     4   1stWon       1519 non-null   float64
     5   2ndWon       1519 non-null   float64
     6   bpSaved      1519 non-null   float64
     7   bpFaced      1519 non-null   float64
     8   bpsaved_per  1515 non-null   float64
     9   nombre_wta   1852 non-null   object 
    dtypes: float64(9), object(1)
    memory usage: 171.8+ KB
    


```python
# Son exactamente los mismos nulos en todas las columnas. Parece que hay jugadoras sin datos
cols_numericas = ['ace', 'df', 'svpt', '1stIn', '1stWon', '2ndWon', 'bpSaved', 'bpFaced', 'bpsaved_per']

parciales = df_cluster[df_cluster[cols_numericas].isna().any(axis=1) & ~df_cluster[cols_numericas].isna().all(axis=1)]
print(len(parciales))
```

    4
    


```python
## elimina las que tienen todas las columnas numéricas nulas 
df_kmeans = df_cluster.dropna(subset=cols_numericas, how='all')  


```


```python
df_kmeans.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 1519 entries, Abigail Spears to Zuzana Ondraskova
    Data columns (total 10 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   ace          1519 non-null   float64
     1   df           1519 non-null   float64
     2   svpt         1519 non-null   float64
     3   1stIn        1519 non-null   float64
     4   1stWon       1519 non-null   float64
     5   2ndWon       1519 non-null   float64
     6   bpSaved      1519 non-null   float64
     7   bpFaced      1519 non-null   float64
     8   bpsaved_per  1515 non-null   float64
     9   nombre_wta   1446 non-null   object 
    dtypes: float64(9), object(1)
    memory usage: 130.5+ KB
    


```python
df_kmeans[df_kmeans['bpsaved_per'].isna() & df_kmeans[['ace', 'df', 'svpt', '1stIn', '1stWon', '2ndWon', 
                                                          'bpSaved', 'bpFaced']].notna().all(axis=1)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ace</th>
      <th>df</th>
      <th>svpt</th>
      <th>1stIn</th>
      <th>1stWon</th>
      <th>2ndWon</th>
      <th>bpSaved</th>
      <th>bpFaced</th>
      <th>bpsaved_per</th>
      <th>nombre_wta</th>
    </tr>
    <tr>
      <th>jugadora</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ayana Rengiil</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>35.0</td>
      <td>26.0</td>
      <td>22.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>Rezai A.</td>
    </tr>
    <tr>
      <th>Matilde Jorge</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>12.0</td>
      <td>8.0</td>
      <td>6.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>Georges M.</td>
    </tr>
    <tr>
      <th>Savini Jayasuriya</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>30.0</td>
      <td>12.0</td>
      <td>8.0</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>None</td>
    </tr>
    <tr>
      <th>Tamachan Momkoonthod</th>
      <td>5.0</td>
      <td>1.0</td>
      <td>26.0</td>
      <td>17.0</td>
      <td>17.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Ningun Break point en contra. Voy a imputar 1. Es lo más parecido a haberlos salvado todos
df_kmeans['bpsaved_per'] = df_kmeans['bpsaved_per'].fillna(1.0)
```

    C:\Users\NaiaJon\AppData\Local\Temp\ipykernel_22896\3587167442.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_kmeans['bpsaved_per'] = df_kmeans['bpsaved_per'].fillna(1.0)
    


```python

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

scaler = StandardScaler()

X = df_kmeans[cols_numericas]
X_scaled = scaler.fit_transform(X)

# Aplicamos el método del codo para ver en cuantos clusters tiene sentido dividir. Por lógica deberían
# ser 3 o como mucho 4
inertias = []
k_range = range(2, 10)

for k in k_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.plot(k_range, inertias, marker='o')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Inercia')
plt.title('Método del codo')
plt.show()
```


    
![png](01_EDA_FeatureEngineering_files/01_EDA_FeatureEngineering_67_0.png)
    



```python
# No se ve claramente el punto de inflexion. Probamos con el silhouette score a ver...
```


```python
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

silhouette_scores = []
k_range = range(2, 10)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

plt.figure(figsize=(8, 5))
plt.plot(k_range, silhouette_scores, marker='o')
plt.title('Coeficiente de silueta')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Silhouette score')
plt.grid(True)
plt.show()
```


    
![png](01_EDA_FeatureEngineering_files/01_EDA_FeatureEngineering_69_0.png)
    



```python
# Parece que el máximo está en K=2. No parece que con estos datos haya una clusterización fiable
```


```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

kmeans_2 = KMeans(n_clusters=2, random_state=42)
labels = kmeans_2.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
for cluster in [0, 1]:
    mask = labels == cluster
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Cluster {cluster}', alpha=0.7)

plt.title('Clusters de saque (PCA)')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.legend()
plt.grid(True)
plt.show()

print(f"Varianza explicada total: {sum(pca.explained_variance_ratio_)*100:.1f}%")
```


    
![png](01_EDA_FeatureEngineering_files/01_EDA_FeatureEngineering_71_0.png)
    


    Varianza explicada total: 66.2%
    

 PC1 (el eje horizontal, 46.8%) es el que realmente separa los dos clusters: el Cluster 1 (naranja) se agrupa a la izquierda y el Cluster 0 (azul) a la derecha, lo que indica que PC1 está capturando la diferencia principal entre los dos perfiles de saque. PC2 (vertical, 19.4%) apenas discrimina entre grupos, ambos se mezclan verticalmente. Hay una zona central con mucho solapamiento, lo cual es consistente con los scores de silueta moderados que obteníamos. Los clusters existen, pero no son grupos nítidos y bien separados, sino más bien dos extremos de un continuo. No aporta nada como feature


```python
# Explicación:
# w_ace	Number of aces by the winner (Integer)
# w_df	Number of double faults by the winner. (Integer)
# l_1stIn	Number of first serves in by the loser. (Integer)
# l_1stWon	Number of first serves won by the loser. (Integer)
# l_2ndWon	Number of second serves won by the loser. (Integer)
# l_SvGms	Number of service games won by the loser. (Integer)
# l_bpSaved	Number of break points saved by the loser. (Integer)
# l_bpFaced	Number of break points faced by the loser. (Integer)
```


```python

```


```python
stats_2024.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tourney_id</th>
      <th>tourney_name</th>
      <th>surface</th>
      <th>draw_size</th>
      <th>tourney_level</th>
      <th>tourney_date</th>
      <th>match_num</th>
      <th>winner_id</th>
      <th>winner_seed</th>
      <th>winner_entry</th>
      <th>...</th>
      <th>l_1stIn</th>
      <th>l_1stWon</th>
      <th>l_2ndWon</th>
      <th>l_SvGms</th>
      <th>l_bpSaved</th>
      <th>l_bpFaced</th>
      <th>winner_rank</th>
      <th>winner_rank_points</th>
      <th>loser_rank</th>
      <th>loser_rank_points</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-9900</td>
      <td>United Cup</td>
      <td>Hard</td>
      <td>18</td>
      <td>I</td>
      <td>20240101</td>
      <td>299</td>
      <td>216347</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>27.0</td>
      <td>13.0</td>
      <td>7.0</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>8.0</td>
      <td>1.0</td>
      <td>9505.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-9900</td>
      <td>United Cup</td>
      <td>Hard</td>
      <td>18</td>
      <td>I</td>
      <td>20240101</td>
      <td>297</td>
      <td>216347</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>47.0</td>
      <td>28.0</td>
      <td>7.0</td>
      <td>12.0</td>
      <td>3.0</td>
      <td>8.0</td>
      <td>1.0</td>
      <td>9505.0</td>
      <td>20.0</td>
      <td>2330.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-9900</td>
      <td>United Cup</td>
      <td>Hard</td>
      <td>18</td>
      <td>I</td>
      <td>20240101</td>
      <td>295</td>
      <td>201493</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>82.0</td>
      <td>47.0</td>
      <td>10.0</td>
      <td>15.0</td>
      <td>11.0</td>
      <td>19.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>292.0</td>
      <td>246.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-9900</td>
      <td>United Cup</td>
      <td>Hard</td>
      <td>18</td>
      <td>I</td>
      <td>20240101</td>
      <td>293</td>
      <td>216347</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>27.0</td>
      <td>18.0</td>
      <td>6.0</td>
      <td>8.0</td>
      <td>6.0</td>
      <td>11.0</td>
      <td>1.0</td>
      <td>9505.0</td>
      <td>14.0</td>
      <td>2770.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-9900</td>
      <td>United Cup</td>
      <td>Hard</td>
      <td>18</td>
      <td>I</td>
      <td>20240101</td>
      <td>291</td>
      <td>201614</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>72.0</td>
      <td>51.0</td>
      <td>22.0</td>
      <td>16.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>20.0</td>
      <td>2330.0</td>
      <td>544.0</td>
      <td>89.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 49 columns</p>
</div>



# Analisis de nuevas variables


```python
import seaborn as sns

sns.pairplot (data=historico_partidos)
```




    <seaborn.axisgrid.PairGrid at 0x23029f4ca90>




    
![png](01_EDA_FeatureEngineering_files/01_EDA_FeatureEngineering_77_1.png)
    



```python
import matplotlib.pyplot as plt
plt.figure (figsize= (16,16))
sns.heatmap(
    historico_partidos.corr(numeric_only=True),
    annot=True,
    fmt='.2f',
    cmap='coolwarm'
)
```




    <Axes: >




    
![png](01_EDA_FeatureEngineering_files/01_EDA_FeatureEngineering_78_1.png)
    


# Features adicionales
Inactividad   
ELO
# Simulacion Roland Garros


```python
import random
import pickle
import pandas as pd
```


```python
from features import (forma_reciente, winrate, headtohead, experiencia, 
                      get_ranking, get_elo)
```


```python
# WTA limpio como fuente de consulta para generar los features del partido a simular
wta=pd.read_csv('wta_limpio.csv', parse_dates=['Date'])

```


```python
# Cargar el modelo de ML
with open ('gbx_v3.model', 'rb') as archivo_entrada:
    modeloML = pickle.load(archivo_entrada)
print(modeloML)
```

    Pipeline(steps=[('preprocessor',
                     ColumnTransformer(transformers=[('num', StandardScaler(),
                                                      ['rank_diff', 'wins2meses_p1',
                                                       'wins2meses_p2',
                                                       'ratio_superficie_p1',
                                                       'ratio_superficie_p2', 'h2h',
                                                       'ratio_ronda_p1',
                                                       'ratio_ronda_p2',
                                                       'experiencia_p1',
                                                       'experiencia_p2',
                                                       'is_new_p1', 'is_new_p2',
                                                       'elo_p1', 'elo_p2',
                                                       'elo_diff', 'elo_global_p1',
                                                       'elo_global_p2',
                                                       'elo_global_di...
                                   feature_types=None, feature_weights=None,
                                   gamma=None, grow_policy=None,
                                   importance_type=None,
                                   interaction_constraints=None, learning_rate=0.05,
                                   max_bin=None, max_cat_threshold=None,
                                   max_cat_to_onehot=None, max_delta_step=None,
                                   max_depth=3, max_leaves=None,
                                   min_child_weight=None, missing=nan,
                                   monotone_constraints=None, multi_strategy=None,
                                   n_estimators=200, n_jobs=None,
                                   num_parallel_tree=None, ...))])
    

┌─────────────────────────────────────────┐
│           SIMULACIÓN MONTE CARLO        │  ← repite 10.000 veces
│                                         │
│   Cuartos → Semis → Final               │
│   para cada partido llama a...          │
│                                         │
│   ┌─────────────────────────────────┐   │
│   │         MODELO ML               │   │  ← entrenado con datos históricos
│   │  input: jugadora A vs B         │   │
│   │  output: probabilidad 0..1      │   │
│   └─────────────────────────────────┘   │
│                                         │
└─────────────────────────────────────────┘
         ↓ resultado final
   {"Swiatek": 38%, "Sabalenka": 22%, ...}


```python
# cargar el cuadro generado del archivo de texto
# el archivo contiene código Python válido que crea la variable cuadro
with open("cuadro_python_list.txt", "r", encoding="utf-8") as f:
    exec(f.read())

print(cuadro[:10])
```

    ['Sabalenka A.', 'Siniakova K.', 'Masarova R.', 'Ruzic A.', 'Krueger A.', 'Osorio C.', 'Yastremska D.', 'Bucsa C.', 'Wang X.', 'Townsend T.']
    


```python
# La simulación tarda muchísimo si tiene que calcular cada vez (10.000) todas las features. 
# Como las jugadoras ya las sabemos, calculamos las features antes, las almacenamos y luego llamamos ahí 
# en vez de a las funciones
```


```python
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
```


```python
cache_features, cache_h2h = calculo_features_jugadoras_torneo (wta, cuadro)
```


```python
# X = construir_features_desde_cache(cache_features, cache_h2h, cuadro[0], cuadro[1], 'Clay', '1st Round', pd.to_datetime('2026-05-05'))
# print(X.columns.tolist())           # columnas que genera la simulación
# print(modeloML.feature_names_in_)  # columnas que espera el modelo
```

    ['surface', 'round', 'rank_diff', 'wins2meses_p1', 'wins2meses_p2', 'ratio_superficie_p1', 'ratio_superficie_p2', 'h2h', 'ratio_ronda_p1', 'ratio_ronda_p2', 'experiencia_p1', 'experiencia_p2', 'tournament_type', 'elo_p1', 'elo_p2', 'elo_diff', 'elo_global_p1', 'elo_global_p2', 'elo_global_diff', 'is_new_p1', 'is_new_p2']
    ['surface' 'round' 'tournament_type' 'rank_diff' 'wins2meses_p1'
     'wins2meses_p2' 'ratio_superficie_p1' 'ratio_superficie_p2' 'h2h'
     'ratio_ronda_p1' 'ratio_ronda_p2' 'experiencia_p1' 'experiencia_p2'
     'is_new_p1' 'is_new_p2' 'elo_p1' 'elo_p2' 'elo_diff' 'elo_global_p1'
     'elo_global_p2' 'elo_global_diff']
    


```python
# FUNCIONES PARA LA SIMULACIÓN DEL TORNEO

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
```


```python

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

# # Cuadro ficticio prueba
# def crear_cuadro_top16():
#     return [
#         "Swiatek I.", "Sabalenka A.", "Gauff C.", "Rybakina E.",
#         "Pegula J.", "Vondrousova", "Jabeur O.", "Zheng",
#         "Sakkari M.", "J. Ostapenko", "Badosa P.", "D. Kasatkina",
#         "Keys M.", "Azarenka V.", "Svitolina E.", "Navarro E."
#     ]


```


```python
# Cuadro ficticio prueba
def crear_cuadro_top16():
    return [
        "Swiatek I.", "Sabalenka A.", "Gauff C.", "Rybakina E.",
        "Pegula J.", "Vondrousova", "Jabeur O.", "Zheng",
        "Sakkari M.", "J. Ostapenko", "Badosa P.", "D. Kasatkina",
        "Keys M.", "Azarenka V.", "Svitolina E.", "Navarro E."
    ]
```


```python
cuadro16 = crear_cuadro_top16()
```


```python
import time
inicio = time.time()
probabilidades, victorias = simulacion_montecarlo(cache_features, cache_h2h, cuadro, modeloML, 100)
fin = time.time()
tiempo_100 = fin - inicio

# Mostrar resultados
print("Resultados después de 10,000 simulaciones:")
print("-" * 40)
for jugadora, prob in sorted(probabilidades.items(), key=lambda x: x[1], reverse=True):
    print(f"{jugadora}: {prob:.2%} ({victorias[jugadora]} títulos)")
print(f"Tiempo de simulación: {tiempo_100:.2f} segundos")
```

    Resultados después de 10,000 simulaciones:
    ----------------------------------------
    Swiatek I.: 24.00% (24 títulos)
    Sabalenka A.: 20.00% (20 títulos)
    Gauff C.: 11.00% (11 títulos)
    Rybakina E.: 11.00% (11 títulos)
    Muchova K.: 7.00% (7 títulos)
    Andreeva M.: 6.00% (6 títulos)
    Anisimova A.: 6.00% (6 títulos)
    Pegula J.: 4.00% (4 títulos)
    Svitolina E.: 4.00% (4 títulos)
    Shnaider D.: 1.00% (1 títulos)
    Vondrousova M.: 1.00% (1 títulos)
    Li A.: 1.00% (1 títulos)
    Mboko V.: 1.00% (1 títulos)
    Bencic B.: 1.00% (1 títulos)
    Salkova D.: 1.00% (1 títulos)
    Kudermetova V.: 1.00% (1 títulos)
    Tiempo de simulación: 46.33 segundos
    


```python
jugadora_test = cuadro[75]
print(f"Jugadora: {jugadora_test}")
X_test = construir_features_desde_cache(cache_features, cache_h2h, 'Swiatek I.', jugadora_test, 'Clay', '1st Round')
print(modeloML.predict_proba(X_test))
```

    Jugadora: Valentova T.
    [[0.0856216 0.9143784]]
    


```python
jugadora_test = cuadro[48]
print(f"Jugadora: {jugadora_test}")
X_test = construir_features_desde_cache(cache_features, cache_h2h, 'Swiatek I.', jugadora_test, 'Clay', '1st Round')
print(modeloML.predict_proba(X_test))
```

    Jugadora: Jovic I.
    [[0.1656447 0.8343553]]
    


```python
# ¿La caché tiene datos diferentes para cada jugadora?
print(cache_features['I. Swiatek']['elo_global'])
print(cache_features['T. Valentova']['elo_global'])
print(cache_features['I. Jovic']['elo_global'])
```

    1500
    1500
    1500
    


```python
# ¿Cómo aparece Swiatek en wta?
wta[wta['Player_1'].str.contains('Swiatek')]['Player_1'].unique()
```




    array(['Swiatek I.'], dtype=object)




```python
import numpy as np
# Input completamente aleatorio
X_random = pd.DataFrame([{
    'surface': 'Clay', 'round': '1st Round', 'tournament_type': 'GS',
    'rank_diff': 999, 'wins2meses_p1': 0, 'wins2meses_p2': 0,
    'ratio_superficie_p1': 0, 'ratio_superficie_p2': 0,
    'h2h': 0, 'ratio_ronda_p1': 0, 'ratio_ronda_p2': 0,
    'experiencia_p1': 0, 'experiencia_p2': 0
}])
print(modeloML.predict_proba(X_random))
```

    [[0.680313 0.319687]]
    
