# WTA Match Predictor

## Memoria del Proyecto de Machine Learning

---

## I. Introducción

### Contexto del problema y justificación

El tenis profesional femenino dispone de una buena cantidad de datos históricos: resultados, rankings, superficies, cuotas de apuestas, estadísticas de juego. Sin embargo, la predicción de resultados en tenis es un problema genuinamente difícil. Intervienen múltiples factores que condicionan el resultado de un partido: ranking, estado de forma, superficie, experiencia, historial entre jugadoras o incluso el momento de la temporada. Existen rankings oficiales y cuotas de apuestas que permiten estimar probabilidades de victoria pero la predicción de partidos sigue siendo un problema complejo debido a la naturaleza dinámica del deporte. Estadísticamente suele ser más impredecible partido a partido que muchos deportes colectivos. En el caso del tenis femenino y el circuito WTA los resultados son aun más volátiles puesto que no hay tanta diferencia sostenida entre las jugadoras top y el resto del circuito. Las casas de apuestas, con equipos de analistas dedicados y acceso a información privilegiada, aciertan alrededor del 68% de los partidos. 

El punto de partida de este proyecto es simple: ¿puede un modelo de ML entrenado sobre datos públicos competir con ese 68%? ¿Qué información es realmente predictiva? ¿Y cómo convertimos un clasificador partido a partido en algo útil, como la simulación completa de un Grand Slam?

### Objetivos y alcance

El proyecto tiene tres objetivos concretos:

1. **Entrenar un modelo de clasificación** que prediga la ganadora de un partido WTA a partir de features históricas de las jugadoras, con un accuracy superior al baseline del ranking (62.8%) y lo más cercano posible al benchmark de las casas de apuestas (68%).
2. **Aplicar el modelo de clasificación al cuadro de Roland Garros 2026**.  Para ello se  construirá un simulador de torneos que, dado un cuadro real, ejecute miles de simulaciones Monte Carlo usando las probabilidades predichas por el modelo y devuelva la distribución de probabilidad de victoria para cada jugadora.
3. **Desplegar una aplicación interactiva en Streamlit** que permita predecir partidos individuales y simular el cuadro completo de Roland Garros 2026.

El alcance cubre partidos WTA desde 2007 hasta mayo de 2026, con un corte temporal para el split train/test en enero de 2025.

---

## II. Dataset

### Descripción del dataset

**Fuente principal:** Dataset `wta.csv` descargado de Kaggle ([dissfya/wta-tennis-2007-2023-daily-update](https://www.kaggle.com/datasets/dissfya/wta-tennis-2007-2023-daily-update)), actualizado diariamente hasta la fecha de ejecución.

**Tamaño:** 44.446 partidos (2007–2026).

**Variables originales:**

| Variable                   | Tipo     | Descripción                                     |
| -------------------------- | -------- | ------------------------------------------------ |
| `Tournament`             | str      | Nombre del torneo                                |
| `Date`                   | datetime | Fecha del partido                                |
| `Court`                  | str      | Indoor / Outdoor                                 |
| `Surface`                | str      | Superficie (Hard, Clay, Grass, Carpet, Greenset) |
| `Round`                  | str      | Ronda del torneo                                 |
| `Best of`                | int      | Siempre 3 en WTA                                 |
| `Player_1`, `Player_2` | str      | Jugadoras enfrentadas                            |
| `Winner`                 | str      | Ganadora del partido                             |
| `Rank_1`, `Rank_2`     | int      | Ranking WTA en el momento del partido            |
| `Pts_1`, `Pts_2`       | int      | Puntos WTA                                       |
| `Odd_1`, `Odd_2`       | float    | Cuotas de casas de apuestas                      |
| `Score`                  | str      | Resultado en sets                                |

**Fuente secundaria (KMeans):** Dataset de Jeff Sackmann con estadísticas de servicio por partido (aces, dobles faltas, puntos de break, % de primer servicio, etc.) desde 2007 hasta 2026*. Se usó para intentar construir perfiles de jugadoras mediante clustering, que finalmente no se integró en el modelo final.

**No se utilizó como dataset principal porque a fecha de inicio del proyecto únicamente disponía de partidos hasta 2024. A partir del 6 de mayo se incorporaron los partidos de 2025 y 2026*

### Análisis exploratorio de datos (EDA)

Se realizó un profiling completo con `ydata_profiling` y un análisis manual de las distribuciones más relevantes.

**Distribución por superficie:**

Tras unificar Greenset y Carpet en Hard (por volumen mínimo y similitud de juego), la distribución queda:

| Superficie | Partidos       |
| ---------- | -------------- |
| Hard       | 27.290 (61.4%) |
| Clay       | 12.347 (27.8%) |
| Grass      | 4.809 (10.8%)  |

La pista rápida es claramente dominante, lo que refleja el calendario WTA real pero introduce un sesgo: el modelo tiene mucho más contexto histórico en Hard que en Grass.

**Distribución por tipo de torneo:**

Se realizó una clasificación de la columna de torneo para categorizarlos en 5 tipos.

| Categoría     | Partidos       |
| -------------- | -------------- |
| WTA250 o menor | 14.363 (32.3%) |
| WTA1000        | 10.354 (23.3%) |
| Grand Slams    | 9.455 (21.3%)  |
| WTA500         | 8.343 (18.8%)  |
| WTA Finals     | 1.931 (4.3%)   |

**Target:** El dataset está perfectamente balanceado: 22.222 victorias para Player_1 y 22.222 para Player_2 (50% / 50%), lo que tiene sentido porque la asignación de Player_1 / Player_2 es arbitraria en el dataset original.

**Cuotas de apuestas:** Se disponía de las cuotas de apuestas en 44.328 de los 44.446 partidos. No se han utilizado como variables sino como métrica adicional para valorar la calidad del predictor. Tras normalizar (eliminando el margen de la casa), la probabilidad implícita media es 0.498, confirmando el balance del dataset. La desviación estándar de 0.214 refleja que el rango va desde partidos muy igualados hasta encuentros muy desequilibrados.

> **📊 Gráficos recomendados para incluir aquí:**
>
> - Histograma de distribución de `prob_1` (probabilidad implícita de la favorita)
> - Barplot de partidos por superficie
> - Barplot de partidos por tipo de torneo
> - Serie temporal del número de partidos por año (para ver la cobertura 2007–2026)

---

## III. Preprocesamiento de los datos

### Verificación de calidad

El dataset base está en muy buen estado. Los problemas encontrados fueron menores:

- **Columnas sin valor predictivo:** `Court` (40K outdoor vs ~400 indoor, información ya capturada por la superficie) y `Best of` (siempre 3 en WTA). Se eliminaron ambas.
- **Nulos:** Únicamente 2 partidos sin fecha y 1 partido sin cuotas tras la conversión a tipos correctos. Se eliminaron los partidos sin fecha. Las odds nulas se conservaron como `NaN` (se usan para el benchmark, no para entrenar).
- **Odds con valores < 1:** Codificaban "sin datos". Se convirtieron a `NaN`.
- **Duplicados:** No se detectaron.

### Decisiones de transformación

**Unificación de superficies:** Las categorías Greenset y Carpet tenían representación mínima y características de juego similares a la pista rápida. Se mapearon a Hard para evitar categorías con muy pocos ejemplos.

**Clasificación de torneos:** El campo `Tournament` es ruidoso: los mismos torneos cambian de nombre con los patrocinadores. Se creó la función `clasificar_torneo()` basada en palabras clave de la ciudad o nombre estructural, generando la variable `tournament_type` con cinco categorías estables: `GS`, `WTA_Finals`, `WTA1000`, `WTA500`, `WTA250_o_menor`.

**Normalización de cuotas:** Las cuotas de apuestas se convirtieron a probabilidades implícitas (`1/odd`) y se normalizaron para eliminar el margen de la casa, de forma que sumen 1. Esto permite usarlas directamente como benchmark probabilístico.

**Variable target:** Se creó la columna `target` como binaria: 1 si gana Player_1, 0 si gana Player_2. El balanceo perfecto (50/50) hace innecesario ningún tratamiento adicional de desbalanceo de clases.

---

## IV. Modelado

### Feature Engineering

Todas las features se calculan con corte temporal estricto: para cada partido, solo se usa información disponible **antes** de la fecha de ese partido. Esto evita data leakage y es esencial para que las métricas de evaluación reflejen la capacidad real de predicción.

Las features construidas se agrupan en cuatro bloques:

**Bloque 1 — Forma reciente**

- `wins2meses_p1` / `wins2meses_p2`: win rate de cada jugadora en los 2 meses previos al partido. Captura el estado de forma actual, lesiones recientes y momentum.

**Bloque 2 — Historial contextual**

- `ratio_superficie_p1` / `p2`: win rate histórico en la superficie del partido. Hay jugadoras que son claramente mejores en arcilla o hierba.
- `ratio_ronda_p1` / `p2`: win rate histórico en esa ronda específica. Existe el "síndrome del cuarto de final" y jugadoras que sobre-rinden en finales.
- `h2h`: % de victorias históricas de Player_1 sobre Player_2 en enfrentamientos directos. Si no hay historial previo, se devuelve 0.5 (neutro).
- `experiencia_p1` / `p2`: número total de partidos jugados antes de ese partido.
- `is_new_p1` / `p2`: flag binario para jugadoras con menos de 10 partidos en el histórico (novatas sin contexto suficiente).

**Bloque 3 — Ranking**

- `rank_diff`: diferencia de ranking entre Player_1 y Player_2. Feature más simple pero extremadamente informativa: el ranking WTA refleja el rendimiento acumulado en los últimos 52 semanas.

**Bloque 4 — ELO**

Con las variables anteriores que fueron las iniciales los modelos mejoraban el baseline pocos puntos porcentuales. Algunos predictores de partidos de tenis usaban el sistema ELO originalmente empleado para el ajedrez para capturar información adicional al ranking.

El sistema ELO ajusta dinámicamente la habilidad estimada de cada jugadora en función de los resultados y la dificultad del rival. A diferencia del ranking WTA oficial, el ELO valora la calidad de cada victoria individual (vale más una victoria cuanto más difícil es):

$$
E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}
$$

Se calcularon dos variantes, siempre registrando el ELO **antes** del partido para evitar leakage:

- `elo_p1` / `elo_p2` / `elo_diff`: ELO por superficie (un rating independiente por Hard, Clay y Grass para cada jugadora).
- `elo_global_p1` / `elo_global_p2` / `elo_global_diff`: ELO global independiente de superficie.

**Variables categóricas:** `surface`, `round`, `tournament_type`, codificadas con OneHotEncoder.

El dataset de entrenamiento final (`historico_partidos.csv`) contiene **42.046 partidos** desde 2008 hasta 2026 con todas las features calculadas y sin ningún nulo. Puesto que todas las features se calculan tomando en cuenta los datos anteriores se eliminó 2007 ya que por no tener historial previo sus features están incompletas.

> La siguiente imagen muestra la matriz de correlación de las features numéricas (heatmap).Es interesante ver la correlación entre `rank_diff`, `elo_diff` y `elo_global_diff`, que capturan información solapada pero complementaria.
>
> ![1778503761160](image/memoria_wta_predictor_buena/1778503761160.png)

### Entrenamiento de modelos

**Split temporal train/test:** Corte en `2025-01-01`. Todo lo anterior es train, lo posterior es test. Este split respeta la naturaleza temporal de los datos y es más honesto que un split aleatorio: mide si el modelo generaliza a partidos futuros, no a partidos intercalados en el tiempo que ya "rodeaban" el entrenamiento.

- **Train:** partidos hasta diciembre 2024
- **Test:** partidos desde enero 2025 (aprox. 2.400 partidos)

**Pipeline:** Se usó un `ColumnTransformer` con `StandardScaler` para las features numéricas y `OneHotEncoder(handle_unknown='ignore')` para las categóricas. Encadenado en un `Pipeline` de scikit-learn para garantizar que el preprocesado del test usa los parámetros ajustados solo en el train.

**Modelos entrenados con `GridSearchCV`:**

*Random Forest:*

```
Mejores parámetros: max_depth=5, max_features=0.5, n_estimators=100
```

*XGBoost:*

```
Mejores parámetros: colsample_bytree=1.0, learning_rate=0.05, 
                    max_depth=3, n_estimators=200, subsample=0.8
```

Ambos modelos se mantienen deliberadamente poco profundos (`max_depth=3` o `5`) para evitar overfitting dado el nivel de ruido inherente al tenis.

**Voting Classifier:** Combinación soft de los mejores estimadores de Random Forest y XGBoost, promediando sus probabilidades. Se testó como alternativa para ver si el ensemble ganaba estabilidad pero no mejoraba los resultados de los otros modelos individualmente.

### Iteraciones y evolución del modelo

Se trazaron cuatro versiones de features, guardando resultados en CSVs para trazabilidad:

| Versión | Cambios                                               | Mejor AUC       |
| -------- | ----------------------------------------------------- | --------------- |
| v1       | Features base (ranking, forma, ratios, h2h)           | 0.704           |
| v2       | Ajuste de hiperparámetros RF, quitar tournament_type | 0.711           |
| v3       | Añadir ELO global y por superficie                   | **0.724** |
| v4       | Voting Classifier (best RF + best XGBoost)            | 0.723           |

El salto más importante fue la incorporación del ELO (v1→v3): +2 puntos de AUC. El Voting Classifier no mejoró sobre el XGBoost individual.

**KMeans de perfiles de jugadoras:** Se intentó enriquecer el modelo añadiendo un cluster de jugadora como feature categórica. Se usó el dataset de Jeff Sackmann (51.935 partidos con estadísticas de servicio) para calcular medias por jugadora de: aces, dobles faltas, % primer servicio, puntos ganados con primer y segundo servicio, break points. Tras escalar con StandardScaler y aplicar KMeans con ks variables los clusters resultantes (ver imagen inferior) no determinaban grupos nítidos  y bien separados sino más bien un continuo (mejores sacadoras vs peores sacadoras). Se decidió no incorporarlo al modelo final porque difícilmente iba a aportar información adicional.

> En la gráfica se muestra la distribución de los clusters con un k=2 (valor para el que el silhouette score era máximo) proyectados en dos ejes (PCA = 2) para verlo en un plano.
>
> ![1778504307713](image/memoria_wta_predictor_buena/1778504307713.png)

### Evaluación y selección del modelo final

**Métricas utilizadas:**

- **Accuracy:** porcentaje de partidos predichos correctamente (con umbral en 0.5).
- **AUC-ROC:** área bajo la curva ROC. Es la métrica principal de optimización porque el objetivo del modelo es generar probabilidades calibradas para la simulación, no una clasificación binaria con un umbral fijo. Un AUC de 0.72 significa que en el 72% de los casos el modelo asigna mayor probabilidad a la jugadora que efectivamente gana.

**Comparativa final de modelos:**

| Modelo                      | Accuracy         | AUC             |
| --------------------------- | ---------------- | --------------- |
| Baseline (ranking puro)     | 62.72%           | —              |
| Random Forest               | 66.24%           | 0.719           |
| XGBoost                     | 66.21%           | 0.724           |
| Voting Classifier           | 66.30%           | 0.723           |
| **Casas de apuestas** | **68.50%** | **0.757** |

El modelo final seleccionado es el **XGBoost** (guardado como `gbx_v3.model`), por su AUC ligeramente superior, métrica más relevante dado que el modelo genera probabilidades utilizadas en simulaciones Monte Carlo. Se seleccionó XGBoost frente a Random Forest por su ligera superioridad en AUC . Aunque las diferencias son marginalmente significativas, pero XGBoost ofrece además mejor calibración de probabilidades por defecto y menor dependencia de features individuales, lo que lo hace más adecuado para el objetivo de este proyecto.

### Interpretación: Feature Importance

Los valores de importancia del XGBoost y del Random Forest cuentan una historia parecida pero con distinta intensidad:

**XGBoost — Top features:**

| Feature                       | Importancia |
| ----------------------------- | ----------- |
| `elo_global_diff`           | 36.5%       |
| `elo_diff` (por superficie) | 12.0%       |
| `rank_diff`                 | 11.3%       |
| `wins2meses_p1`             | 2.6%        |
| `wins2meses_p2`             | 2.5%        |
| `experiencia_p2`            | 2.1%        |
| `ratio_superficie_p1`       | 2.1%        |
| `h2h`                       | 2.0%        |

**Random Forest — Top features:**

| Feature                       | Importancia |
| ----------------------------- | ----------- |
| `elo_global_diff`           | 54.6%       |
| `rank_diff`                 | 21.7%       |
| `elo_diff` (por superficie) | 12.8%       |

La conclusión es clara: **el diferencial de ELO global entre las dos jugadoras concentra la mayor parte del poder predictivo**, seguido del ELO por superficie y la diferencia de ranking. Las features de forma reciente, head-to-head y ratios por superficie aportan señal marginal pero consistente. Las variables de ronda y `is_new` tienen importancia casi nula: el modelo aprende que estas variables no discriminan bien.

El hecho de que las tres features más importantes sean variantes del mismo concepto (quién es mejor jugadora) confirma que el tenis es en gran medida predecible por la calidad relativa de las jugadoras, y que es el ruido inherente al deporte de competición y difícilmente capturable por un modelo (lesiones y problemas físicos, rendimiento puntualmente bueno o malo , rachas) lo que separa el 66% del modelo del 100%.

> **📊 Gráfico recomendado:** Barplot horizontal de feature importance del XGBoost (top 15 features). Ya está generado en el notebook: `02_Modelo_ML_36_0.png`.

---

## V. Predicción y resultados finales

### Solución final: arquitectura del sistema

El sistema completo se compone de tres piezas que trabajan juntas:

```
wta_limpio.csv ──→ features.py ──→ XGBoost Pipeline ──→ probabilidad [0..1]
                        ↑
              (forma_reciente, winrate,
               headtohead, experiencia,
               get_ranking, get_elo)
```

Para la simulación de torneos, se añade una capa de Monte Carlo sobre este predictor:

```
Cuadro real (128 jugadoras)
        ↓
Precalcular features por jugadora (cache)
        ↓
Repetir N=10.000 veces:
    Para cada ronda → predecir prob(A gana B) → dado cargado → avanzar ganadora
        ↓
Contar victorias → P(ganar torneo) por jugadora
```

Se realiza un precalculo de la caché de features porque sin él calcular 10.000 simulaciones de un cuadro de 128 jugadoras tardaría horas. Con la caché, una prueba de 100 simulaciones tardó 46 segundos; 10.000 simulaciones toman del orden de 70 minutos en local.

### Aplicación Streamlit

Se desarrolló una aplicación interactiva (`app.py`) con dos funcionalidades:

**1. Predictor de partido individual**
El usuario selecciona dos jugadoras y la superficie. La app calcula en tiempo real todas las features de ambas jugadoras y devuelve la probabilidad predicha por el modelo para cada una. El diseño es visual, mostrando las barras de probabilidad para que sea intuitivo.

**2. Simulador de torneo (Roland Garros 2026)**
La app carga el cuadro real del torneo, precalcula la caché de features para todas las jugadoras y ejecuta N simulaciones Monte Carlo. El resultado se presenta en dos niveles:

- **Por ronda:** tabla con las jugadoras que el modelo predice que avanzan en cada ronda y la probabilidad asociada.
- **Probabilidad de ganar el torneo:** `st.bar_chart` con la distribución completa, ordenada de mayor a menor probabilidad.

### Resultados de ejemplo: Simulación Roland Garros 2026

Una ejecución de prueba con 100 simulaciones sobre el cuadro real de Roland Garros 2026 (con fecha de corte 2026-05-05) devolvió:

| Jugadora     | % veces campeona |
| ------------ | ---------------- |
| Swiatek I.   | 24%              |
| Sabalenka A. | 20%              |
| Gauff C.     | 11%              |
| Rybakina E.  | 11%              |
| Muchova K.   | 7%               |
| Andreeva M.  | 6%               |
| Otros        | 21%              |

Swiatek como favorita en Roland Garros (su superficie y torneo histórico) es coherente con el consenso de analistas y casas de apuestas. El modelo está generando probabilidades razonables.

---

## VI. Conclusiones y futuros pasos

### Análisis de resultados

El modelo alcanza un **accuracy del 66.3% y un AUC de 0.724** en el test, frente al 62.7% del baseline de ranking y el 68.5% de las casas de apuestas. Esto sitúa el modelo en un punto intermedio: por encima de la heurística simple del ranking, y a 2 puntos de accuracy del benchmark.

La diferencia de ~2.3 puntos respecto a las casas de apuestas puede entenderse en el hecho de que las casas tienen analistas dedicados, información de mercado en tiempo real, y probablemente acceso a datos de entreno físico, lesiones no declaradas y condiciones del día que no están en ningún dataset público. Esos 2 puntos representan el "conocimiento cualitativo" que el modelo no puede capturar con datos históricos de resultados.

**Fortalezas del proyecto:**

- El pipeline es limpio y sin data leakage: el corte temporal se respeta en cada feature y en el split train/test.
- El sistema ELO aporta una representación dinámica de la habilidad que mejora significativamente sobre el ranking estático.
- La simulación Monte Carlo convierte el clasificador en algo útil y comunicable.
- La app en Streamlit hace el proyecto demostrable y entendible para cualquier audiencia.

**Debilidades y limitaciones:**

- El modelo no tiene acceso a información sobre el estado físico real de las jugadoras (lesiones, fatiga acumulada, días sin jugar).
- El ruido intrínseco del tenis individual es alto: upsets frecuentes hacen que el techo de precisión de cualquier modelo sea bajo.
- La caché de features para la simulación tarda en calcularse. Para 128 jugadoras, el tiempo de preproceso es notable.
- El KMeans de perfiles de jugadoras (basado en estadísticas de servicio) no aportó mejora, probablemente porque el dataset de estadísticas no era excesivamente exhaustivo y las variables más discriminantes ya estaban capturadas por el ELO.
- La simulación de 10.000 iteraciones en local toma del orden de horas, lo que limita su uso interactivo en tiempo real.

### Futuros pasos

**Mejoras del modelo:**

1. **Inactividad y tendencia de ranking:** Se implementaron las funciones `inactividad()` y `tendencia_ranking()` pero no se incluyeron en la versión final por no mostrar mejora clara en primeras pruebas. Merece más exploración con un rango temporal bien calibrado.
2. **Ponderación temporal en el entrenamiento:** Dar más peso en el entrenamiento a los partidos recientes (decay exponencial). Los últimos 3 años deberían pesar más que los de 2007–2012.
3. **Integrar estadísticas de servicio del dataset de Jeff Sackmann** directamente como features del partido (no como perfil de jugadora). Calcular el % de primer servicio o break points salvados en los últimos N meses sería más informativo que el promedio histórico.
4. **Calibración de probabilidades:** Evaluar si las probabilidades del modelo están bien calibradas (reliability diagram) y aplicar calibración si es necesario.
5. **Apuestas como señal de mercado:** Las cuotas de apuestas son el mejor predictor individual disponible. Explorar si incorporarlas como feature de entrada (cuando están disponibles) mejoraría significativamente el AUC, aceptando que en producción solo estarían disponibles cuando el partido ya está anunciado.

**Mejoras del sistema:**

5. **Optimizar la simulación:** Vectorizar el cálculo de features para poder ejecutar 10.000 simulaciones en minutos en lugar de horas, lo que permitiría integrarlo de forma interactiva en Streamlit.
6. **Actualización automática del cuadro:** Conectar la app a una fuente de datos en tiempo real para cargar automáticamente el cuadro del próximo Grand Slam, sin intervención manual.

---

*Stack técnico: Python · pandas · scikit-learn · XGBoost · Streamlit*
*Datos: Kaggle (WTA 2007–2026) · Jeff Sackmann (estadísticas de servicio)*
*Repositorio: notebooks `01_EDA_FeatureEngineering.ipynb`, `02_Modelo_ML.ipynb`, `03_SimulacionTorneo.ipynb` · `app.py`*
