# Predictor de Partidos WTA mediante Machine Learning

## I. Introducción

### Contexto del problema

El tenis profesional es un entorno altamente competitivo en el que intervienen múltiples factores que condicionan el resultado de un partido: ranking, estado de forma, superficie, experiencia, historial entre jugadoras o incluso el momento de la temporada. Existen rankings oficiales y cuotas de apuestas que permiten estimar probabilidades de victoria pero la predicción de partidos sigue siendo un problema complejo debido a la naturaleza dinámica del deporte.

En este contexto, el objetivo de este proyecto consiste en desarrollar un sistema de predicción de partidos de la WTA (Women’s Tennis Association) utilizando técnicas de Machine Learning. El modelo busca estimar la probabilidad de victoria de una jugadora frente a otra a partir de datos históricos de partidos disputados entre 2007 y 2026.

Además de construir un modelo predictivo, el proyecto incorpora una simulación de torneos mediante Monte Carlo y una interfaz interactiva desarrollada en Streamlit que permite realizar predicciones y simular cuadros completos de competición.

### Justificación de la necesidad

El análisis predictivo aplicado al deporte tiene múltiples aplicaciones:

- Análisis estadístico y scouting deportivo.
- Generación de simulaciones de torneos.
- Comparación con cuotas de casas de apuestas.
- Sistemas de apoyo a decisiones.
- Desarrollo de productos interactivos orientados al usuario.

El tenis resulta especialmente interesante para este tipo de problemas porque cada partido es un enfrentamiento individual donde es posible modelar variables relacionadas con el rendimiento de las jugadoras.

### Objetivos del proyecto

Los objetivos principales del proyecto son:

1. Construir un dataset histórico limpio y consistente de partidos WTA.
2. Diseñar variables predictivas relevantes mediante feature engineering.
3. Entrenar modelos de clasificación capaces de estimar probabilidades de victoria.
4. Comparar el rendimiento del modelo frente a un baseline basado únicamente en ranking y frente a las cuotas de apuestas.
5. Implementar una simulación de torneos mediante Monte Carlo.
6. Desarrollar una aplicación interactiva en Streamlit.

### Alcance del proyecto

El proyecto se centra exclusivamente en partidos WTA individuales disputados entre 2007 y 2026. El modelo predice probabilidades de victoria previas al partido y no utiliza información posterior al encuentro.

No se pretende superar a las casas de apuestas, sino construir un modelo competitivo capaz de aproximarse a sus probabilidades utilizando únicamente información histórica y variables derivadas.

---

# II. Dataset

## Origen de los datos

El dataset principal utilizado procede de Kaggle y contiene resultados históricos de partidos WTA actualizados diariamente. El conjunto incluye aproximadamente 44.000 partidos disputados entre 2007 y 2026.

Adicionalmente, para experimentar con perfiles de jugadoras mediante clustering, se utilizó el dataset público de Jeff Sackmann, que incluye estadísticas de servicio y break points. Dispone de datos de partidos históricos desde 1960... pero para ser coherente con el  El conjunto incluye más de 50.000 partidos dis

## Variables originales

El dataset principal incluye, entre otras, las siguientes variables:

| Variable            | Descripción                 |
| ------------------- | ---------------------------- |
| Player_1 / Player_2 | Jugadoras participantes      |
| Winner              | Ganadora del partido         |
| Rank_1 / Rank_2     | Ranking WTA de cada jugadora |
| Surface             | Superficie del torneo        |
| Round               | Ronda del torneo             |
| Tournament          | Nombre del torneo            |
| Date                | Fecha del partido            |
| Odd_1 / Odd_2       | Cuotas de apuestas           |
| Score               | Resultado final              |

El dataset inicial contiene 44.446 partidos y 16 variables.

## Análisis exploratorio de datos (EDA)

El análisis exploratorio permitió comprender la estructura del dataset, detectar problemas de calidad y evaluar el comportamiento de las variables.

### Distribución por superficie

Las superficies se distribuyen de forma desigual:

- Hard: aproximadamente 61%
- Clay: aproximadamente 28%
- Grass: aproximadamente 11%

Las superficies “Greenset” y “Carpet” tenían muy pocas observaciones, por lo que fueron agrupadas dentro de “Hard”.

### Balance de la variable objetivo

La variable target se construyó de manera binaria:

- 1 → gana Player_1
- 0 → gana Player_2

El dataset quedó perfectamente balanceado:

- 22.222 victorias de Player_1
- 22.222 victorias de Player_2

Este balance evita problemas típicos de clasificación desbalanceada.

### Distribución temporal

El dataset cubre casi veinte años de competición, lo que permite capturar distintas generaciones de jugadoras y cambios de rendimiento a lo largo del tiempo.

Durante el análisis exploratorio se observó que ciertas variables tenían una fuerte relación con el resultado del partido:

- Diferencia de ranking.
- Cuotas de apuestas.
- Rendimiento histórico en superficie.
- Estado de forma reciente.
- Experiencia acumulada.
- Diferencias de ELO.

### Calidad inicial de los datos

Se detectaron varios problemas:

- Valores nulos en cuotas y fechas.
- Inconsistencias en nombres de torneos.
- Variables redundantes.
- Variables con muy poca variabilidad.
- Datos categóricos no normalizados.

---

# III. Preprocesamiento de los datos

## Limpieza de datos

### Eliminación de columnas irrelevantes

Se eliminaron variables que no aportaban valor al modelo:

- Court: muy desbalanceada (casi todos los partidos eran outdoor).
- Best of: en WTA prácticamente todos los partidos son al mejor de tres sets.

### Conversión de tipos

Se transformaron variables a los tipos adecuados:

- Date → datetime.
- Odds → numérico.
- Variables categóricas → string.

### Tratamiento de valores nulos

Los valores inválidos en cuotas de apuestas se transformaron en NaN.

Los partidos sin fecha fueron eliminados debido a que imposibilitaban realizar cálculos temporales.

### Normalización de cuotas de apuestas

Las cuotas fueron convertidas a probabilidades implícitas:

Probabilidad = 1 / cuota

Posteriormente se normalizaron para eliminar el margen de beneficio de la casa de apuestas.

## Feature Engineering

La fase de feature engineering fue una de las partes más importantes del proyecto.

Todas las variables se calcularon utilizando únicamente información previa al partido para evitar data leakage.

### Variables creadas

#### 1. Diferencia de ranking

Variable:

- rank_diff = Rank_1 - Rank_2

Es la feature más importante del modelo.

### 2. Forma reciente

Se calculó el porcentaje de victorias de cada jugadora en los dos meses anteriores al partido:

- wins2meses_p1
- wins2meses_p2

Esta variable permite capturar estados de forma temporales.

### 3. Rendimiento por superficie

Se calculó el win rate histórico de cada jugadora en la superficie del partido:

- ratio_superficie_p1
- ratio_superficie_p2

### 4. Rendimiento por ronda

Algunas jugadoras tienen mejor rendimiento en rondas finales.

Se añadieron:

- ratio_ronda_p1
- ratio_ronda_p2

### 5. Head to Head (H2H)

Se calculó el histórico de enfrentamientos directos:

- h2h

Cuando no existían partidos previos se asignaba un valor neutro de 0.5.

### 6. Experiencia

Número total de partidos disputados previamente:

- experiencia_p1
- experiencia_p2

Además se creó:

- is_new_p1
- is_new_p2

para identificar jugadoras con menos de 10 partidos históricos.

### 7. Sistema ELO

Se implementó un sistema ELO personalizado para tenis.

Se calcularon dos variantes:

#### ELO global

Mide el nivel general de la jugadora independientemente de la superficie.

Variables:

- elo_global_p1
- elo_global_p2
- elo_global_diff

#### ELO por superficie

Mantiene un ELO independiente para cada superficie.

Variables:

- elo_p1
- elo_p2
- elo_diff

Estas variables mejoraron notablemente el rendimiento del modelo.

## Prevención de Data Leakage

Uno de los aspectos críticos del proyecto fue evitar el uso de información futura.

Para ello:

- Todas las features se calculan únicamente con datos anteriores al partido.
- El ELO se guarda antes de actualizarse.
- Se utilizó un split temporal en lugar de aleatorio.

---

# IV. Modelado

## Definición del problema

El problema se definió como:

- Tipo: clasificación binaria.
- Supervisado.
- Predicción probabilística.

El objetivo del modelo es estimar:

P(Player_1 gana el partido)

## Preparación de los datos

Se utilizó un pipeline de Scikit-Learn con:

- StandardScaler para variables numéricas.
- OneHotEncoder para variables categóricas.
- ColumnTransformer para combinar ambos procesos.

### Variables numéricas

- rank_diff
- wins2meses
- ratio_superficie
- ratio_ronda
- experiencia
- ELOs
- h2h

### Variables categóricas

- surface
- round
- tournament_type

## Split temporal

Para simular un entorno realista:

- Train → partidos anteriores a 2025.
- Test → partidos desde 2025.

Esto evita entrenar con información futura.

## Modelos entrenados

### Random Forest

Se entrenó un RandomForestClassifier utilizando GridSearchCV.

Ventajas:

- Robusto.
- Fácil de interpretar.
- Buen rendimiento con variables heterogéneas.

### XGBoost

El modelo final seleccionado fue XGBoost.

Ventajas:

- Mejor capacidad predictiva.
- Manejo eficiente de relaciones no lineales.
- Excelente rendimiento en problemas tabulares.

## Evaluación de modelos

### Métricas utilizadas

Se utilizaron principalmente:

- Accuracy.
- ROC-AUC.

El objetivo principal era obtener buenas probabilidades para simulaciones, por lo que AUC era especialmente relevante.

## Resultados obtenidos

| Modelo            | Accuracy | AUC  |
| ----------------- | -------- | ---- |
| Baseline ranking  | ~62.8%   | -    |
| Random Forest     | ~66%     | ~72% |
| XGBoost           | ~66%     | ~72% |
| Casas de apuestas | ~68%     | ~75% |

## Interpretación de resultados

Los resultados muestran que:

- El modelo supera al baseline basado únicamente en ranking.
- Las casas de apuestas siguen siendo superiores.
- El modelo consigue aproximarse bastante a probabilidades reales utilizando únicamente datos históricos.

## Importancia de variables

Las variables más importantes fueron:

1. rank_diff
2. elo_global_diff
3. elo_diff
4. win rate reciente
5. rendimiento en superficie

Esto confirma que el ranking y el nivel histórico siguen siendo factores dominantes en tenis profesional.

## Iteraciones y mejoras probadas

Durante el proyecto se probaron distintas mejoras:

### Ajustes de modelos

- GridSearchCV.
- Ajuste de hiperparámetros.
- Comparación entre RandomForest y XGBoost.

### Nuevas features

- Tendencia de ranking.
- Inactividad.
- ELO por superficie.
- Variables temporales.

### Clustering de jugadoras

Se intentó generar perfiles de jugadoras mediante KMeans utilizando estadísticas avanzadas:

- aces.
- dobles faltas.
- puntos ganados con primer servicio.
- break points.

El objetivo era identificar estilos de juego.

Sin embargo, la mejora obtenida fue limitada y finalmente no se incorporó al modelo final.

---

# V. Predicción y resultados finales

## Solución final

La solución final consiste en:

1. Un modelo XGBoost entrenado con más de 40.000 partidos históricos.
2. Un sistema de generación automática de features.
3. Un simulador de torneos mediante Monte Carlo.
4. Una aplicación interactiva en Streamlit.

## Simulador de partidos

La aplicación permite introducir:

- Jugadora 1.
- Jugadora 2.
- Superficie.
- Ronda.

El sistema devuelve:

- Probabilidad de victoria de cada jugadora.
- Variables relevantes utilizadas para la predicción.

## Simulación Monte Carlo

La simulación del torneo funciona de la siguiente manera:

1. El modelo predice la probabilidad de cada partido.
2. Se genera aleatoriamente una ganadora ponderando dicha probabilidad.
3. El cuadro avanza automáticamente.
4. El proceso se repite miles de veces.

De esta forma se obtiene:

- Probabilidad de ganar el torneo.
- Probabilidad de alcanzar cada ronda.
- Distribución de resultados.

## Visualización en Streamlit

La aplicación muestra:

- Resultados de cada ronda.
- Probabilidades de victoria.
- Tablas dinámicas.
- Gráficos de barras con probabilidades de ganar el torneo.

La interfaz permite convertir el modelo en una herramienta interactiva fácilmente interpretable.

## Impacto y utilidad

Aunque el proyecto tiene un enfoque académico, demuestra aplicaciones reales:

- Simulación de torneos.
- Análisis deportivo.
- Sistemas predictivos.
- Comparación con mercados de apuestas.
- Visualización interactiva de datos.

---

# VI. Conclusiones y futuros pasos

## Conclusiones principales

El proyecto demuestra que es posible construir un modelo competitivo de predicción deportiva utilizando Machine Learning y datos históricos.

Los resultados obtenidos son razonablemente sólidos considerando la complejidad del problema:

- Accuracy cercana al 66%.
- AUC alrededor del 72%.
- Mejora clara respecto a un baseline simple.

Además, la incorporación del sistema ELO permitió capturar información de nivel competitivo que no aparece directamente en el ranking WTA.

## Fortalezas del proyecto

### 1. Pipeline completo

El proyecto cubre todo el ciclo de un proyecto de Machine Learning:

- EDA.
- Limpieza.
- Feature engineering.
- Entrenamiento.
- Evaluación.
- Despliegue.

### 2. Prevención de leakage

El diseño temporal del pipeline evita fugas de información.

### 3. Aplicación práctica

La integración con Streamlit convierte el modelo en una herramienta funcional.

### 4. Simulación avanzada

La simulación Monte Carlo añade una capa probabilística muy interesante.

## Debilidades y limitaciones

### 1. Ausencia de estadísticas avanzadas completas

El dataset principal no incluye:

- velocidad de saque.
- winners.
- errores no forzados.
- estadísticas detalladas de rally.

### 2. Dependencia fuerte del ranking

El ranking sigue siendo la variable dominante.

### 3. Datos incompletos en algunas jugadoras

Jugadoras nuevas o con poca actividad tienen menos historial.

### 4. Limitaciones temporales

El rendimiento de las jugadoras cambia rápidamente debido a lesiones, descansos o cambios de entrenador.

## Futuras mejoras

### Incorporación de estadísticas avanzadas

Integrar datasets con:

- porcentajes de primer servicio.
- break points.
- winners.
- errores no forzados.

### Modelos más avanzados

Explorar:

- LightGBM.
- CatBoost.
- Redes neuronales.
- Modelos secuenciales.

### Features temporales dinámicas

Aplicar ponderaciones temporales para dar más importancia a partidos recientes.

### Mejoras en simulación

- Simulación de lesiones.
- Fatiga acumulada.
- Rendimiento por torneo.

### Automatización y despliegue

- Actualización automática de datos.
- API de predicción.
- Despliegue cloud.

## Conclusión final

El proyecto cumple satisfactoriamente los objetivos planteados y demuestra una aplicación realista de técnicas de Machine Learning al análisis deportivo.

Además de construir un modelo predictivo funcional, el trabajo permitió desarrollar un pipeline completo de ciencia de datos, desde la adquisición y limpieza de datos hasta la creación de una aplicación interactiva y un sistema de simulación probabilística.

Los resultados obtenidos muestran que, aunque las casas de apuestas siguen siendo ligeramente superiores, es posible aproximarse a sus predicciones mediante un enfoque basado exclusivamente en datos históricos y feature engineering.

En conjunto, el proyecto representa una aplicación sólida y completa de Data Science y Machine Learning orientada al deporte profesional.
