"""Genera los notebooks del repositorio.

Los notebooks son la narrativa; ``src/`` es la implementación. Este script
construye los ``.ipynb`` a partir de las celdas definidas acá, de modo que no
haya divergencia entre lo que el repo dice y lo que el código hace.

    python -m scripts.build_notebooks
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C

NOTEBOOKS = C.ROOT / "notebooks"

KERNEL = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

SETUP = """import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import config as C
from src import data as D
from src import evaluate as E
from src import interpret as I
from src import segment as S
from src.features import add_engineered_features, build_preprocessor
from src.models import build_model_zoo

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 40)"""


def _lines(text: str) -> list[str]:
    """Divide en líneas conservando el salto final de cada una.

    El formato ipynb espera que cada elemento de ``source`` termine en ``\n``,
    salvo el último. Sin eso Jupyter concatena todo el contenido en una sola
    línea y el notebook queda ilegible (y el código, sintácticamente roto).
    """
    return text.splitlines(keepends=True)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(text),
    }


# ---------------------------------------------------------------------------
# 01 — Exploración
# ---------------------------------------------------------------------------

EDA = [
    md("""# 01 — Exploración de datos

Objetivo de negocio: anticipar qué clientes están por darse de baja para poder
intervenir antes. Objetivo estadístico: estimar P(baja | atributos del cliente)
con suficiente poder de ordenamiento como para priorizar una campaña de
retención.

Este notebook responde tres preguntas y nada más:

1. ¿Qué hay en los datos y qué problemas estructurales tienen?
2. ¿Cuán desbalanceada está la variable objetivo y qué implica eso para las métricas?
3. ¿Qué variables se asocian con el abandono y cuáles no, más allá del ruido?"""),
    code(SETUP),
    md("""## 1. Carga y verificación estructural

`src/data.py` es la única puerta de entrada a los CSV. Antes de mirar nada,
verifica las identidades algebraicas conocidas entre columnas."""),
    code("""crudo = D.load_raw(C.TRAIN_CSV)
print(f"Filas: {len(crudo):,}   Columnas: {crudo.shape[1]}")
print(f"Nulos: {crudo.isna().sum().sum()}")
print(f"Filas duplicadas: {crudo.duplicated().sum()}")

for identidad, error in D.check_linear_dependencies(crudo).items():
    print(f"\\n{identidad}\\n   error absoluto máximo: {error:.2e}")"""),
    md("""El error es del orden de 1e-14, es decir, cero numérico:
`Avg_Open_To_Buy` es una combinación lineal exacta de otras dos columnas. No
aporta información y sí genera problemas: infla la varianza de los coeficientes
en los modelos lineales y reparte la importancia entre columnas redundantes en
los de árboles. Se descarta.

La columna `Unnamed: 0` es el índice de fila del CSV. Usarla como predictor es
un error clásico: si el archivo estuviera ordenado por alguna variable
relacionada con el target, el modelo aprendería el orden de las filas."""),
    code("""train = add_engineered_features(D.load_split("train"))
test = add_engineered_features(D.load_split("test"))

pd.DataFrame([D.describe_split(train, "train"), D.describe_split(test, "holdout")])"""),
    md("""Las dos particiones tienen prácticamente la misma tasa de abandono
(16.07% y 16.04%), lo que sugiere un split aleatorio estratificado o
suficientemente grande. El holdout **no se vuelve a tocar** hasta el notebook 02,
y una sola vez."""),
    md("""## 2. La variable objetivo

Con 16% de positivos, un modelo que prediga "nadie se va" acierta el 84% de las
veces. Ese número es la razón por la cual accuracy no sirve acá."""),
    code("""base = E.baseline_metrics(train[C.TARGET])
print("Modelo trivial (predecir 0 para todos):")
for k, v in base.items():
    print(f"  {k:>10}: {v:.4f}")

print("\\nUn modelo con 84% de accuracy y 0% de recall no identifica")
print("un solo cliente en riesgo. La métrica de referencia será PR-AUC,")
print("cuya línea base es la prevalencia (0.16), no 0.5.")"""),
    md("""## 3. Variables numéricas

Correlación con el target. Es lineal y bivariada, así que subestima relaciones
no monótonas y no captura interacciones; sirve como primer mapa, no como
selección de variables."""),
    code("""num = [c for c in train.select_dtypes("number").columns if c != C.TARGET]
corr = train[num + [C.TARGET]].corr()[C.TARGET].drop(C.TARGET).sort_values()

fig, ax = plt.subplots(figsize=(8, 6))
colores = ["#c0392b" if v > 0 else "#2c5f8a" for v in corr]
ax.barh([C.label(i) for i in corr.index], corr.values, color=colores, alpha=0.9)
ax.axvline(0, color="black", lw=1)
ax.set_xlabel("Correlación de Pearson con abandono")
ax.set_title("Rojo: asociado a mayor riesgo. Azul: a menor riesgo.")
ax.grid(alpha=0.25, ls=":")
plt.show()"""),
    md("""El patrón es coherente y transaccional: **transaccionar menos, hacerlo
por montos menores y desacelerar entre trimestres** se asocia al abandono, y
**contactar más al banco** también, lo que sugiere fricción antes de la baja.

Ninguna correlación supera 0.4 en valor absoluto. Sin embargo, los modelos de
árboles llegan a PR-AUC de 0.96 en el notebook 02. La brecha es informativa: el
poder predictivo está en las **interacciones**, no en los efectos marginales."""),
    md("""## 4. Variables categóricas

Acá conviene mirar con cuidado. La versión original de este análisis reportaba
que las tarjetas Platinum tienen una tasa de abandono de 23.5%, la más alta de
todas, y la interpretaba como un segmento premium en riesgo.

El problema: son 17 clientes y 4 bajas."""),
    code("""def tabla_categoria(df, col):
    g = df.groupby(col)[C.TARGET].agg(conteo="size", bajas="sum", tasa="mean")
    # intervalo de confianza binomial normal al 95%
    err = 1.96 * np.sqrt(g["tasa"] * (1 - g["tasa"]) / g["conteo"])
    g["ic_inferior"] = (g["tasa"] - err).clip(0)
    g["ic_superior"] = (g["tasa"] + err).clip(upper=1)
    return g.sort_values("tasa", ascending=False).round(3)

tabla_categoria(train, "Card_Category")"""),
    code("""base_rate = train[C.TARGET].mean()

# ¿Qué categorías tienen un intervalo que NO contiene la tasa general?
# Sólo esas son distinguibles del promedio de la cartera.
for col in C.CATEGORICAL:
    g = tabla_categoria(train, col)
    excluye = ~((g["ic_inferior"] <= base_rate) & (g["ic_superior"] >= base_rate))
    print(f"{C.label(col):<24} {excluye.sum()}/{len(g)} categorías distinguibles"
          + (f"  ->  {', '.join(g.index[excluye])}" if excluye.any() else ""))"""),
    md("""De 23 categorías repartidas en cinco variables, sólo cuatro tienen
intervalos que excluyen la tasa general:

| categoría | n | tasa | IC 95% |
|---|---|---|---|
| Género femenino | 4.272 | 17,5% | [16,4% – 18,7%] |
| Género masculino | 3.829 | 14,4% | [13,3% – 15,6%] |
| Nivel educativo: Doctorate | 356 | 20,5% | [16,3% – 24,7%] |
| Ingresos $60K–$80K | 1.136 | 13,8% | [11,8% – 15,8%] |

Las dos categorías de género se distinguen porque los grupos son grandes, no
porque la diferencia sea grande: 3,1 puntos porcentuales. Las otras dos apenas
rozan el borde del intervalo, y con cinco variables y 23 categorías examinadas,
encontrar dos casos marginales al 5% es lo que se espera por azar.

Lo importante es que **ninguna de estas diferencias resulta útil para
predecir**: en el notebook 02, la importancia por permutación deja al género
fuera de las diez primeras variables. Una diferencia puede ser
estadísticamente distinguible y a la vez inútil para ordenar clientes por
riesgo, y confundir esas dos cosas es un error frecuente al leer tablas como
esta.

La conclusión operativa se mantiene: la demografía no explica el abandono en
este dataset, y el análisis se concentra en el comportamiento transaccional. La
figura `reports/figures/01_tasa_por_categoria.png` muestra los intervalos
graficados junto al tamaño de cada grupo."""),
    md("""## 5. Variables derivadas

`src/features.py` construye cinco variables de intensidad transaccional
(ticket promedio, transacciones por mes, contactos por producto, ratio de
inactividad, caída de actividad) más un indicador de saldo rotativo en cero.

La motivación viene de la literatura de churn bancario: Brito et al. (2024)
encuentran que las variables de recencia, frecuencia y valor monetario superan
a las demográficas. El dataset no trae fechas, así que recencia no se puede
construir, pero sí intensidades normalizadas por antigüedad.

**El notebook 02 muestra que estas variables no mejoran la performance.** Se
documentan igual, con el resultado de la ablación."""),
    code("""derivadas = ["ticket_promedio", "trans_por_mes", "contactos_por_producto",
             "ratio_inactividad", "caida_actividad", "saldo_rotativo_cero"]

comparacion = pd.DataFrame({
    "correlación con abandono": train[derivadas].corrwith(train[C.TARGET]).round(3),
    "media (se quedan)": train.loc[train[C.TARGET] == 0, derivadas].mean().round(2),
    "media (se van)": train.loc[train[C.TARGET] == 1, derivadas].mean().round(2),
})
comparacion.index = [C.label(i) for i in comparacion.index]
comparacion"""),
    md("""## Qué se lleva el notebook 02

- `Avg_Open_To_Buy` y el índice de fila quedan fuera.
- Ninguna variable demográfica discrimina; la señal es transaccional.
- Ninguna correlación bivariada es fuerte, así que hay que probar modelos
  capaces de capturar interacciones.
- Accuracy queda descartada como métrica de selección; se usa PR-AUC."""),
]


# ---------------------------------------------------------------------------
# 02 — Modelado
# ---------------------------------------------------------------------------

MODELADO = [
    md("""# 02 — Modelado y decisión

Dos objetos distintos, que se eligen con criterios distintos:

- un **modelo**, que produce una probabilidad de baja;
- una **política**, que convierte esa probabilidad en la decisión de contactar
  o no contactar.

Confundirlos es el error más común en este tipo de trabajo. Un umbral de 0.5 no
es una propiedad del modelo: es una convención heredada de problemas
balanceados, sin ningún contenido económico acá.

## Protocolo

1. `churn_train.csv` se usa para todo el desarrollo.
2. Selección de modelo por validación cruzada estratificada de 5 folds sobre
   train, optimizando PR-AUC.
3. El umbral se fija con predicciones **fuera de muestra** (`cross_val_predict`)
   sobre train.
4. `churn_test.csv` se evalúa **una sola vez**, al final, con modelo y umbral
   ya congelados.

El paso 3 es el que corrige el problema principal de la versión original: si el
umbral se busca sobre el mismo conjunto donde después se reporta performance, es
un hiperparámetro ajustado al test y las métricas quedan infladas."""),
    code(SETUP),
    code("""from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)

train = add_engineered_features(D.load_split("train"))
test = add_engineered_features(D.load_split("test"))

X_train, y_train = D.split_X_y(train)
X_test, y_test = D.split_X_y(test)

cv = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED)
print(f"Desarrollo: {len(X_train):,} filas   Holdout: {len(X_test):,} filas")"""),
    md("""## 1. Por qué el preprocesamiento va dentro del pipeline

Si el `StandardScaler` y el `OneHotEncoder` se ajustaran una sola vez sobre
todos los datos y después se pasaran las matrices ya transformadas a la
validación cruzada, cada fold de validación habría contribuido a la media y el
desvío del escalado. El sesgo es chico pero sistemático, y desaparece gratis
metiendo el preprocesador dentro del `Pipeline`.

Todos los modelos de `src/models.py` están construidos así."""),
    code("""zoo = build_model_zoo(engineered=True)

pd.DataFrame([
    {"modelo": s.name,
     "escalado": "sí" if s.needs_scaling else "no",
     "combinaciones": int(np.prod([len(v) for v in s.param_grid.values()]) or 1),
     "rol": s.nota}
    for s in zoo
]).set_index("modelo")"""),
    md("""Cuatro familias con supuestos distintos:

- **Lineales regularizados** (L2, L1, elastic net): asumen que el log-odds es
  lineal en los predictores. Son el baseline interpretable.
- **KNN**: no paramétrico, sin forma funcional impuesta, pero sensible a la
  escala y a la dimensión.
- **Árbol con poda de coste-complejidad**: captura interacciones y se puede
  leer como reglas.
- **Ensambles**: bagging (Random Forest, Extra Trees) y boosting
  (Hist Gradient Boosting).

Sobre el desbalance: se usa `class_weight="balanced"`, que reponderar la clase
minoritaria dentro de la función de pérdida, en vez de remuestreo sintético
(SMOTE, ADASYN). Con 16% de positivos y 8.100 filas la clase rara no es tan rara
como para necesitar ejemplos artificiales, y reponderar no distorsiona la
distribución conjunta de las variables. El desbalance real se maneja en el
umbral, no cambiando los datos."""),
    md("""## 2. Validación cruzada

La celda siguiente tarda varios minutos. Los resultados ya calculados están en
`reports/results/02_comparacion_modelos.csv`."""),
    code("""resultados, ajustados = [], {}

for spec in zoo:
    busqueda = GridSearchCV(spec.estimator, spec.param_grid,
                            scoring="average_precision", cv=cv, n_jobs=-1)
    busqueda.fit(X_train, y_train)
    ajustados[spec.name] = busqueda.best_estimator_
    i = busqueda.best_index_
    resultados.append({
        "modelo": spec.name,
        "pr_auc_cv": busqueda.cv_results_["mean_test_score"][i],
        "pr_auc_std": busqueda.cv_results_["std_test_score"][i],
        "mejores_params": busqueda.best_params_,
    })
    print(f"{spec.name:<26} PR-AUC = {resultados[-1]['pr_auc_cv']:.4f}")

comparacion = pd.DataFrame(resultados).sort_values("pr_auc_cv", ascending=False)
comparacion.set_index("modelo").round(4)"""),
    md("""El orden es nítido: boosting (0.967) > bagging (0.943 y 0.923) > árbol
único (0.854) > KNN (0.833) > lineales (0.795).

La brecha de 17 puntos entre el boosting y las logísticas confirma lo que
anticipaba el notebook 01: **el poder predictivo está en las interacciones**.
Ninguna correlación bivariada superaba 0.4 y sin embargo el ordenamiento es
casi perfecto.

Las tres logísticas empatan en la tercera cifra decimal. Eso significa que la
regularización no está haciendo trabajo: con 24 predictores y 8.100
observaciones no hay sobreajuste que corregir. El Lasso no descarta variables
porque no le sobran."""),
    md("""## 3. Ablación: ¿sirven las variables derivadas?

La pregunta honesta sobre cualquier feature engineering es si aporta algo. Se
compara el mismo modelo con y sin las variables construidas."""),
    code("""from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

for con_derivadas in (False, True):
    df = D.load_split("train")
    if con_derivadas:
        df = add_engineered_features(df)
    X, y = D.split_X_y(df)
    pipe = Pipeline([
        ("pre", build_preprocessor(engineered=con_derivadas, scale=False)),
        ("clf", HistGradientBoostingClassifier(random_state=C.SEED)),
    ])
    s = cross_val_score(pipe, X, y, scoring="average_precision", cv=cv)
    print(f"variables derivadas = {str(con_derivadas):<5}  "
          f"PR-AUC = {s.mean():.4f} ± {s.std():.4f}")"""),
    md("""**0.9679 sin ellas, 0.9678 con ellas.** Las variables derivadas no
aportan nada.

Tiene sentido: el boosting parte recursivamente el espacio de predictores, y un
ratio como `monto / cantidad` es reconstruible mediante cortes sucesivos en sus
dos componentes. El feature engineering ayuda cuando el modelo no puede expresar
la transformación por sí mismo, que es el caso de los lineales, no el de los
ensambles de árboles.

Se dejan en el repositorio porque mejoran la interpretabilidad de la logística,
y porque el resultado negativo es parte del análisis."""),
    md("""## 4. Del score a la decisión

El modelo devuelve probabilidades. Falta decidir a partir de qué valor se
contacta a un cliente. Se evalúan cuatro criterios sobre predicciones fuera de
muestra:

- **F1 máximo**: criterio estadístico, equilibra precisión y recall sin
  referencia a costos.
- **Capacidad**: contacta al 30% más riesgoso, el techo operativo del equipo
  de retención.
- **Costo mínimo**: minimiza el costo esperado. Perder un cliente cuesta 100;
  contactar a uno que se iba a quedar, 15. Son supuestos ilustrativos: lo que
  importa es la relación entre ambos, no la magnitud.
- **Convención 0.5**: la referencia a batir."""),
    code("""mejor_nombre = comparacion.iloc[0]["modelo"]
mejor = ajustados[mejor_nombre]

oof = cross_val_predict(mejor, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
politicas = E.compare_threshold_policies(y_train, oof)
politicas.round(4)"""),
    md("""Comparar las filas de capacidad y costo mínimo:

| criterio | cartera contactada | precisión | recall | costo |
|---|---|---|---|---|
| Capacidad (30%) | 30.0% | 0.53 | 0.995 | 17.740 |
| Costo mínimo | 20.0% | 0.78 | 0.972 | 8.880 |

Contactar al 30% en vez del 20% gana 2,2 puntos de recall y **duplica el
costo**. La capacidad del equipo es un techo, no un objetivo: agotarla porque
existe multiplica los falsos positivos sin ganar casi nada.

Por eso el umbral congelado usa el criterio de costo, con `THRESHOLD_POLICY` en
`src/config.py`. El 20% que resulta está holgadamente por debajo del techo del
30%, así que la restricción operativa no es activa."""),
    code("""umbral = E.freeze_threshold(y_train, oof, C.THRESHOLD_POLICY)
print(f"Umbral congelado ({C.THRESHOLD_POLICY}): {umbral:.4f}")
print(f"Contacta al {(oof >= umbral).mean():.1%} de la cartera "
      f"(techo operativo: {C.CAPACITY:.0%})")

E.decile_table(y_train, oof)"""),
    md("""La tabla de deciles se calcula sobre predicciones **out-of-fold**, no
sobre predicciones en entrenamiento. Ese es el punto que corrige el error de la
versión original, donde el modelo se reajustaba sobre todos los datos y después
se scoreaba ese mismo conjunto.

El primer decil concentra el 61% de las bajas con un lift de 6,1. Los primeros
dos deciles acumulan el 97%."""),
    md("""## 5. Interpretabilidad

Dos preguntas distintas que requieren herramientas distintas.

**Cuánto aporta cada variable**: importancia por permutación, calculada fuera
de muestra. Se prefiere al `feature_importances_` de los árboles, que se
calcula en entrenamiento y sobrevalora a las variables de alta cardinalidad.

**En qué dirección empuja**: odds ratios de la logística L1.

Sobre esto último hay que ser explícito. La versión original de este trabajo
ajustaba `LassoCV` —que es **regresión lineal** con penalización L1— sobre un
target binario, y después exponenciaba los coeficientes para presentarlos como
odds ratios. Eso no es válido: exponenciar un coeficiente de mínimos cuadrados
no produce un odds ratio, produce un número sin interpretación. Los odds ratios
requieren un modelo logístico. `src/interpret.py` levanta un `TypeError` si se
le pasa un estimador sin `coef_` logístico."""),
    code("""imp = I.permutation_importances(mejor, X_train, y_train, n_repeats=5)
imp.head(10).round(4)"""),
    code("""ors = I.odds_ratios(ajustados["logistica_l1"], top=12)
ors[["etiqueta", "coeficiente", "odds_ratio", "efecto"]].round(3)"""),
    md("""Las dos listas coinciden en lo esencial —cantidad y monto de
transacciones dominan— pero difieren en el orden, porque miden cosas distintas.
La logística atribuye un odds ratio alto al ticket promedio; el boosting le da
un tercio de la importancia que le da a la cantidad de transacciones. La
diferencia viene de que la logística no puede representar la interacción entre
monto y cantidad, y la comprime en el cociente.

Un detalle a no sobreinterpretar: el género masculino aparece con odds ratio
0.44. En el notebook 01 vimos que la diferencia por género no sobrevive al
intervalo de confianza. Un coeficiente ajustado por otras variables puede ser
estadísticamente distinguible sin que la variable tenga poder predictivo real:
en la importancia por permutación, el género no aparece entre las diez
primeras."""),
    md("""## 6. Holdout: una sola evaluación

Modelo y umbral congelados. Esta celda se ejecuta una vez."""),
    code("""mejor.fit(X_train, y_train)
proba_test = mejor.predict_proba(X_test)[:, 1]

libres = E.threshold_free_metrics(y_test, proba_test)
decision = E.metrics_at_threshold(y_test, proba_test, umbral)

print("Métricas independientes del umbral")
for k, v in libres.items():
    print(f"  {k:>14}: {v:.4f}")

print(f"\\nDecisión al umbral congelado ({umbral:.4f})")
for k in ["precision", "recall", "f1", "accuracy", "cartera_contactada"]:
    print(f"  {k:>18}: {decision[k]:.4f}")
print(f"\\n  Matriz de confusión: TP={decision['TP']} FP={decision['FP']} "
      f"FN={decision['FN']} TN={decision['TN']}")"""),
    md("""ROC-AUC 0.992, PR-AUC 0.965, Brier 0.023. Al umbral congelado:
precisión 0.78, recall 0.95, contactando al 19,5% de la cartera.

Las métricas del holdout coinciden con las de validación cruzada, lo que indica
que no hubo sobreajuste en la selección.

**Advertencia necesaria.** Un ROC-AUC de 0.99 no es normal y no debe leerse
como mérito del modelo. Se auditó: no hay filas duplicadas dentro de cada
partición, no hay solapamiento entre train y test, y ninguna variable separa
por sí sola (el mejor AUC univariado es 0.796). El dataset —el
*Credit Card Customers* de Kaggle— es sintético o fuertemente depurado, y
resulta casi separable para ensambles de árboles. Sobre datos bancarios reales,
con clases del orden del 2% y ruido de medición, cualquier resultado de esta
magnitud sería motivo para buscar el error antes que para celebrar."""),
    code("""E.decile_table(y_test, proba_test)"""),
]


# ---------------------------------------------------------------------------
# 03 — Segmentación
# ---------------------------------------------------------------------------

SEGMENTACION = [
    md("""# 03 — Segmentación de la cohorte de alto riesgo

El clustering no compite con el modelo predictivo: lo complementa. El modelo
dice *quién* tiene riesgo alto. El clustering pregunta si dentro de ese grupo
hay perfiles distintos que justifiquen acciones de retención distintas.

Dos decisiones de diseño:

**Se clusteriza sólo dentro de la cohorte priorizada**, no sobre toda la
cartera. Agrupar los 2.026 clientes del holdout produciría sobre todo la
separación entre activos e inactivos, que ya conocemos por el modelo y que no
agrega nada accionable.

**Se reduce la dimensión con PCA antes de aplicar k-means.** K-means mide
distancia euclídea, que en alta dimensión se concentra y pierde poder de
discriminación, y las dummies del one-hot inflan la dimensión aportando poca
varianza."""),
    code(SETUP),
    code("""from sklearn.model_selection import StratifiedKFold

train = add_engineered_features(D.load_split("train"))
test = add_engineered_features(D.load_split("test"))
X_train, y_train = D.split_X_y(train)
X_test, y_test = D.split_X_y(test)

# Se carga el modelo ya ajustado y el umbral congelado que dejó
# scripts/run_pipeline.py. Reajustar acá con hiperparámetros por defecto daría
# un modelo distinto del que se validó, y por lo tanto otra cohorte.
import joblib
import json

congelado = json.loads((C.REPORTS / "results" / "modelo_congelado.json").read_text())
umbral = congelado["umbral"]
mejor = joblib.load(C.MODELS_DIR / "modelo_final.joblib")

proba_test = mejor.predict_proba(X_test)[:, 1]

cohorte = test.copy()
cohorte["score"] = proba_test
cohorte = cohorte[cohorte["score"] >= umbral].copy()

print(f"Cohorte priorizada: {len(cohorte)} clientes "
      f"({len(cohorte) / len(test):.1%} del holdout)")
print(f"Tasa de abandono en la cohorte: {cohorte[C.TARGET].mean():.3f} "
      f"(vs {test[C.TARGET].mean():.3f} en la cartera completa)")"""),
    md("""## 1. Cuántos grupos

Se comparan silhouette, Calinski-Harabasz e inercia. La inercia decrece siempre
al aumentar K, así que por sí sola no decide nada; el codo suele ser ambiguo con
datos de clientes."""),
    code("""Z, _ = S.build_cluster_matrix(cohorte, mejor, n_components=0.90)
print(f"Matriz de clustering: {Z.shape[0]} clientes × {Z.shape[1]} componentes "
      f"(90% de la varianza)")

k, diagnostico = S.choose_k(Z)
diagnostico"""),
    md("""El máximo de silhouette está en K=2 (0.185) y cae abruptamente a 0.099
en K=3. Calinski-Harabasz apunta en la misma dirección.

Hay que ser honesto sobre la magnitud: un silhouette de 0.185 es **bajo**.
Indica que los grupos se solapan considerablemente. Con datos de comportamiento
de clientes esto es lo esperable —no hay tipos discretos de cliente, hay un
continuo— pero significa que la segmentación describe una tendencia, no una
partición nítida. Presentarla como "encontramos dos perfiles claramente
diferenciados" sería exagerar lo que los datos sostienen."""),
    code("""labels = S.fit_kmeans(Z, k)

# contraste con un modelo de mezclas gaussianas, que admite clusters
# elípticos y de tamaños distintos
labels_gmm = S.fit_gmm(Z, k)
acuerdo = (labels == labels_gmm).mean()
acuerdo = max(acuerdo, 1 - acuerdo)  # las etiquetas son arbitrarias
print(f"Acuerdo entre k-means y GMM: {acuerdo:.1%}")"""),
    md("""## 2. Perfil de los segmentos

En unidades originales, no estandarizadas: el equipo comercial necesita leer
"transaccionan 41 veces al año", no "-0,8 desvíos"."""),
    code("""columnas = ["Total_Trans_Ct", "Total_Trans_Amt", "ticket_promedio",
            "Credit_Limit", "Total_Revolving_Bal", "Total_Relationship_Count",
            "Contacts_Count_12_mon", "Total_Ct_Chng_Q4_Q1", "Customer_Age"]

perfil = S.profile_clusters(cohorte, labels, columnas)
perfil.columns = [C.label(c) if c in C.LABELS_ES else c for c in perfil.columns]
perfil"""),
    md("""Los dos segmentos, sobre la cohorte priorizada del holdout:

**Segmento 0 — alto volumen, 20% de la cohorte (79 clientes).** Límite de
tarjeta de ~14.500, monto anual de ~7.500 con ticket promedio de ~109, y 69
transacciones al año. Su indicador de deterioro es la desaceleración: la
variación trimestral cayó a 0,80.

**Segmento 1 — bajo volumen, 80% de la cohorte (317 clientes).** Límite de
~6.900, monto anual de ~2.200 con ticket promedio de ~53, y 41 transacciones.
La caída trimestral es mucho más pronunciada (0,51) y el ratio de utilización
es tres veces mayor.

Las tasas de abandono son casi idénticas (0,80 y 0,77): **los segmentos no se
diferencian en riesgo, se diferencian en valor**. Esa es exactamente su
utilidad. El modelo ya ordenó por riesgo; el clustering agrega la dimensión que
falta para decidir cuánto invertir en retener a cada uno."""),
    code("""cat = S.categorical_profile(cohorte, labels)
cat"""),
    md("""La composición demográfica de los dos segmentos es prácticamente la
misma, consistente con el notebook 01: la demografía no separa nada en este
dataset. La segmentación es puramente conductual."""),
    md("""## 3. Qué hacer con esto

La segmentación sugiere dos tratamientos distintos, con la salvedad de que el
solapamiento es alto y esto es una orientación, no una asignación estricta:

**Segmento 0 (alto volumen).** Pocos clientes, mucho valor por cliente.
Justifica contacto humano y ofertas individualizadas. La señal a monitorear es
la desaceleración trimestral, que aparece antes de que caiga el volumen anual.

**Segmento 1 (bajo volumen).** Cuatro veces más clientes, un tercio del monto
por cliente. El costo de contacto individual no se paga; corresponden acciones
automatizadas. La utilización de crédito alta con montos bajos sugiere clientes
con límite ajustado, para quienes una revisión de límite puede ser más efectiva
que un descuento.

**Lo que estos datos no permiten decir.** No hay costo de adquisición, margen
por cliente ni valor de vida, así que no se puede calcular el retorno de una
campaña. Los costos usados para fijar el umbral (100 y 15) son supuestos
ilustrativos elegidos por su relación, no cifras de un banco. Cualquier
implementación real tiene que reemplazarlos por números propios antes de
convertir estas probabilidades en presupuesto."""),
]


def build(nombre: str, celdas: list[dict]) -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    nb = {"cells": celdas, "metadata": KERNEL, "nbformat": 4, "nbformat_minor": 5}
    destino = NOTEBOOKS / nombre
    destino.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  -> notebooks/{nombre}  ({len(celdas)} celdas)")


def main() -> None:
    print("Generando notebooks")
    build("01_exploracion.ipynb", EDA)
    build("02_modelado.ipynb", MODELADO)
    build("03_segmentacion.ipynb", SEGMENTACION)
    print("Listo.")


if __name__ == "__main__":
    main()
