"""Genera las figuras del informe a partir de los resultados ya calculados.

Ejecutar después de ``scripts/run_pipeline.py``::

    python -m scripts.make_figures

Criterio de diseño: sin estilos exóticos, sin ejes truncados, sin colores que
codifiquen algo que no está explicado. Cada figura responde una pregunta.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C

RESULTS = C.REPORTS / "results"
AZUL, ROJO, GRIS = "#2c5f8a", "#c0392b", "#8c8c8c"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": ":",
    }
)


def _save(fig, name: str) -> None:
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES / f"{name}.png")
    plt.close(fig)
    print(f"  -> reports/figures/{name}.png")


def fig_tasa_por_categoria() -> None:
    """Tasa de abandono por categoría, con el tamaño de cada grupo visible.

    La versión original del análisis mostraba las tasas sin los conteos. Sin
    ellos no se puede saber que la tasa de 23.5% de las tarjetas Platinum
    surge de 17 clientes y 4 bajas: una diferencia que el ruido muestral
    explica por completo. Las únicas categorías cuyo intervalo excluye la tasa
    general son las dos de género, Doctorate e ingresos $60K-$80K, y en todos
    los casos por márgenes pequeños.
    """
    from src import data as D

    train = D.load_split("train")
    base = train[C.TARGET].mean()

    cols = ["Gender", "Education_Level", "Marital_Status", "Income_Category", "Card_Category"]
    fig, axes = plt.subplots(1, len(cols), figsize=(21, 4.4))

    for ax, col in zip(axes, cols):
        g = train.groupby(col)[C.TARGET].agg(["mean", "size"]).sort_values("mean")
        # intervalo binomial normal aproximado
        err = 1.96 * np.sqrt(g["mean"] * (1 - g["mean"]) / g["size"])
        ax.barh(range(len(g)), g["mean"], xerr=err, color=AZUL, alpha=0.85,
                error_kw={"ecolor": GRIS, "lw": 1})
        ax.axvline(base, color=ROJO, ls="--", lw=1.2)
        ax.set_yticks(range(len(g)))
        ax.set_yticklabels([f"{i}  (n={n})" for i, n in zip(g.index, g["size"])], fontsize=8)
        ax.set_title(C.label(col), fontsize=10)
        ax.set_xlabel("Tasa de abandono")

    fig.tight_layout(w_pad=3.5)
    fig.suptitle(
        "Las diferencias demográficas son marginales: casi todos los intervalos del 95% "
        f"cruzan la tasa general ({base:.1%}, línea roja)",
        y=1.04,
        fontsize=11,
    )
    _save(fig, "01_tasa_por_categoria")


def fig_comparacion_modelos() -> None:
    df = pd.read_csv(RESULTS / "02_comparacion_modelos.csv").sort_values("pr_auc_cv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colores = [ROJO if m == df["modelo"].iloc[-1] else AZUL for m in df["modelo"]]
    ax.barh(df["modelo"], df["pr_auc_cv"], xerr=df["pr_auc_std"], color=colores,
            alpha=0.9, error_kw={"ecolor": GRIS, "lw": 1})
    for y, v in enumerate(df["pr_auc_cv"]):
        ax.text(v + 0.008, y, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlim(0.7, 1.02)
    ax.set_xlabel("PR-AUC (validación cruzada, 5 folds)")
    ax.set_title("Los ensambles de árboles dominan a los lineales por un margen amplio")
    _save(fig, "02_comparacion_modelos")


def fig_deciles() -> None:
    d = pd.read_csv(RESULTS / "08_deciles_holdout.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))

    colores = [ROJO if t > 0.3 else AZUL for t in d["tasa_abandono"]]
    a1.bar(d["decil"], d["tasa_abandono"], color=colores, alpha=0.9)
    for x, v in zip(d["decil"], d["tasa_abandono"]):
        a1.text(x, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    a1.set_xticks(d["decil"])
    a1.set_xlabel("Decil de riesgo (1 = mayor)")
    a1.set_ylabel("Tasa de abandono observada")
    a1.set_title("Concentración del riesgo por decil")

    a2.plot(d["cartera_acum"], d["captura_acum"], "o-", color=AZUL, label="Modelo")
    a2.plot([0, 1], [0, 1], "--", color=GRIS, label="Selección al azar")
    hold = pd.read_csv(RESULTS / "07_holdout.csv", index_col=0)["valor"]
    contactada = float(hold["cartera_contactada"])
    a2.axvline(contactada, color=ROJO, ls=":", lw=1.2)
    a2.text(contactada + 0.02, 0.35, f"umbral congelado\n({contactada:.1%} contactado)",
            fontsize=8, color=ROJO)
    a2.set_xlabel("Fracción de cartera contactada")
    a2.set_ylabel("Fracción de bajas capturadas")
    a2.set_title("Curva de ganancias (holdout)")
    a2.legend(frameon=False)
    _save(fig, "03_deciles_y_ganancias")


def fig_umbrales() -> None:
    p = pd.read_csv(RESULTS / "03_politicas_umbral.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))

    destacado = {"cost": "Costo mínimo", "capacity": f"Capacidad (top {int(C.CAPACITY * 100)}%)",
                 "f1": "F1 máximo"}[C.THRESHOLD_POLICY]
    cols = [ROJO if n == destacado else AZUL for n in p["política"]]
    a1.bar(p["política"], p["umbral"], color=cols, alpha=0.9)
    for x, v in enumerate(p["umbral"]):
        a1.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
    a1.set_ylabel("Umbral de probabilidad")
    a1.set_title("Cada criterio define un corte distinto (rojo = congelado)")
    a1.tick_params(axis="x", rotation=20, labelsize=8)

    x = np.arange(len(p))
    a2.bar(x - 0.2, p["precision"], 0.4, label="Precisión", color=AZUL, alpha=0.9)
    a2.bar(x + 0.2, p["recall"], 0.4, label="Recall", color=ROJO, alpha=0.9)
    a2.set_xticks(x)
    a2.set_xticklabels(p["política"], rotation=20, fontsize=8)
    a2.set_ylim(0, 1.05)
    a2.legend(frameon=False)
    a2.set_title("El intercambio que implica cada criterio")
    _save(fig, "04_politicas_umbral")


def fig_importancias() -> None:
    imp = pd.read_csv(RESULTS / "06_importancia_permutacion.csv").head(12).iloc[::-1]
    ors = pd.read_csv(RESULTS / "05_odds_ratios.csv").head(12).iloc[::-1]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5))

    a1.barh(imp["etiqueta"], imp["importancia"], xerr=imp["desvio"],
            color=AZUL, alpha=0.9, error_kw={"ecolor": GRIS, "lw": 1})
    a1.set_xlabel("Caída de PR-AUC al permutar")
    a1.set_title("Importancia por permutación (boosting)")

    colores = [ROJO if o > 1 else "#2e7d4f" for o in ors["odds_ratio"]]
    a2.barh(ors["etiqueta"], ors["odds_ratio"], color=colores, alpha=0.9)
    a2.axvline(1, color="black", ls="--", lw=1)
    a2.set_xscale("log")
    a2.set_xlabel("Odds ratio (escala logarítmica)")
    a2.set_title("Efecto direccional (logística L1)")

    fig.suptitle(
        "Dos preguntas distintas: cuánto aporta cada variable y en qué dirección empuja",
        y=1.02, fontsize=11,
    )
    _save(fig, "05_importancias_y_odds")


def fig_segmentos() -> None:
    diag = pd.read_csv(RESULTS / "10_diagnostico_k.csv")
    perfil = pd.read_csv(RESULTS / "11_perfil_segmentos.csv")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.2))

    a1.plot(diag["k"], diag["silhouette"], "o-", color=AZUL)
    kbest = int(diag.loc[diag["silhouette"].idxmax(), "k"])
    a1.axvline(kbest, color=ROJO, ls=":", lw=1.2)
    a1.set_xlabel("K")
    a1.set_ylabel("Silhouette")
    a1.set_title(f"Selección de K (máximo en K={kbest})")

    vars_ = ["Total_Trans_Ct", "Total_Trans_Amt", "Credit_Limit", "Total_Relationship_Count"]
    disponibles = [v for v in vars_ if v in perfil.columns]
    x = np.arange(len(disponibles))
    ancho = 0.8 / len(perfil)
    for i, row in perfil.iterrows():
        vals = [row[v] for v in disponibles]
        # normalizado al máximo de cada variable, para poder comparar en un eje
        maxs = [perfil[v].max() for v in disponibles]
        a2.bar(x + i * ancho, [v / m for v, m in zip(vals, maxs)], ancho,
               label=f"Segmento {int(row['cluster'])} (n={int(row['clientes'])})")
    a2.set_xticks(x + ancho * (len(perfil) - 1) / 2)
    a2.set_xticklabels([C.label(v) for v in disponibles], rotation=15, fontsize=8)
    a2.set_ylabel("Valor relativo al máximo")
    a2.set_title("Perfil de los segmentos de alto riesgo")
    a2.legend(frameon=False, fontsize=8)
    _save(fig, "06_segmentos")


def main() -> None:
    print("Generando figuras")
    for fn in (
        fig_tasa_por_categoria,
        fig_comparacion_modelos,
        fig_deciles,
        fig_umbrales,
        fig_importancias,
        fig_segmentos,
    ):
        fn()
    print("Listo.")


if __name__ == "__main__":
    main()
