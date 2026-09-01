"""Pipeline completo, de los CSV crudos a los resultados reportados.

Ejecutar desde la raíz del repositorio::

    python -m scripts.run_pipeline

Protocolo experimental, en orden:

1. ``churn_train.csv`` se usa para todo el desarrollo. ``churn_test.csv`` es el
   holdout final y no se toca hasta el paso 5.
2. Cada candidato se evalúa con validación cruzada estratificada de 5 folds
   sobre train, optimizando PR-AUC.
3. El ganador se elige por PR-AUC de validación cruzada, no por accuracy.
4. El umbral se fija con predicciones fuera de muestra (``cross_val_predict``)
   sobre train. Esto es lo que evita que el umbral quede ajustado al conjunto
   donde después se reporta performance.
5. Recién entonces se evalúa una vez sobre el holdout, con el modelo y el
   umbral ya congelados.
"""

from __future__ import annotations

import json
import pickle
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

from src import config as C
from src import data as D
from src import evaluate as E
from src import interpret as I
from src import segment as S
from src.features import add_engineered_features
from src.models import build_model_zoo

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

RESULTS = C.REPORTS / "results"


def _save(df: pd.DataFrame, name: str) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS / f"{name}.csv", index=True)
    print(f"  -> reports/results/{name}.csv")


def main() -> None:
    rng = np.random.default_rng(C.SEED)
    cv = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED)

    # ---------------------------------------------------------------- 1. datos
    print("\n[1] Datos")
    train = add_engineered_features(D.load_split("train"))
    test = add_engineered_features(D.load_split("test"))

    resumen = pd.DataFrame(
        [D.describe_split(train, "train"), D.describe_split(test, "holdout")]
    )
    print(resumen.to_string(index=False))
    _save(resumen.set_index("partición"), "01_particiones")

    X_train, y_train = D.split_X_y(train)
    X_test, y_test = D.split_X_y(test)

    base = E.baseline_metrics(y_train)
    print(
        f"\n  Modelo trivial (predecir 0 para todos): "
        f"accuracy={base['accuracy']:.4f}, recall={base['recall']:.4f}"
    )

    # ------------------------------------------------------- 2. selección de modelo
    print("\n[2] Validación cruzada de candidatos (PR-AUC, 5 folds sobre train)")
    zoo = build_model_zoo(engineered=True)
    filas, fitted = [], {}
    cache = C.ROOT / ".cache"
    cache.mkdir(exist_ok=True)

    for spec in zoo:
        ckpt = cache / f"{spec.name}.pkl"
        if ckpt.exists():
            fila, est = pickle.loads(ckpt.read_bytes())
            filas.append(fila)
            fitted[spec.name] = est
            print(f"  {spec.name:<26} PR-AUC={fila['pr_auc_cv']:.4f} (cache)")
            continue
        search = GridSearchCV(
            spec.estimator,
            spec.param_grid,
            scoring="average_precision",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        fitted[spec.name] = search.best_estimator_

        idx = search.best_index_
        filas.append(
            {
                "modelo": spec.name,
                "pr_auc_cv": search.cv_results_["mean_test_score"][idx],
                "pr_auc_std": search.cv_results_["std_test_score"][idx],
                "mejores_params": json.dumps(
                    {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
                    default=str,
                ),
                "nota": spec.nota,
            }
        )
        print(
            f"  {spec.name:<26} PR-AUC={filas[-1]['pr_auc_cv']:.4f} "
            f"± {filas[-1]['pr_auc_std']:.4f}",
            flush=True,
        )
        ckpt.write_bytes(pickle.dumps((filas[-1], search.best_estimator_)))

    comparacion = (
        pd.DataFrame(filas).sort_values("pr_auc_cv", ascending=False).set_index("modelo")
    )
    _save(comparacion.round(4), "02_comparacion_modelos")

    ganador = comparacion.index[0]
    best = fitted[ganador]
    print(f"\n  Modelo elegido: {ganador}")

    # ----------------------------------------------- 3. umbral, fuera de muestra
    print("\n[3] Política de umbral (predicciones out-of-fold sobre train)")
    oof = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba")[:, 1]

    politicas = E.compare_threshold_policies(y_train, oof)
    print(politicas.round(4).to_string(index=False))
    _save(politicas.round(4).set_index("política"), "03_politicas_umbral")

    umbral = E.freeze_threshold(y_train, oof, C.THRESHOLD_POLICY)
    print(
        f"\n  Umbral congelado (criterio '{C.THRESHOLD_POLICY}'): {umbral:.4f}"
        f"  ->  contacta al {(oof >= umbral).mean():.1%} de la cartera"
        f" (techo operativo: {C.CAPACITY:.0%})"
    )

    deciles_oof = E.decile_table(y_train, oof)
    print("\n  Deciles de riesgo (out-of-fold):")
    print(deciles_oof.to_string())
    _save(deciles_oof, "04_deciles_oof")

    # --------------------------------------------------- 4. interpretabilidad
    print("\n[4] Interpretabilidad")
    logistica = fitted["logistica_l1"]
    ors = I.odds_ratios(logistica, top=15)
    print("\n  Odds ratios (logística L1, top 15):")
    print(ors[["etiqueta", "coeficiente", "odds_ratio", "efecto"]].round(3).to_string(index=False))
    _save(ors.round(4).set_index("variable"), "05_odds_ratios")

    imps = I.permutation_importances(best, X_train, y_train, n_repeats=5)
    print("\n  Importancia por permutación (top 10):")
    print(imps.head(10)[["etiqueta", "importancia", "desvio"]].round(4).to_string(index=False))
    _save(imps.round(5).set_index("variable"), "06_importancia_permutacion")

    # ------------------------------------------------------ 5. holdout final
    print("\n[5] Holdout final (una sola evaluación, modelo y umbral congelados)")
    best.fit(X_train, y_train)
    proba_test = best.predict_proba(X_test)[:, 1]

    libres = E.threshold_free_metrics(y_test, proba_test)
    en_umbral = E.metrics_at_threshold(y_test, proba_test, umbral)
    final = pd.Series({**libres, **en_umbral, "modelo": ganador})
    print(final.to_string())
    _save(final.to_frame("valor"), "07_holdout")

    deciles_test = E.decile_table(y_test, proba_test)
    print("\n  Deciles de riesgo (holdout):")
    print(deciles_test.to_string())
    _save(deciles_test, "08_deciles_holdout")

    ganancias = E.gains_curve(y_test, proba_test)
    _save(ganancias.round(5), "09_curva_ganancias")

    # ------------------------------------------------------- 6. segmentación
    print("\n[6] Segmentación de la cohorte de alto riesgo")
    cohorte = test.copy()
    cohorte["score"] = proba_test
    cohorte = cohorte[cohorte["score"] >= umbral].copy()
    print(f"  Cohorte: {len(cohorte)} clientes ({len(cohorte) / len(test):.1%} del holdout), "
          f"tasa de abandono {cohorte[C.TARGET].mean():.3f}")

    Z, _ = S.build_cluster_matrix(cohorte, best, n_components=0.90)
    k, diag = S.choose_k(Z)
    print(f"\n  Diagnóstico de K:\n{diag.to_string(index=False)}")
    print(f"\n  K elegido: {k}")
    _save(diag.set_index("k"), "10_diagnostico_k")

    labels = S.fit_kmeans(Z, k)
    perfil = S.profile_clusters(cohorte, labels)
    print(f"\n  Perfil de segmentos:\n{perfil.to_string()}")
    _save(perfil, "11_perfil_segmentos")
    _save(S.categorical_profile(cohorte, labels), "12_perfil_categorico")

    # ------------------------------------------------------------- 7. scoring
    print("\n[7] Exportando scoring")
    scoring = test.copy()
    scoring["score"] = proba_test
    scoring["decil"] = pd.qcut(
        pd.Series(proba_test).rank(method="first", ascending=False),
        q=10,
        labels=range(1, 11),
    ).astype(int)
    scoring["priorizado"] = (proba_test >= umbral).astype(int)
    scoring["segmento"] = np.nan
    scoring.loc[cohorte.index, "segmento"] = labels

    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    destino = C.DATA_PROCESSED / "scoring_holdout.csv"
    scoring.to_csv(destino, index=False)
    print(f"  -> {destino.relative_to(C.ROOT)}")

    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, C.MODELS_DIR / "modelo_final.joblib")
    print(f"  -> models/modelo_final.joblib")

    (RESULTS / "modelo_congelado.json").write_text(
        json.dumps(
            {
                "modelo": ganador,
                "umbral": round(umbral, 6),
                "criterio_umbral": C.THRESHOLD_POLICY,
                "techo_capacidad": C.CAPACITY,
                "seed": C.SEED,
                "k_segmentos": k,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("\nListo.")


if __name__ == "__main__":
    main()
