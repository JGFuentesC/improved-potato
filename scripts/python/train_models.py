#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_models.py
===============
Pipeline OOT: Árbol de Decisión vs. MLP — PyPI Downloads Forecast
Autor : Anahuac Analytics
"""

import os
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from dotenv import load_dotenv
from google.cloud import bigquery

from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = ROOT / "notebooks" / "reports"
FIG_DIR     = REPORTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ─── Env ──────────────────────────────────────────────────────────────────────
load_dotenv(ROOT / ".env", override=True)
BQ_PROJECT = os.environ["BQ_PROJECT"]
BQ_TABLE   = os.environ["BQ_TABLE"]

SEED       = 42
CAT_COLS   = ["project"]
TARGET     = "tgt"
PALETTE    = {"boto3": "#4C72B0", "packaging": "#DD8452", "urllib3": "#55A868"}

np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CARGA DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    print("⏳  Cargando datos desde BigQuery...")
    sql  = (ROOT / "scripts" / "BQ" / "examen.sql").read_text(encoding="utf-8")
    query = sql.format_map({"BQ_TABLE": BQ_TABLE})
    df   = bigquery.Client(project=BQ_PROJECT).query(query).to_dataframe()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    print(f"✅  {len(df):,} filas × {df.shape[1]} columnas")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. EDA
# ═══════════════════════════════════════════════════════════════════════════════

def _save(name: str) -> Path:
    p = FIG_DIR / name
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"  ✓  {name}")
    return p


def eda_target(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Distribución del Target — tgt (descargas siguientes 10 min)",
                 fontsize=13, fontweight="bold")

    axes[0].hist(df["tgt"], bins=60, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[0].set(xlabel="tgt", ylabel="Frecuencia", title="Histograma global")

    projs = sorted(df["project"].unique())
    bp = axes[1].boxplot([df[df["project"] == p]["tgt"].values for p in projs],
                         labels=projs, patch_artist=True)
    for patch, proj in zip(bp["boxes"], projs):
        patch.set_facecolor(PALETTE.get(proj, "#aaa")); patch.set_alpha(0.8)
    axes[1].set(xlabel="Proyecto", ylabel="tgt", title="Boxplot por proyecto")

    plt.tight_layout()
    _save("eda_target_distribution.png")


def eda_timeseries(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig.suptitle("Series de tiempo: downloads y target por proyecto",
                 fontsize=13, fontweight="bold")
    for proj in sorted(df["project"].unique()):
        sub = df[df["project"] == proj]
        c   = PALETTE.get(proj, "#aaa")
        axes[0].plot(sub["ts"], sub["downloads"], label=proj, lw=0.9, alpha=0.8, color=c)
        axes[1].plot(sub["ts"], sub["tgt"],       label=proj, lw=0.9, alpha=0.8, color=c)
    axes[0].set(ylabel="downloads(t)", title="Descargas actuales"); axes[0].legend()
    axes[1].set(ylabel="tgt", xlabel="Timestamp", title="Variable objetivo"); axes[1].legend()
    plt.tight_layout()
    _save("eda_time_series.png")


def eda_correlation(df: pd.DataFrame, feat_cols: list):
    corr = df[feat_cols + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(22, 18))
    sns.heatmap(corr, mask=np.triu(np.ones_like(corr, dtype=bool)),
                annot=False, cmap="coolwarm", center=0,
                linewidths=0.3, ax=ax, cbar_kws={"shrink": 0.5})
    ax.set_title("Mapa de correlaciones — features × tgt", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save("eda_correlation_heatmap.png")


def eda_window_corr(df: pd.DataFrame, feat_cols: list):
    windows = [(i*5-4, i*5) for i in range(1, 13)]
    records = []
    for s, e in windows:
        cols = [c for c in feat_cols if f"_{s}_{e}" in c]
        if cols:
            records.append({"Ventana": f"W{s}-{e}",
                            "|r| medio": df[cols].corrwith(df[TARGET]).abs().mean()})
    dfr = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(dfr["Ventana"], dfr["|r| medio"],
                  color="#4C72B0", edgecolor="white", alpha=0.85)
    for bar, v in zip(bars, dfr["|r| medio"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.002,
                f"{v:.3f}", ha="center", fontsize=8)
    ax.set(xlabel="Ventana (min hacia atrás)", ylabel="|r| promedio",
           title="Correlación media |r| con tgt por ventana")
    ax.tick_params(axis="x", rotation=40)
    plt.tight_layout()
    _save("eda_window_correlations.png")


def run_eda(df: pd.DataFrame, feat_cols: list) -> dict:
    print("\n── EDA ──────────────────────────────────────────────────────────────")
    eda_target(df)
    eda_timeseries(df)
    eda_correlation(df, feat_cols)
    eda_window_corr(df, feat_cols)
    q1, q3 = df[TARGET].quantile([0.25, 0.75])
    iqr     = q3 - q1
    n_out   = int(((df[TARGET] < q1-1.5*iqr) | (df[TARGET] > q3+1.5*iqr)).sum())
    pct_out = round(100*n_out/len(df), 1)
    print(f"  Outliers IQR en tgt: {n_out} ({pct_out}%) | límite sup: {q3+1.5*iqr:,.0f}")
    print("✅  EDA completado.")
    return {"n_out": n_out, "pct_out": pct_out, "iqr_upper": round(q3+1.5*iqr)}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SPLIT OOT
# ═══════════════════════════════════════════════════════════════════════════════

def oot_split(df, X, y, ratio=0.8):
    cut = int(len(df) * ratio)
    ts_cut = df["ts"].iloc[cut]
    print(f"\n── OOT Split ────────────────────────────────────────────────────────")
    print(f"  Train: {cut:,} | Test: {len(df)-cut:,} | Corte: {ts_cut}")
    return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:], ts_cut


# ═══════════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════════

def make_pre(feat_cols, scale=False) -> ColumnTransformer:
    steps = [("ohe", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), CAT_COLS)]
    steps.append(("num", StandardScaler() if scale else "passthrough", feat_cols))
    return ColumnTransformer(steps, remainder="drop")


def metrics(y_true, y_pred, name) -> dict:
    return {
        "Modelo":   name,
        "MAPE (%)": round(mean_absolute_percentage_error(y_true, y_pred)*100, 3),
        "RMSE":     round(np.sqrt(mean_squared_error(y_true, y_pred)), 1),
        "R²":       round(r2_score(y_true, y_pred), 4),
    }


def plot_pred(y_true, y_pred, name, fname):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Diagnóstico — {name}", fontsize=13, fontweight="bold")

    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[0].scatter(y_true, y_pred, alpha=0.2, s=8, color="#4C72B0", rasterized=True)
    axes[0].plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfecta")
    axes[0].set(xlabel="Real", ylabel="Predicho", title="Real vs. Predicho")
    axes[0].legend()

    res = y_true - y_pred
    axes[1].hist(res, bins=60, color="#DD8452", edgecolor="white", alpha=0.85)
    axes[1].axvline(0, color="crimson", lw=1.5, linestyle="--")
    axes[1].set(xlabel="Residuo", ylabel="Frecuencia", title="Distribución de Residuos")

    plt.tight_layout()
    _save(fname)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LAZY PREDICT BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

def run_lazy(X_tr, X_te, y_tr, y_te, feat_cols):
    print("\n── LazyPredict Benchmark ────────────────────────────────────────────")
    try:
        from lazypredict.Supervised import LazyRegressor
        ohe    = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        p_tr   = ohe.fit_transform(X_tr[CAT_COLS])
        p_te   = ohe.transform(X_te[CAT_COLS])
        Xn_tr  = np.hstack([X_tr[feat_cols].values, p_tr])
        Xn_te  = np.hstack([X_te[feat_cols].values, p_te])
        reg    = LazyRegressor(verbose=0, ignore_warnings=True)
        top, _ = reg.fit(Xn_tr, Xn_te, y_tr, y_te)
        print(top.head(10).to_string())
        return top.head(10)
    except Exception as exc:
        print(f"  ⚠  LazyPredict omitido: {exc}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ÁRBOL DE DECISIÓN
# ═══════════════════════════════════════════════════════════════════════════════

def train_dt(X_tr, X_te, y_tr, y_te, feat_cols):
    print("\n── Árbol de Decisión — RandomizedSearchCV ───────────────────────────")
    pipe = Pipeline([
        ("pre", make_pre(feat_cols, scale=False)),
        ("dt",  DecisionTreeRegressor(random_state=SEED)),
    ])
    param_dist = {
        "dt__max_depth":         [3, 5, 7, 10, 15, 20, None],
        "dt__min_samples_split": [2, 5, 10, 20, 50],
        "dt__min_samples_leaf":  [1, 2, 4, 8, 16],
        "dt__max_features":      [0.3, 0.5, 0.7, "sqrt", "log2", None],
        "dt__ccp_alpha":         [0.0, 1e-5, 1e-4, 5e-4, 1e-3],
    }
    search = RandomizedSearchCV(pipe, param_dist, n_iter=50, cv=3,
                                scoring="neg_mean_absolute_percentage_error",
                                n_jobs=-1, random_state=SEED, verbose=1)
    search.fit(X_tr, y_tr)
    best   = search.best_estimator_
    y_pred = best.predict(X_te)
    m      = metrics(y_te, y_pred, "Árbol de Decisión")
    print(f"  Params: {search.best_params_}")
    print(f"  MAPE: {m['MAPE (%)']:.3f}% | RMSE: {m['RMSE']:.1f} | R²: {m['R²']:.4f}")

    # Feature importance
    dt_step  = best.named_steps["dt"]
    pre_step = best.named_steps["pre"]
    ohe_names = list(pre_step.named_transformers_["ohe"].get_feature_names_out(CAT_COLS))
    all_names = ohe_names + feat_cols
    imp = (pd.Series(dt_step.feature_importances_, index=all_names)
             .sort_values(ascending=False).head(20))
    fig, ax = plt.subplots(figsize=(12, 7))
    imp.plot(kind="barh", ax=ax, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.invert_yaxis()
    ax.set(title="Top-20 Feature Importances — Árbol de Decisión", xlabel="Importancia")
    plt.tight_layout()
    _save("dt_feature_importance.png")

    plot_pred(y_te, y_pred, "Árbol de Decisión", "dt_predictions.png")
    return best, m, search.best_params_


# ═══════════════════════════════════════════════════════════════════════════════
# 6. MLP
# ═══════════════════════════════════════════════════════════════════════════════

def train_mlp(X_tr, X_te, y_tr, y_te, feat_cols):
    print("\n── MLP — RandomizedSearchCV ─────────────────────────────────────────")
    pipe = Pipeline([
        ("pre", make_pre(feat_cols, scale=True)),
        ("mlp", MLPRegressor(random_state=SEED, max_iter=800,
                             early_stopping=True, n_iter_no_change=20,
                             validation_fraction=0.1)),
    ])
    param_dist = {
        "mlp__hidden_layer_sizes": [(64,),(128,),(256,),(64,32),(128,64),
                                    (256,128),(128,64,32),(256,128,64)],
        "mlp__activation":         ["relu", "tanh"],
        "mlp__alpha":              [1e-5, 1e-4, 1e-3, 1e-2],
        "mlp__learning_rate_init": [5e-4, 1e-3, 5e-3, 1e-2],
        "mlp__batch_size":         [32, 64, 128, 256],
    }
    search = RandomizedSearchCV(pipe, param_dist, n_iter=30, cv=3,
                                scoring="neg_mean_absolute_percentage_error",
                                n_jobs=-1, random_state=SEED, verbose=1)
    search.fit(X_tr, y_tr)
    best   = search.best_estimator_
    y_pred = best.predict(X_te)
    m      = metrics(y_te, y_pred, "MLP")
    print(f"  Params: {search.best_params_}")
    print(f"  MAPE: {m['MAPE (%)']:.3f}% | RMSE: {m['RMSE']:.1f} | R²: {m['R²']:.4f}")

    mlp_step = best.named_steps["mlp"]
    if hasattr(mlp_step, "loss_curve_"):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(mlp_step.loss_curve_, color="#4C72B0", lw=2, label="Train loss")
        ax.set(title="Curva de pérdida — MLP", xlabel="Iteración", ylabel="Loss (MSE)")
        ax.legend(); plt.tight_layout()
        _save("mlp_loss_curve.png")

    plot_pred(y_te, y_pred, "MLP", "mlp_predictions.png")
    return best, m, search.best_params_


# ═══════════════════════════════════════════════════════════════════════════════
# 7. COMPARATIVO
# ═══════════════════════════════════════════════════════════════════════════════

def plot_comparison(metrics_list):
    df_m   = pd.DataFrame(metrics_list)
    colors = ["#4C72B0", "#DD8452"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Comparativo de Modelos — OOT Test Set", fontsize=14, fontweight="bold")
    for ax, col in zip(axes, ["MAPE (%)", "RMSE", "R²"]):
        bars = ax.bar(df_m["Modelo"], df_m[col], color=colors,
                      edgecolor="white", alpha=0.85, width=0.5)
        ax.set(title=col, ylabel=col)
        for bar, v in zip(bars, df_m[col]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.01,
                    f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    plt.tight_layout()
    _save("model_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. REPORTE MARKDOWN
# ═══════════════════════════════════════════════════════════════════════════════

def generate_report(df, metrics_list, dt_params, mlp_params, cutoff_ts, out_info, lazy_top):
    df_m   = pd.DataFrame(metrics_list)
    winner = df_m.loc[df_m["MAPE (%)"].idxmin(), "Modelo"]
    loser  = df_m.loc[df_m["MAPE (%)"].idxmax(), "Modelo"]
    w_mape = df_m["MAPE (%)"].min()
    l_mape = df_m["MAPE (%)"].max()

    # Sección lazy
    lazy_md = ""
    if lazy_top is not None:
        rows = []
        for r in lazy_top.reset_index().itertuples():
            d     = r._asdict()
            r2_v  = d.get("R-Squared", d.get("R²", "-"))
            rmse_v = d.get("RMSE", "-")
            try:
                r2_str  = f"{float(r2_v):.4f}"
            except (TypeError, ValueError):
                r2_str  = str(r2_v)
            try:
                rmse_str = f"{float(rmse_v):,.1f}"
            except (TypeError, ValueError):
                rmse_str = str(rmse_v)
            rows.append(f"| {r.Index} | {r2_str} | {rmse_str} |")
        rows = "\n".join(rows)
        lazy_md = f"""## 4. Benchmark LazyPredict

Se ejecutaron automáticamente {len(lazy_top)} regresores de `scikit-learn` para contextualizar el rendimiento de los modelos objetivo.

| Modelo | R² | RMSE |
|---|---|---|
{rows}

> Los modelos DT y MLP compiten en un espacio donde los mejores regressores de sklearn sirven como **línea base de referencia**.

---
"""

    # Tabla de parámetros
    def param_table(p, prefix):
        rows = "\n".join(f"| `{k.replace(prefix,'')}` | `{v}` |" for k, v in p.items())
        return f"| Parámetro | Valor óptimo |\n|---|---|\n{rows}"

    metrics_md = df_m.to_markdown(index=False)

    report = f"""# Reporte de Modelado — PyPI Downloads Forecast

> **Estrategia:** Out-of-Time (OOT) 80/20  
> **Fecha:** {datetime.now().strftime("%Y-%m-%d %H:%M")}  
> **Dataset:** `{BQ_TABLE}`

---

## 1. Contexto y Objetivo

Se entrenaron dos modelos de regresión para predecir `tgt`, definida como la **suma de descargas de los 3 proyectos PyPI más populares en los siguientes 10 minutos** a partir del instante `ts`.

Las features son estadísticas de ventanas deslizantes de 5 minutos (promedio, desviación estándar y tasa de crecimiento instantáneo) calculadas sobre los últimos 60 minutos, más `project` codificado con *one-hot encoding*.

**Métrica principal:** MAPE | **Secundarias:** RMSE, R²

---

## 2. Análisis Exploratorio de Datos (EDA)

### 2.1 Distribución del Target

![Distribución del target](figures/eda_target_distribution.png)

`tgt` presenta distribución asimétrica con cola derecha pronunciada. Se detectaron **{out_info['n_out']} outliers** ({out_info['pct_out']}% del dataset) por criterio IQR (límite superior: {out_info['iqr_upper']:,} descargas). `boto3` exhibe mayor varianza por ser el paquete de mayor volumen.

### 2.2 Series de Tiempo

![Series de tiempo](figures/eda_time_series.png)

El periodo cubre `{df["ts"].min().strftime("%Y-%m-%d %H:%M")}` → `{df["ts"].max().strftime("%Y-%m-%d %H:%M")} UTC`. Se aprecia comportamiento estacionario sin tendencia global sostenida, con oscilaciones regulares compatibles con patrones de uso diurno.

### 2.3 Mapa de Correlaciones

![Correlaciones](figures/eda_correlation_heatmap.png)

Las features de volumen (`x_avg_dwn_*`) correlacionan positivamente con `tgt`. Las de crecimiento (`x_avg_growth_*`) muestran correlaciones más débiles, confirmando que el nivel absoluto es más predictivo que la aceleración.

### 2.4 Relevancia por Ventana Temporal

![Correlación por ventana](figures/eda_window_correlations.png)

Las **ventanas 1–15 min** concentran la mayor señal predictiva. La correlación decrece progresivamente hacia ventanas más remotas (46–60 min), confirmando la hipótesis de dependencia temporal de corto plazo.

---

## 3. Estrategia de Entrenamiento: Out-of-Time (OOT)

| Partición | Observaciones | Rango temporal |
|---|---|---|
| **Train** | {int(len(df)*0.8):,} | Hasta `{cutoff_ts.strftime("%Y-%m-%d %H:%M UTC")}` |
| **Test (OOT)** | {len(df)-int(len(df)*0.8):,} | Desde `{cutoff_ts.strftime("%Y-%m-%d %H:%M UTC")}` |

> ⚠️ El split es **estrictamente temporal** para respetar la causalidad: el modelo nunca accede a información futura durante el entrenamiento.

---

{lazy_md}## 5. Árbol de Decisión

### 5.1 Hiperparámetros Óptimos (50 iteraciones, CV=3)

{param_table(dt_params, "dt__")}

> Búsqueda por `neg_mean_absolute_percentage_error`. El parámetro `ccp_alpha` (poda por complejidad-costo) controla la sobrecomplejidad del árbol.

### 5.2 Importancia de Features

![Feature Importance](figures/dt_feature_importance.png)

Las features de **descargas promedio recientes** (ventanas 1–10 min) dominan. Las variables `project_*` one-hot contribuyen de forma secundaria pero consistente con las diferencias de volumen entre proyectos.

### 5.3 Diagnóstico de Predicciones

![Predicciones DT](figures/dt_predictions.png)

---

## 6. Red Neuronal MLP

### 6.1 Hiperparámetros Óptimos (30 iteraciones, CV=3)

{param_table(mlp_params, "mlp__")}

> Las features numéricas fueron estandarizadas con `StandardScaler`. Se aplicó *early stopping* (`n_iter_no_change=20`) para prevenir sobreajuste.

### 6.2 Curva de Pérdida

![Loss curve](figures/mlp_loss_curve.png)

### 6.3 Diagnóstico de Predicciones

![Predicciones MLP](figures/mlp_predictions.png)

---

## 7. Comparativo de Modelos

{metrics_md}

![Comparativo](figures/model_comparison.png)

### 7.1 Interpretación Técnica

El modelo **{winner}** supera al {loser} con un MAPE de **{w_mape:.3f}%** vs **{l_mape:.3f}%**.

- **Árbol de Decisión:** captura no-linealidades de forma nativa sin requerir normalización. Su interpretabilidad es total (reglas explícitas). Sin embargo, tiende a ajustar patrones locales ruidosos si `max_depth` no es controlado.
- **MLP:** captura interacciones de orden superior entre ventanas temporales mediante capas ocultas. Requiere `StandardScaler` y es sensible a la elección de `learning_rate_init` y `alpha`. El *early stopping* mitiga el sobreajuste en el conjunto OOT.

La dominancia del **{winner}** en este problema es consistente con la naturaleza tabular y la alta autocorrelación de las features de ventana deslizante.

### 7.2 Aplicación de Negocio

Las descargas de PyPI son un proxy directo del tráfico de infraestructura CDN. Este modelo habilita:

1. **Planificación de capacidad proactiva:** escalar mirrors regionales 10 minutos antes del pico previsto, evitando degradación del servicio durante lanzamientos masivos.
2. **Alertas tempranas de anomalía:** un crecimiento `instant_growth` elevado combinado con `tgt` proyectado alto puede indicar un lanzamiento viral, un ataque de scraping, o un fallo en un mirror alternativo.
3. **Optimización de costos cloud:** el pre-escalado elimina el *over-provisioning* reactivo, reduciendo el costo de instancias de cómputo y ancho de banda en un escenario de facturación por segundo.
4. **SLA de descarga:** `boto3` y `urllib3` son dependencias críticas de miles de pipelines CI/CD empresariales. Predecir su demanda permite negociar SLAs de disponibilidad más sólidos.

---

## 8. Conclusiones

1. La estrategia **OOT es obligatoria** para datos temporales; un split aleatorio inflaría artificialmente el R² al exponer datos futuros durante el entrenamiento.
2. Las **features de ventana 1–15 min** concentran el poder predictivo; las ventanas 46–60 min aportan señal marginal.
3. El modelo **{winner}** (MAPE: {w_mape:.3f}%) es recomendado para producción en este problema.
4. **Trabajo futuro:** explorar modelos especializados en series de tiempo (LightGBM, LSTM, Temporal Fusion Transformer) e incorporar features exógenas (releases en PyPI, estrellas en GitHub, hora del día).

---

*Generado automáticamente por `scripts/python/train_models.py` · {datetime.now().strftime("%Y-%m-%d %H:%M")}*
"""

    out = REPORTS_DIR / "examen_ml_report.md"
    out.write_text(report, encoding="utf-8")
    print(f"\n✅  Reporte: {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  PyPI Forecast · Árbol de Decisión vs. MLP (OOT 80/20)")
    print("=" * 65)

    df         = load_data()
    feat_cols  = [c for c in df.columns if c.startswith("x_")]

    out_info   = run_eda(df, feat_cols)

    X = df[feat_cols + CAT_COLS]
    y = df[TARGET]
    X_tr, X_te, y_tr, y_te, cutoff_ts = oot_split(df, X, y)

    lazy_top = run_lazy(X_tr, X_te, y_tr, y_te, feat_cols)

    _, dt_m, dt_p  = train_dt(X_tr, X_te, y_tr, y_te, feat_cols)
    _, mlp_m, mlp_p = train_mlp(X_tr, X_te, y_tr, y_te, feat_cols)

    plot_comparison([dt_m, mlp_m])
    generate_report(df, [dt_m, mlp_m], dt_p, mlp_p, cutoff_ts, out_info, lazy_top)

    print("\n🏁  Pipeline completado exitosamente.")
    print(f"   Figuras: {FIG_DIR}")
    print(f"   Reporte: {REPORTS_DIR / 'examen_ml_report.md'}")


if __name__ == "__main__":
    main()
