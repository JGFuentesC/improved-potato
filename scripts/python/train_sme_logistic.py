import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
)

# --- Configuración Estética Premium (Midnight Gold) ---
sns.set_theme(style="dark")
plt.rcParams["figure.dpi"] = 120
BKG_COLOR = "#0D1117"  # Deep Charcoal
TRAIN_COLOR = "#F2C94C"  # Gold
TEST_COLOR = "#2D9CDB"  # Elegant Blue
GRID_COLOR = "#30363D"
TEXT_COLOR = "#C9D1D9"


def train_sme_model():
    print(
        "🚀 Iniciando proceso de entrenamiento: Regresión Logística (SMEs) con Optimización..."
    )

    # 1. Carga de datos
    data_path = "data/sme_mx.csv"
    if not os.path.exists(data_path):
        print(f"❌ Error: No se encontró el archivo {data_path}")
        return

    df = pd.read_csv(data_path)
    print(f"📦 Registros cargados: {len(df)}")

    # 2. Preprocesamiento (Siguiendo patrones del notebook)
    df_filtered = df[df["altaSAT"] != "o"].copy()
    df_filtered["target"] = df_filtered["altaSAT"].map({"s": 1, "n": 0})

    cols_to_keep = [
        "edadEmprendedor",
        "sexoEmprendedor",
        "escolaridadEmprendedor",
        "dependientesEconomicos",
        "estadoCivil",
        "familiaAyuda",
        "antiguedadNegocio",
        "giroNegocio",
        "numEmpleados",
        "ventasPromedioDiarias",
        "registroVentas",
        "registroContabilidad",
        "usaCredito",
        "tiempoCreditoProveedores",
        "target",
    ]

    df_model = df_filtered[cols_to_keep].dropna().copy()
    print(f"✅ Registros tras limpieza y filtrado: {len(df_model)}")

    # 3. Preparación de X e y
    X = df_model.drop(columns=["target"])
    y = df_model["target"]

    # Codificación de variables (One-Hot Encoding)
    cat_cols = X.select_dtypes(include=["object"]).columns
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # 4. Partición de datos (70/30)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.3, random_state=42, stratify=y
    )

    # 5. Estandarización
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6. Híper-parametrización y Selección (Lasso)
    print("🧠 Optimizando hiperparámetros con GridSearchCV (L1/Lasso)...")
    param_grid = {"C": np.logspace(-3, 2, 12)}

    base_model = LogisticRegression(penalty="l1", solver="liblinear", random_state=42)
    grid_search = GridSearchCV(
        base_model, param_grid, cv=5, scoring="roc_auc", n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)

    model = grid_search.best_estimator_
    print(
        f"✨ Mejor C: {grid_search.best_params_['C']:.5f} (AUC CV: {grid_search.best_score_:.4f})"
    )

    # 7. Evaluación
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "auc": auc(*roc_curve(y_test, y_proba)[:2]),
    }

    print(f"📈 Métricas de Validación:")
    print(f"   AUC: {metrics['auc']:.4f}")
    print(f"   F1:  {metrics['f1']:.4f}")
    print(f"   Acc: {metrics['accuracy']:.4f}")

    # 8. Gráficas de Alta Estética
    os.makedirs("model_assets", exist_ok=True)

    # --- Curva ROC ---
    y_proba_train = model.predict_proba(X_train_scaled)[:, 1]
    fpr_train, tpr_train, _ = roc_curve(y_train, y_proba_train)
    auc_train = auc(fpr_train, tpr_train)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_proba)

    plt.figure(figsize=(10, 8), facecolor=BKG_COLOR)
    ax = plt.gca()
    ax.set_facecolor(BKG_COLOR)

    # Usando 4 decimales para coincidir con el reporte de texto
    plt.plot(
        fpr_train,
        tpr_train,
        color=TRAIN_COLOR,
        lw=2.5,
        label=f"Train ROC (AUC = {auc_train:.4f})",
        alpha=0.9,
    )
    plt.plot(
        fpr_test,
        tpr_test,
        color=TEST_COLOR,
        lw=4,
        label=f'Test ROC (AUC = {metrics["auc"]:.4f})',
    )
    plt.plot([0, 1], [0, 1], color=GRID_COLOR, lw=1.5, linestyle=":", alpha=0.8)
    plt.fill_between(fpr_test, tpr_test, alpha=0.1, color=TEST_COLOR)

    plt.xlabel("False Positive Rate", fontsize=12, color=TEXT_COLOR, labelpad=10)
    plt.ylabel("True Positive Rate", fontsize=12, color=TEXT_COLOR, labelpad=10)
    plt.title(
        "Performance Analysis: ROC Curves",
        color="white",
        fontsize=18,
        fontweight="bold",
        pad=30,
    )

    legend = plt.legend(loc="lower right", facecolor=BKG_COLOR, edgecolor=GRID_COLOR)
    plt.setp(legend.get_texts(), color=TEXT_COLOR)
    plt.tick_params(colors=TEXT_COLOR, labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COLOR)
    plt.grid(color=GRID_COLOR, linestyle="-", alpha=0.4)
    plt.savefig(
        "model_assets/sme_logistic_roc.png", bbox_inches="tight", facecolor=BKG_COLOR
    )
    plt.close()

    # --- Matriz de Confusión ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6), facecolor=BKG_COLOR)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True),
        cbar=False,
        annot_kws={"size": 16, "weight": "bold", "color": "white"},
    )
    plt.title(
        "Confusion Matrix: Predictive Accuracy",
        fontsize=16,
        fontweight="bold",
        pad=25,
        color="white",
    )
    plt.xlabel("Predicted", fontsize=12, color=TEXT_COLOR)
    plt.ylabel("Actual", fontsize=12, color=TEXT_COLOR)
    plt.tick_params(colors=TEXT_COLOR)
    plt.savefig(
        "model_assets/sme_logistic_cm.png", bbox_inches="tight", facecolor=BKG_COLOR
    )
    plt.close()

    # 9. Guardar Artefactos
    with open("model_assets/sme_logistic_v1.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("model_assets/sme_scaler_v1.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("model_assets/feature_names.txt", "w") as f:
        for col in X_encoded.columns:
            f.write(f"{col}\n")

    print("✨ Modelo y activos guardados en 'model_assets/'.")
    return metrics, model, X_encoded.columns


if __name__ == "__main__":
    train_sme_model()
