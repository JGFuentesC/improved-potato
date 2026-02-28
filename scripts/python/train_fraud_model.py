import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_curve, auc, classification_report

# ==========================================
# CONFIGURACIÓN
# ==========================================
DATA_PATH = 'data/fraud_sample.csv'
INF_PATH = 'data/inferencia.csv'
ASSETS_DIR = 'model_assets'
MODEL_NAME = 'DecisionTree_Fraud_v1'

def train_model():
    print(f"--- Iniciando entrenamiento: {MODEL_NAME} ---")
    os.makedirs(ASSETS_DIR, exist_ok=True)
    
    # 1. Carga de datos
    df = pd.read_csv(DATA_PATH)
    discrete_cols = [col for col in df.columns if col.startswith('d_')]
    
    # Encoding consistente
    df_encoded = pd.get_dummies(df, columns=discrete_cols, drop_first=True)
    X = df_encoded.drop('target', axis=1)
    y = df_encoded['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 4. Hiperparametrización (GridSearch con CV=5)
    # Reducimos un poco el espacio para que no tarde demasiado pero mantenga calidad
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [4, 6, 8], # Profundidad controlada para visualización clara
        'min_samples_leaf': [20, 50],
        'class_weight': ['balanced']
    }
    
    base_clf = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(base_clf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_clf = grid_search.best_estimator_

    # ==========================================
    # 5. CURVA ROC CON IDENTIFICADOR CLARO
    # ==========================================
    y_test_prob = best_clf.predict_proba(X_test)[:, 1]
    y_train_prob = best_clf.predict_proba(X_train)[:, 1]
    
    fpr_t, tpr_t, _ = roc_curve(y_train, y_train_prob)
    fpr_v, tpr_v, _ = roc_curve(y_test, y_test_prob)
    
    plt.figure(figsize=(10, 8), facecolor='#f0f0f0')
    plt.plot(fpr_t, tpr_t, color='#1f77b4', lw=3, label=f'TRAIN AUC: {auc(fpr_t, tpr_t):.4f}')
    plt.plot(fpr_v, tpr_v, color='#d62728', lw=3, label=f'VALID AUC: {auc(fpr_v, tpr_v):.4f}')
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', alpha=0.5)
    
    plt.title(f'PERFORMANCE METRICS\nMODEL ID: {MODEL_NAME}', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(loc="lower right", fontsize=12, frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    roc_path = os.path.join(ASSETS_DIR, f'{MODEL_NAME}_ROC.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC guardada: {roc_path}")

    # ==========================================
    # 6. VISUALIZACIÓN DETALLADA DEL ÁRBOL
    # ==========================================
    plt.figure(figsize=(24, 12))
    plot_tree(best_clf, 
              feature_names=X.columns.tolist(),
              class_names=['No Fraud', 'Fraud'],
              filled=True, 
              rounded=True, 
              fontsize=10,
              proportion=True,
              precision=2)
    plt.title(f'Decision Tree Structure - {MODEL_NAME}', fontsize=20, fontweight='bold')
    tree_path = os.path.join(ASSETS_DIR, 'decision_tree_detail.png')
    plt.savefig(tree_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Estructura del árbol guardada: {tree_path}")

    # ==========================================
    # 7. STACKED BAR 100% (INFERENCIA)
    # ==========================================
    print("Iniciando calificación de inferencia...")
    df_inf = pd.read_csv(INF_PATH)
    # Asegurar que inferencia tenga las mismas columnas que el entrenamiento
    df_inf_enc = pd.get_dummies(df_inf, columns=discrete_cols)
    # Reindexar para asegurar match de columnas
    df_inf_enc = df_inf_enc.reindex(columns=df_encoded.columns, fill_value=0)
    
    X_inf = df_inf_enc.drop('target', axis=1)
    y_inf = df_inf_enc['target']
    
    # Obtener el ID del nodo terminal para cada fila
    leaf_ids = best_clf.apply(X_inf)
    df_results = pd.DataFrame({'node': leaf_ids, 'target': y_inf})
    
    # Calcular proporciones por nodo
    node_stats = df_results.groupby('node')['target'].value_counts(normalize=True).unstack().fillna(0)
    # Asegurar que existen columnas 0 y 1
    if 1 not in node_stats.columns: node_stats[1] = 0
    if 0 not in node_stats.columns: node_stats[0] = 0
    
    # ORDENAR POR PROPORCIÓN DE MALOS (target=1) descendente
    node_stats = node_stats.sort_values(by=1, ascending=False)
    
    # Gráfica Premium Stacked Bar 100%
    ax = node_stats[[1, 0]].plot(kind='bar', stacked=True, figsize=(14, 7), 
                                 color=['#d62728', '#2ca02c'], width=0.8)
    
    plt.title(f'NODE CALIBRATION - FRAUD PROPORTION\nModel: {MODEL_NAME}', fontsize=16, fontweight='bold')
    plt.xlabel('Terminal Node ID (Sorted by Risk)', fontsize=12)
    plt.ylabel('Event Proportion (100%)', fontsize=12)
    plt.legend(['Fraud (1)', 'No Fraud (0)'], loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Formato de porcentaje en Y
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    bar_path = os.path.join(ASSETS_DIR, 'node_risk_distribution.png')
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfica de nodos guardada: {bar_path}")

    # Guardar modelo
    with open(os.path.join(ASSETS_DIR, f'{MODEL_NAME}.pkl'), 'wb') as f:
        pickle.dump(best_clf, f)

if __name__ == "__main__":
    train_model()
