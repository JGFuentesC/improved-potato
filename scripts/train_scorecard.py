# /// script
# dependencies = [
#   "pandas",
#   "scikit-learn",
#   "numpy",
#   "scipy",
#   "matplotlib",
#   "seaborn"
# ]
# ///

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os

def run_scoring_model():
    # 1. Cargar datos
    data_path = 'data/sme_mx.csv'
    if not os.path.exists(data_path):
        print(f"Error: No se encuentra el archivo en {data_path}")
        return

    df = pd.read_csv(data_path)
    
    # 2. Filtrado y Target
    # Eliminar altaSAT = 'o'
    df = df[df['altaSAT'] != 'o'].copy()
    
    # Target: altaSAT = 'n' (Evento/Bad)
    df['target'] = (df['altaSAT'] == 'n').astype(int)
    
    # 3. Limpieza de columnas
    # Identificar columnas a eliminar
    cols_to_drop = ['id', 'latitud', 'longitud', 'Respuesta Original', 'CategoríaDeseo', 'altaSAT']
    # Eliminar columnas vacías o con nombres extraños
    for col in df.columns:
        if 'Unnamed' in col or col == '':
            cols_to_drop.append(col)
            
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # 4. Binning Uniforme y WoE
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'target' in numeric_cols: numeric_cols.remove('target')
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    bins_data = {}
    
    def calculate_woe_iv(df, feature, target):
        tmp = df[[feature, target]].copy()
        # Agrupar por valor, incluyendo nulos si los hay
        stats = tmp.groupby(feature, dropna=False)[target].agg(['count', 'sum'])
        stats.columns = ['Total', 'Bad']
        stats['Good'] = stats['Total'] - stats['Bad']
        
        total_good = stats['Good'].sum()
        total_bad = stats['Bad'].sum()
        
        # Ajuste para evitar log(0)
        stats['Share_Good'] = (stats['Good'] + 0.5) / total_good
        stats['Share_Bad'] = (stats['Bad'] + 0.5) / total_bad
        
        stats['WoE'] = np.log(stats['Share_Good'] / stats['Share_Bad'])
        stats['IV'] = (stats['Share_Good'] - stats['Share_Bad']) * stats['WoE']
        
        return stats

    # Aplicar Binning Uniforme a numéricas
    for col in numeric_cols:
        # Si tiene pocos valores únicos, tratar como categórica
        if df[col].nunique() > 10:
            df[col + '_bin'] = pd.cut(df[col], bins=5, labels=False, duplicates='drop').astype(str)
        else:
            df[col + '_bin'] = df[col].astype(str)
            
    # Tratar categóricas
    for col in categorical_cols:
        df[col + '_bin'] = df[col].fillna('MISSING').astype(str)
        
    bin_cols = [c for c in df.columns if c.endswith('_bin')]
    
    woe_maps = {}
    iv_summary = {}
    
    for col in bin_cols:
        stats = calculate_woe_iv(df, col, 'target')
        woe_maps[col] = stats['WoE'].to_dict()
        iv_summary[col] = stats['IV'].sum()
        
    # 5. Selección de variables
    # Seleccionamos variables con IV sustancial (> 0.02)
    selected_features = [f for f, iv in iv_summary.items() if iv > 0.02]
    
    # 6. Transformación WoE
    df_woe = pd.DataFrame()
    for col in selected_features:
        df_woe[col] = df[col].map(woe_maps[col])
        
    X = df_woe
    y = df['target']
    
    # 7. Modelado
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = LogisticRegression(C=1.0, solver='lbfgs')
    model.fit(X_train, y_train)
    
    # 8. Evaluación
    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]
    
    train_auc = roc_auc_score(y_train, train_probs)
    test_auc = roc_auc_score(y_test, test_probs)
    
    def calculate_ks(y_true, y_probs):
        df_ks = pd.DataFrame({'target': y_true, 'prob': y_probs})
        data_good = df_ks[df_ks['target'] == 0]['prob']
        data_bad = df_ks[df_ks['target'] == 1]['prob']
        return ks_2samp(data_good, data_bad).statistic

    train_ks = calculate_ks(y_train, train_probs)
    test_ks = calculate_ks(y_test, test_probs)
    
    # 9. Escalamiento (Scorecard)
    # Factor = PDO / ln(2)
    # Offset = Target Score - Factor * ln(Target Odds)
    PDO = 20
    TargetScore = 600
    TargetOdds = 50 
    
    factor = PDO / np.log(2)
    offset = TargetScore - (factor * np.log(TargetOdds))
    
    intercept = model.intercept_[0]
    coefficients = model.coef_[0]
    n_features = len(selected_features)
    
    # Generar Tabla de Scorecard
    scorecard_rows = []
    
    # Puntos del Intercepto (distribuidos)
    # Score = (Offset / n) - Factor * ( (Intercept / n) + Coef * WoE )
    # Simplificado por bin: Points = -(Coef * WoE + Intercept/n) * Factor + Offset/n
    
    for i, col in enumerate(selected_features):
        coef = coefficients[i]
        woe_dict = woe_maps[col]
        
        for bin_val, woe in woe_dict.items():
            points = -(coef * woe + intercept / n_features) * factor + (offset / n_features)
            scorecard_rows.append({
                'Variable': col.replace('_bin', ''),
                'Bin': bin_val,
                'WoE': woe,
                'Points': round(points)
            })
            
    scorecard_df = pd.DataFrame(scorecard_rows)
    scorecard_df.to_csv('model_assets/scorecard.csv', index=False)
    
    def get_score(row):
        logit = intercept + np.sum(row * coefficients)
        return offset + factor * (-logit)

    df['Score'] = X.apply(get_score, axis=1)
    
    # 10. Graficar ROC
    fpr_train, tpr_train, _ = roc_curve(y_train, train_probs)
    fpr_test, tpr_test, _ = roc_curve(y_test, test_probs)
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr_train, tpr_train, label=f'ROC Train (AUC = {train_auc:.4f})', color='royalblue', lw=2)
    plt.plot(fpr_test, tpr_test, label=f'ROC Test (AUC = {test_auc:.4f})', color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curvas ROC - Modelo de Scoring Siddiqi', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    
    asset_path = 'model_assets/roc_curve.png'
    os.makedirs('model_assets', exist_ok=True)
    plt.savefig(asset_path, dpi=300, bbox_inches='tight')
    plt.close()

    # --- RESULTADOS ---
    print("\n" + "="*50)
    print("✨ REPORTE DE ENTRENAMIENTO - MODELO SIDDIQI ✨")
    print("="*50)
    
    print(f"\n📈 MÉTRICAS DE DESEMPEÑO:")
    print(f"  - AUC (Train): {train_auc:.4f}")
    print(f"  - AUC (Test):  {test_auc:.4f}")
    print(f"  - Gini (Test): {2*test_auc - 1:.4f}")
    print(f"  - KS (Test):   {test_ks:.4f}")
    
    overfit = (train_auc - test_auc) / train_auc
    print(f"\n🔍 ANÁLISIS DE SOBREAJUSTE:")
    if overfit > 0.1:
        print(f"  - Advertencia: El modelo presenta sobreajuste del {overfit:.2%}.")
    else:
        print(f"  - Estabilidad: El modelo es estable (Diferencia: {overfit:.2%}).")
        
    print(f"\n💎 VARIABLES MÁS RELEVANTES (Top 5 por IV):")
    sorted_iv = sorted(iv_summary.items(), key=lambda x: x[1], reverse=True)
    for i, (var, iv) in enumerate(sorted_iv[:5]):
        print(f"  {i+1}. {var.replace('_bin', ''):<25} | IV: {iv:.4f}")
        
    print(f"\n🎯 SCORECARD FINAL (Extracto):")
    print(scorecard_df.sort_values(['Variable', 'Points'], ascending=[True, False]).head(15).to_string(index=False))
    print("\n... tabla completa guardada en model_assets/scorecard.csv")

    print(f"\n📊 DISTRIBUCIÓN DEL SCORE:")
    print(df['Score'].describe()[['mean', 'min', 'max', '50%']])
    
    print(f"\n📸 Imagen de curvas ROC guardada en: {asset_path}")
    
    print("\n" + "="*50)
    print("El script ha finalizado exitosamente, Dr. Fuentes. ✨")
    print("="*50)

if __name__ == "__main__":
    run_scoring_model()
