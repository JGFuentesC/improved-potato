# Análisis de Modelado Predictivo - Pymes México (SME)

Este repositorio contiene un flujo de trabajo avanzado para el análisis y modelado de datos de Pymes en México. Evolucionó de una fase didáctica basada en **Árboles de Decisión** hacia una fase de producción y optimización mediante **Regresión Logística**.

## 🚀 Objetivo del Proyecto

El corazón del proyecto es predecir el estatus de formalidad de las Pymes (`altaSAT`) mediante algoritmos de aprendizaje supervisado, permitiendo identificar patrones de comportamiento comercial y contable en el ecosistema mexicano.

## 🛠️ Tecnologías Utilizadas

- **Python 3.13+**
- **uv**: Gestión ultra rápida de entornos y dependencias.
- **Scikit-Learn**: Entrenamiento, optimización (GridSearch) y validación.
- **Pandas/NumPy**: Ingeniería de datos.
- **Matplotlib/Seaborn**: Visualizaciones premium (Midnight Gold Style).

---

## 📈 Hitos del Desarrollo

### 1. Resolución de Práctica: Árboles de Decisión (Commit Anterior)

Se completó la resolución de la práctica de modelos no paramétricos:

- **Notebook**: `notebooks/practica_arboles.ipynb`.
- **Implementación**: Análisis de nodos, profundidad óptima y visualización de reglas de decisión.
- **Resultado**: Modelo `DecisionTree_Fraud_v1` enfocado en la interpretabilidad de reglas de negocio.

### 2. Modelo de Regresión Logística Optimizado (Tarea Actual)

Se implementó un script de entrenamiento robusto para Pymes:

- **Script**: `scripts/train_sme_logistic.py`.
- **Preprocesamiento**: Estandarización de variables y codificación One-Hot.
- **Optimización**: Uso de **GridSearchCV** para hallar la regularización óptima (`C`) y **Lasso (L1)** para selección de variables.
- **Visualización Midnight Gold**:
  - **Curva ROC Dual**: Comparativa sincronizada entre Entrenamiento y Prueba.
  - **Matriz de Confusión**: Análisis de precisión predictiva con estética de alto contraste.
- **Reporte Ejecutivo**: Generación automática de `model_assets/sme_logistic_report.md`.

---

## 📁 Estructura del Repositorio

- `data/sme_mx.csv`: Dataset original de Pymes.
- `notebooks/`:
  - `sme_decision_tree_analysis.ipynb`: Replicación del flujo de Orange.
  - `practica_arboles.ipynb`: Resolución de la práctica académica.
- `scripts/`:
  - `train_sme_logistic.py`: Entrenamiento optimizado del modelo logístico.
- `model_assets/`: Reportes, modelos serializados (`.pkl`) y gráficas de alta fidelidad.
- `ai-assisted.md`: Log detallado de tareas y progreso asistido.

## ⚙️ Instalación y Uso

```bash
# Iniciar entorno y dependencias con uv
uv sync

# Ejecutar el entrenamiento del modelo logístico
uv run python scripts/train_sme_logistic.py
```

---

_Desarrollado con rigor académico y asistencia de Antigravity._
