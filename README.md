# Análisis de Árbol de Decisión - Pymes México (SME)

Este repositorio contiene un flujo de trabajo completo e interactivo para el análisis y modelado de datos de Pymes en México, enfocado originalmente en un flujo de **Orange Data Mining** y replicado didácticamente en **Python**.

## 🚀 Objetivo del Proyecto

El objetivo principal es predecir la variable `altaSAT` (estatus de alta ante el SAT) utilizando un modelo de **Árbol de Decisión**, optimizado mediante técnicas de ciencia de datos profesionales como la selección de variables por Ganancia de Información y la sintonización de hiperparámetros.

## 🛠️ Tecnologías Utilizadas

- **Python 3.13+**
- **uv**: Gestión ultra rápida de entornos virtuales y dependencias.
- **Scikit-Learn**: Entrenamiento y validación del modelo.
- **Pandas/NumPy**: Preprocesamiento de datos.
- **Matplotlib/Seaborn**: Visualizaciones premium y didácticas.
- **Jupyter Notebook**: Documentación interactiva.

## 📈 Lo que hicimos hoy

Replicamos el flujo visual de Orange en el notebook `notebooks/sme_decision_tree_analysis.ipynb`, cubriendo los siguientes pasos:

1.  **Ingesta y Limpieza**: Filtrado de la clase `o` en `altaSAT` y mapeo binario (`s=1`, `n=0`).
2.  **Ranking de Variables (Rank)**: Implementación de Ganancia de Información (Information Gain) mediante `mutual_info_classif` para identificar los 5 predictores más potentes.
3.  **Híper-parametrización**: Uso de `GridSearchCV` con validación cruzada ($k=5$) para encontrar la mejor profundidad y criterio del árbol.
4.  **Entrenamiento Didáctico**: Partición de datos 70/30 (Train/Test).
5.  **Visualización Premium**:
    - **Tree Viewer**: Diagrama del árbol final con proporciones de clase y colores vibrantes.
    - **Curva ROC y AUC**: Gráfica estética para medir la capacidad de discriminación del modelo.
    - **Métricas de Score**: Precision, Recall, F1-Score y Accuracy con explicaciones detalladas para estudiantes.

## 📁 Estructura del Repositorio

- `data/sme_mx.csv`: Dataset original de Pymes.
- `notebooks/sme_decision_tree_analysis.ipynb`: Notebook principal con el análisis detallado.
- `Orange/DT SME.ows`: Flujo de trabajo original de Orange Data Mining.
- `requirements.txt`: Lista de dependencias del proyecto.
- `ai-assisted.md`: Log de progreso del desarrollo asistido por IA.

## ⚙️ Instalación y Uso

Para ejecutar este proyecto localmente usando `uv`:

```bash
# Instalar uv si no lo tienes
curl -LsSf https://astral.sh/uv/install.sh | sh

# Crear entorno e instalar dependencias
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Lanzar el notebook
jupyter notebook notebooks/sme_decision_tree_analysis.ipynb
```

---

_Desarrollado para fines académicos por JGFuentesC con asistencia de Antigravity._
