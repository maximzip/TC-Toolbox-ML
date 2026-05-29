# toolbox_ml

Paquete de Python con herramientas para análisis exploratorio de datos (EDA) y Machine Learning, desarrollado como Team Challenge del Bootcamp The Bridge de Data Science.

---

## Instalación

```bash
git clone https://github.com/vuestro-grupo/toolbox_ml.git
cd toolbox_ml
python -m venv venv
source venv/bin/activate   # Mac/Linux
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
pip install -e .
```

---

## Funciones disponibles

### `describe_df(df)`
Devuelve un resumen estadístico del DataFrame: tipo de dato, porcentaje de nulos, valores únicos y porcentaje de cardinalidad por columna.

```python
from toolbox_ml.eda.core import describe_df
describe_df(df)
```

### `tipifica_variables(df, umbral_categoria, umbral_continua)`
Sugiere el tipo estadístico de cada columna: Binaria, Categórica, Numérica Discreta o Numérica Continua.

```python
from toolbox_ml.eda.core import tipifica_variables
tipifica_variables(df, umbral_categoria=10, umbral_continua=30.0)
```

### `get_features_num_regression(df, target_col, umbral_corr, pvalue=None)`
Devuelve columnas numéricas correlacionadas con el target por encima del umbral indicado.

### `plot_features_num_regression(df, target_col, columns, umbral_corr, pvalue=None)`
Pinta pairplots de las columnas numéricas seleccionadas frente al target.

### `get_features_cat_regression(df, target_col, pvalue=0.05)`
Devuelve columnas categóricas con relación estadísticamente significativa con el target (Mann-Whitney U o ANOVA según cardinalidad).

### `plot_features_cat_regression(df, target_col, columns, pvalue=0.05, with_individual_plot=False)`
Pinta histogramas agrupados de las columnas categóricas seleccionadas.

---

## Ejecutar los tests

```bash
pytest tests/ -v
```

---

## Equipo

| Nombre | Rol | Funciones |
|--------|-----|-----------|
| (nombre) | Scrum Master | Setup del repo, integración, notebook demo |
| Maksym | Dev 1 | `describe_df`, `tipifica_variables` |
| (nombre) | Dev 2 | `get/plot_features_num_regression` |
| (nombre) | Dev 3 | `get/plot_features_cat_regression` |

---

## Stack tecnológico

Python 3.10+ · pandas · numpy · scipy · matplotlib · seaborn · scikit-learn · pytest
