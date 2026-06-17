import pytest
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

from toolbox_ml.eda.core import (
    describe_df,
    tipifica_variables,
    get_features_num_regression,
    plot_features_num_regression,
    get_features_cat_regression,
    plot_features_cat_regression,
    detect_outliers,
)


# ===========================================================================
# describe_df
# ===========================================================================

def test_describe_df_devuelve_dataframe():
    """Caso correcto: input valido -> retorna DataFrame."""
    df = pd.DataFrame({'a': [1, 2, None], 'b': ['x', 'y', 'z']})
    resultado = describe_df(df)
    assert isinstance(resultado, pd.DataFrame)


def test_describe_df_columnas_correctas():
    """El DataFrame resultado tiene exactamente las columnas esperadas."""
    df = pd.DataFrame({'a': [1, 2, 3]})
    resultado = describe_df(df)
    assert set(resultado.columns) == {
        'tipo', 'porcentaje_nulos', 'valores_unicos', 'porcentaje_cardinalidad'
    }


def test_describe_df_porcentaje_nulos_correcto():
    """Calcula correctamente el porcentaje de nulos."""
    df = pd.DataFrame({'a': [1, None, None, None]})
    resultado = describe_df(df)
    assert resultado.loc['a', 'porcentaje_nulos'] == pytest.approx(75.0, abs=0.01)


def test_describe_df_retorna_none_con_input_invalido():
    """Caso de error: input no es DataFrame -> retorna None."""
    assert describe_df("no es un dataframe") is None
    assert describe_df([1, 2, 3]) is None


# ===========================================================================
# get_features_num_regression
# ===========================================================================

def test_get_features_num_regression_devuelve_columnas_correlacionadas():
    """Caso correcto: selecciona variables numericas correlacionadas."""
    df = pd.DataFrame({
        "target": [1, 2, 3, 4, 5, 6],
        "muy_corr": [2, 4, 6, 8, 10, 12],
        "poco_corr": [1, 1, 2, 2, 1, 2],
        "categoria": ["a", "b", "a", "b", "a", "b"],
    })

    resultado = get_features_num_regression(df, "target", 0.9)

    assert resultado == ["muy_corr"]


def test_get_features_num_regression_filtra_por_pvalue():
    """Caso correcto: aplica filtro adicional de significancia estadistica."""
    df = pd.DataFrame({
        "target": [1, 2, 3, 4, 5, 6],
        "muy_corr": [2, 4, 6, 8, 10, 12],
    })

    resultado = get_features_num_regression(df, "target", 0.9, pvalue=0.05)

    assert resultado == ["muy_corr"]


def test_get_features_num_regression_ignora_constantes_y_nulos():
    """Caso limite: ignora columnas constantes o sin pares suficientes."""
    df = pd.DataFrame({
        "target": [1, 2, 3, 4],
        "constante": [7, 7, 7, 7],
        "con_nulos": [np.nan, np.nan, np.nan, 1],
    })

    resultado = get_features_num_regression(df, "target", 0.0)

    assert resultado == []


def test_get_features_num_regression_retorna_none_con_input_invalido():
    """Caso de error: retorna None con argumentos invalidos."""
    df = pd.DataFrame({"target": [1, 2, 3], "x": [1, 2, 3]})

    assert get_features_num_regression("no df", "target", 0.3) is None
    assert get_features_num_regression(df, "no_existe", 0.3) is None
    assert get_features_num_regression(df, "target", 1) is None


# ===========================================================================
# plot_features_num_regression
# ===========================================================================

def test_plot_features_num_regression_devuelve_columnas_representadas():
    """Caso correcto: devuelve las columnas que pasan el filtro numerico."""
    df = pd.DataFrame({
        "target": [1, 2, 3, 4, 5, 6],
        "muy_corr": [2, 4, 6, 8, 10, 12],
        "poco_corr": [1, 1, 2, 2, 1, 2],
    })

    resultado = plot_features_num_regression(df, "target", umbral_corr=0.9)

    assert resultado == ["muy_corr"]


def test_plot_features_num_regression_respeta_columns():
    """Caso correcto: filtra solo sobre las columnas candidatas recibidas."""
    df = pd.DataFrame({
        "target": [1, 2, 3, 4, 5, 6],
        "muy_corr": [2, 4, 6, 8, 10, 12],
        "otra_corr": [3, 6, 9, 12, 15, 18],
    })

    resultado = plot_features_num_regression(
        df,
        "target",
        columns=["otra_corr"],
        umbral_corr=0.9
    )

    assert resultado == ["otra_corr"]


def test_plot_features_num_regression_retorna_none_con_columns_invalidas():
    """Caso de error: columns debe ser una lista de columnas existentes."""
    df = pd.DataFrame({"target": [1, 2, 3], "x": [1, 2, 3]})

    assert plot_features_num_regression(df, "target", columns="x", umbral_corr=0.3) is None
    assert plot_features_num_regression(df, "target", columns=["no_existe"], umbral_corr=0.3) is None


# ===========================================================================
# get_features_cat_regression
# ===========================================================================

def test_get_features_cat_regression_detecta_significativa():
    """Caso correcto: selecciona la categórica relacionada con el target."""
    df = pd.DataFrame({
        "target": [1, 2, 1, 2, 10, 11, 10, 11],
        "significativa": ["a", "a", "a", "a", "b", "b", "b", "b"],
        "irrelevante": ["x", "y", "x", "y", "x", "y", "x", "y"],
    })

    resultado = get_features_cat_regression(df, "target", pvalue=0.05)

    assert "significativa" in resultado
    assert "irrelevante" not in resultado


def test_get_features_cat_regression_usa_anova_mas_de_dos_grupos():
    """Caso correcto: con 3+ categorías aplica ANOVA y detecta la relación."""
    df = pd.DataFrame({
        "target": [1, 1, 1, 5, 5, 5, 10, 10, 10],
        "tres_grupos": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
    })

    resultado = get_features_cat_regression(df, "target", pvalue=0.05)

    assert resultado == ["tres_grupos"]


def test_get_features_cat_regression_dataframe_vacio():
    """Caso límite: DataFrame sin filas -> lista vacía."""
    df = pd.DataFrame({"target": pd.Series([], dtype=float),
                       "cat": pd.Series([], dtype=object)})

    resultado = get_features_cat_regression(df, "target", pvalue=0.05)

    assert resultado == []


def test_get_features_cat_regression_una_sola_categoria():
    """Caso límite: categórica con un único valor -> no se puede testear."""
    df = pd.DataFrame({
        "target": [1, 2, 3, 4],
        "constante": ["a", "a", "a", "a"],
    })

    resultado = get_features_cat_regression(df, "target", pvalue=0.05)

    assert resultado == []


def test_get_features_cat_regression_retorna_none_con_input_invalido():
    """Caso de error: retorna None con argumentos inválidos."""
    df = pd.DataFrame({"target": [1, 2, 3], "cat": ["a", "b", "c"]})

    assert get_features_cat_regression("no df", "target") is None
    assert get_features_cat_regression(df, "no_existe") is None
    assert get_features_cat_regression(df, "cat") is None          # target no numérica
    assert get_features_cat_regression(df, "target", pvalue=1.5) is None
    assert get_features_cat_regression(df, "target", pvalue=2) is None  # no es float


# ===========================================================================
# plot_features_cat_regression
# ===========================================================================

def test_plot_features_cat_regression_devuelve_columnas_representadas():
    """Caso correcto: devuelve las categóricas que superan el test."""
    df = pd.DataFrame({
        "target": [1, 2, 1, 2, 10, 11, 10, 11],
        "significativa": ["a", "a", "a", "a", "b", "b", "b", "b"],
        "irrelevante": ["x", "y", "x", "y", "x", "y", "x", "y"],
    })

    resultado = plot_features_cat_regression(df, "target", pvalue=0.05)

    assert "significativa" in resultado
    assert "irrelevante" not in resultado


def test_plot_features_cat_regression_respeta_columns():
    """Caso correcto: filtra solo sobre las columnas candidatas recibidas."""
    df = pd.DataFrame({
        "target": [1, 2, 1, 2, 10, 11, 10, 11],
        "significativa": ["a", "a", "a", "a", "b", "b", "b", "b"],
        "otra_sig": ["p", "p", "p", "p", "q", "q", "q", "q"],
    })

    resultado = plot_features_cat_regression(
        df,
        "target",
        columns=["otra_sig"],
        pvalue=0.05
    )

    assert resultado == ["otra_sig"]


def test_plot_features_cat_regression_modo_individual():
    """Caso correcto: con with_individual_plot=True devuelve igualmente la lista."""
    df = pd.DataFrame({
        "target": [1, 2, 1, 2, 10, 11, 10, 11],
        "significativa": ["a", "a", "a", "a", "b", "b", "b", "b"],
    })

    resultado = plot_features_cat_regression(
        df,
        "target",
        pvalue=0.05,
        with_individual_plot=True
    )

    assert resultado == ["significativa"]


def test_plot_features_cat_regression_dataframe_vacio():
    """Caso límite: DataFrame sin filas -> lista vacía sin pintar nada."""
    df = pd.DataFrame({"target": pd.Series([], dtype=float),
                       "cat": pd.Series([], dtype=object)})

    resultado = plot_features_cat_regression(df, "target", pvalue=0.05)

    assert resultado == []


def test_plot_features_cat_regression_retorna_none_con_input_invalido():
    """Caso de error: retorna None con argumentos inválidos."""
    df = pd.DataFrame({"target": [1, 2, 3], "cat": ["a", "b", "c"]})

    assert plot_features_cat_regression("no df", "target") is None
    assert plot_features_cat_regression(df, "no_existe") is None
    assert plot_features_cat_regression(df, "cat") is None              # target no numérica
    assert plot_features_cat_regression(df, "target", columns="cat") is None
    assert plot_features_cat_regression(df, "target", columns=["no_existe"]) is None
    assert plot_features_cat_regression(df, "target", pvalue=1.5) is None
# ===========================================================================
# detect_outliers (BONUS)
# ===========================================================================

def test_detect_outliers_devuelve_dict():
    """Caso correcto: input válido → retorna un diccionario."""
    df = pd.DataFrame({'valores': [10, 12, 11, 13, 12, 1000]})
    resultado = detect_outliers(df)
    assert isinstance(resultado, dict)


def test_detect_outliers_detecta_outlier_iqr():
    """Caso correcto: detecta el outlier evidente por el método IQR."""
    df = pd.DataFrame({'valores': [10, 12, 11, 13, 12, 1000]})
    resultado = detect_outliers(df)
    # El valor 1000 (posición 5) debe detectarse como único outlier por IQR
    assert resultado['valores']['iqr']['n_outliers'] == 1
    assert resultado['valores']['iqr']['indices'] == [5]


def test_detect_outliers_estructura_correcta():
    """El resultado tiene las claves esperadas para cada columna numérica."""
    df = pd.DataFrame({'a': [1, 2, 3, 4, 100]})
    resultado = detect_outliers(df)
    assert set(resultado['a'].keys()) == {'iqr', 'zscore'}
    assert set(resultado['a']['iqr'].keys()) == {'n_outliers', 'porcentaje', 'indices'}


def test_detect_outliers_ignora_columnas_no_numericas():
    """Las columnas de texto no se analizan."""
    df = pd.DataFrame({'num': [1, 2, 3, 4, 100], 'texto': ['a', 'b', 'c', 'd', 'e']})
    resultado = detect_outliers(df)
    assert 'num' in resultado
    assert 'texto' not in resultado


def test_detect_outliers_dataframe_vacio():
    """Caso límite: DataFrame vacío → diccionario vacío."""
    df = pd.DataFrame()
    resultado = detect_outliers(df)
    assert resultado == {}


def test_detect_outliers_columna_todos_nulos():
    """Caso límite: columna con todos los valores nulos → 0 outliers."""
    df = pd.DataFrame({'a': [np.nan, np.nan, np.nan]})
    resultado = detect_outliers(df)
    assert resultado['a']['iqr']['n_outliers'] == 0
    assert resultado['a']['zscore']['n_outliers'] == 0


def test_detect_outliers_input_invalido():
    """Caso de error: input no es un DataFrame → retorna None."""
    assert detect_outliers("esto no es un dataframe") is None
    assert detect_outliers([1, 2, 3]) is None


def test_detect_outliers_umbral_z_invalido():
    """Caso de error: umbral_z no válido → retorna None."""
    df = pd.DataFrame({'a': [1, 2, 3]})
    assert detect_outliers(df, umbral_z=-1) is None
    assert detect_outliers(df, umbral_z="alto") is None
    
    assert plot_features_cat_regression(df, "target", pvalue=1.5) is None
