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