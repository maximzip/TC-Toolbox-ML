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
