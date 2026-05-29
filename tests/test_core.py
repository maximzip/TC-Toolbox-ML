import pytest
import pandas as pd
import numpy as np
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
    """Caso correcto: input válido → retorna DataFrame."""
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
    """Caso de error: input no es DataFrame → retorna None."""
    assert describe_df("no es un dataframe") is None
    assert describe_df([1, 2, 3]) is None


# ===========================================================================
# tipifica_variables
# ===========================================================================

def test_tipifica_variables_devuelve_dataframe():
    """Caso correcto: input válido → retorna DataFrame."""
    df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
    resultado = tipifica_variables(df, umbral_categoria=10, umbral_continua=30.0)
    assert isinstance(resultado, pd.DataFrame)


def test_tipifica_variables_columnas_correctas():
    """El resultado tiene las columnas 'nombre_variable' y 'tipo_sugerido'."""
    df = pd.DataFrame({'a': [1, 2, 3]})
    resultado = tipifica_variables(df, umbral_categoria=10, umbral_continua=30.0)
    assert set(resultado.columns) == {'nombre_variable', 'tipo_sugerido'}


def test_tipifica_variables_detecta_binaria():
    """Una columna con exactamente 2 valores únicos se clasifica como Binaria."""
    df = pd.DataFrame({'sexo': ['M', 'F', 'M', 'F', 'M']})
    resultado = tipifica_variables(df, umbral_categoria=10, umbral_continua=30.0)
    assert resultado.loc[resultado['nombre_variable'] == 'sexo', 'tipo_sugerido'].values[0] == 'Binaria'


def test_tipifica_variables_retorna_none_umbral_invalido():
    """Caso de error: umbral_categoria no entero → retorna None."""
    df = pd.DataFrame({'a': [1, 2, 3]})
    assert tipifica_variables(df, umbral_categoria=3.5, umbral_continua=30.0) is None
