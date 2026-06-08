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
    detect_outliers,
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
    