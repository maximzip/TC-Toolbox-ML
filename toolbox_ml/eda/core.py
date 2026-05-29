import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# describe_df
# ---------------------------------------------------------------------------

def describe_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera un resumen estadístico descriptivo de un DataFrame.

    Argumentos:
        df (pd.DataFrame): DataFrame a analizar.

    Retorna:
        pd.DataFrame: DataFrame con una fila por columna del input y las
        siguientes columnas: 'tipo', 'porcentaje_nulos', 'valores_unicos',
        'porcentaje_cardinalidad'.
        Retorna None si el input no es un DataFrame válido.
    """
    # Comprobación de entrada: debe ser un DataFrame
    if not isinstance(df, pd.DataFrame):
        print(f"Error: se esperaba un pd.DataFrame, se recibió {type(df)}")
        return None

    # Cada operación ya actúa sobre todas las columnas a la vez → no necesitamos bucle
    resultado = pd.DataFrame({
        "tipo":                   df.dtypes,
        "porcentaje_nulos":       df.isnull().sum() / len(df) * 100,
        "valores_unicos":         df.nunique(),
        "porcentaje_cardinalidad": df.nunique() / len(df) * 100,
    })

    # El índice del resultado son los nombres de las columnas del DataFrame original
    resultado.index.name = None

    return resultado

# ---------------------------------------------------------------------------
# tipifica_variables
# ---------------------------------------------------------------------------

def tipifica_variables(
    df: pd.DataFrame,
    umbral_categoria: int,
    umbral_continua: float
) -> pd.DataFrame:
    """
    Sugiere el tipo estadístico de cada columna de un DataFrame.

    Argumentos:
        df (pd.DataFrame): DataFrame a analizar.
        umbral_categoria (int): Número mínimo de valores únicos para que una
            variable deje de considerarse categórica.
        umbral_continua (float): Porcentaje mínimo de cardinalidad (0-100)
            para considerar una variable como numérica continua.

    Retorna:
        pd.DataFrame: DataFrame con columnas 'nombre_variable' y 'tipo_sugerido'.
        Retorna None si algún argumento no es válido.
    """
    pass


# ---------------------------------------------------------------------------
# get_features_num_regression
# ---------------------------------------------------------------------------

def get_features_num_regression(
    df: pd.DataFrame,
    target_col: str,
    umbral_corr: float,
    pvalue: float = None
) -> list:
    """
    Devuelve columnas numéricas correlacionadas con target_col.

    Argumentos:
        df (pd.DataFrame): DataFrame de entrada.
        target_col (str): Nombre de la columna target (debe ser numérica).
        umbral_corr (float): Umbral mínimo de correlación de Pearson en valor
            absoluto (entre 0 y 1).
        pvalue (float, opcional): Si se especifica, filtra además por
            significancia estadística (p-valor < pvalue).

    Retorna:
        list: Lista con los nombres de las columnas que superan los criterios.
        Retorna None si algún argumento no es válido.
    """
    pass


# ---------------------------------------------------------------------------
# plot_features_num_regression
# ---------------------------------------------------------------------------

def plot_features_num_regression(
    df: pd.DataFrame,
    target_col: str = "",
    columns: list = [],
    umbral_corr: float = 0,
    pvalue: float = None
) -> list:
    """
    Pinta pairplots de las columnas numéricas correlacionadas con target_col.

    Argumentos:
        df (pd.DataFrame): DataFrame de entrada.
        target_col (str): Nombre de la columna target.
        columns (list): Lista de columnas candidatas. Si está vacía, se usan
            todas las columnas numéricas del DataFrame.
        umbral_corr (float): Umbral mínimo de correlación (entre 0 y 1).
        pvalue (float, opcional): Umbral de significancia estadística.

    Retorna:
        list: Lista de columnas representadas.
        Retorna None si algún argumento no es válido.
    """
    pass


# ---------------------------------------------------------------------------
# get_features_cat_regression
# ---------------------------------------------------------------------------

def get_features_cat_regression(
    df: pd.DataFrame,
    target_col: str,
    pvalue: float = 0.05
) -> list:
    """
    Devuelve columnas categóricas con relación significativa con target_col.

    Usa Mann-Whitney U para variables con 2 categorías y ANOVA para más de 2.

    Argumentos:
        df (pd.DataFrame): DataFrame de entrada.
        target_col (str): Nombre de la columna target (debe ser numérica).
        pvalue (float): Umbral de significancia estadística (entre 0 y 1).

    Retorna:
        list: Lista con los nombres de las columnas categóricas significativas.
        Retorna None si algún argumento no es válido.
    """
    pass


# ---------------------------------------------------------------------------
# plot_features_cat_regression
# ---------------------------------------------------------------------------

def plot_features_cat_regression(
    df: pd.DataFrame,
    target_col: str = "",
    columns: list = [],
    pvalue: float = 0.05,
    with_individual_plot: bool = False
) -> list:
    """
    Pinta histogramas agrupados de target_col por cada variable categórica
    que supere el test estadístico correspondiente.

    Argumentos:
        df (pd.DataFrame): DataFrame de entrada.
        target_col (str): Nombre de la columna target.
        columns (list): Lista de columnas candidatas. Si está vacía, se usan
            todas las columnas categóricas del DataFrame.
        pvalue (float): Umbral de significancia estadística (entre 0 y 1).
        with_individual_plot (bool): Si True, genera una figura por variable;
            si False, agrupa todo en una sola figura con subplots.

    Retorna:
        list: Lista de columnas representadas.
        Retorna None si algún argumento no es válido.
    """
    pass
