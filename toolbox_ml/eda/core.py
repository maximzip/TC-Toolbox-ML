import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns


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

    # Validaciones de entrada
    if not isinstance(df, pd.DataFrame):
        print("Error: 'df' debe ser un pandas DataFrame.")
        return None
    if not isinstance(umbral_categoria, int) or umbral_categoria <= 0:
        print("Error: 'umbral_categoria' debe ser un entero positivo.")
        return None
    if not isinstance(umbral_continua, (int, float)) or not (0 <= umbral_continua <= 100):
        print("Error: 'umbral_continua' debe ser un número entre 0 y 100.")
        return None


    # Clasificación de variables
    resultados = []
    for columna in df.columns:
        cardinalidad = df[columna].nunique(dropna=True)
        porcentaje_cardinalidad = (cardinalidad / len(df)) * 100

        # Lógica en cascada
        if cardinalidad == 2:
            tipo = "Binaria"
        elif cardinalidad < umbral_categoria:
            tipo = "Categórica"
        elif (cardinalidad >= umbral_categoria and
              porcentaje_cardinalidad >= umbral_continua):
            tipo = "Numérica Continua"
        else:
            tipo = "Numérica Discreta"
        resultados.append({
            "nombre_variable": columna,
            "tipo_sugerido": tipo
        })

    return pd.DataFrame(resultados)


def _validar_regresion_numerica(
    df: pd.DataFrame,
    target_col: str,
    umbral_corr: float,
    pvalue: float = None
) -> bool:
    """Valida argumentos comunes de las funciones numéricas de regresión."""
    if not isinstance(df, pd.DataFrame):
        print(f"Error: se esperaba un pd.DataFrame, se recibió {type(df)}")
        return False

    if target_col not in df.columns:
        print(f"Error: target_col '{target_col}' no existe en el DataFrame")
        return False

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"Error: target_col '{target_col}' debe ser numérica")
        return False

    if not isinstance(umbral_corr, float) or not 0 <= umbral_corr <= 1:
        print("Error: umbral_corr debe ser un float entre 0 y 1")
        return False

    if pvalue is not None and (not isinstance(pvalue, float) or not 0 <= pvalue <= 1):
        print("Error: pvalue debe ser None o un float entre 0 y 1")
        return False

    return True


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
        pvalue (float, opcional): Si se indica, aplica un filtro adicional
            según el p-valor.

    Retorna:
        list: Lista con los nombres de las columnas que superan los criterios.
        Retorna None si algún argumento no es válido.
    """
    if not _validar_regresion_numerica(df, target_col, umbral_corr, pvalue):
        return None

    columnas_validas = []
    columnas_numericas = df.select_dtypes(include=np.number).columns.drop(target_col, errors="ignore")

    for columna in columnas_numericas:
        # Pearson necesita pares completos y al menos dos valores no constantes.
        datos = df[[columna, target_col]].dropna()
        if len(datos) < 2 or datos[columna].nunique() < 2 or datos[target_col].nunique() < 2:
            continue

        corr, p_valor = pearsonr(datos[columna], datos[target_col])
        if abs(corr) >= umbral_corr and (pvalue is None or p_valor < pvalue):
            columnas_validas.append(columna)

    return columnas_validas


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
        pvalue (float, opcional): Filtro adicional según el p-valor.

    Retorna:
        list: Lista de columnas representadas.
        Retorna None si algún argumento no es válido.
    """
    if not _validar_regresion_numerica(df, target_col, umbral_corr, pvalue):
        return None

    if not isinstance(columns, list):
        print("Error: columns debe ser una lista")
        return None

    if columns:
        columnas_inexistentes = [col for col in columns if col not in df.columns]
        if columnas_inexistentes:
            print(f"Error: columns contiene columnas inexistentes: {columnas_inexistentes}")
            return None

        columnas_no_numericas = [col for col in columns if not pd.api.types.is_numeric_dtype(df[col])]
        if columnas_no_numericas:
            print(f"Error: columns contiene columnas no numéricas: {columnas_no_numericas}")
            return None

        df_candidatas = df[columns + [target_col]].copy() if target_col not in columns else df[columns].copy()
    else:
        df_candidatas = df.copy()

    columnas_representadas = get_features_num_regression(
        df=df_candidatas,
        target_col=target_col,
        umbral_corr=umbral_corr,
        pvalue=pvalue
    )

    if columnas_representadas is None:
        return None

    # Se pintan grupos de hasta cinco variables predictoras, incluyendo el target en cada pairplot.
    for inicio in range(0, len(columnas_representadas), 5):
        grupo = columnas_representadas[inicio:inicio + 5]
        sns.pairplot(df[grupo + [target_col]].dropna())
        plt.show()

    return columnas_representadas


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
