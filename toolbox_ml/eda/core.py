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


# ---------------------------------------------------------------------------
# detect_outliers (BONUS)
# ---------------------------------------------------------------------------

def detect_outliers(df: pd.DataFrame, umbral_z: float = 3.0) -> dict:
    """
    Detecta valores atípicos (outliers) en las columnas numéricas de un DataFrame
    usando dos métodos: rango intercuartílico (IQR) y Z-score.

    Argumentos:
        df (pd.DataFrame): DataFrame a analizar.
        umbral_z (float): Número de desviaciones estándar a partir del cual un valor
            se considera outlier por el método Z-score. Por defecto 3.0.

    Retorna:
        dict: Diccionario con una entrada por cada columna numérica. Para cada columna
        se devuelve, en los métodos 'iqr' y 'zscore', un diccionario con:
            - 'n_outliers' (int): número de outliers detectados.
            - 'porcentaje' (float): porcentaje de outliers sobre el total de filas.
            - 'indices' (list): índices del DataFrame donde están los outliers.
        Retorna None si 'df' no es un DataFrame o 'umbral_z' no es un número positivo.
    """
    # Comprobación de entrada: df debe ser un DataFrame
    if not isinstance(df, pd.DataFrame):
        print("Error: 'df' debe ser un pandas DataFrame.")
        return None

    # Comprobación de entrada: umbral_z debe ser un número positivo (y no un booleano)
    if not isinstance(umbral_z, (int, float)) or isinstance(umbral_z, bool) or umbral_z <= 0:
        print("Error: 'umbral_z' debe ser un número positivo.")
        return None

    # Seleccionamos solo las columnas numéricas: son las únicas en las que tiene sentido buscar outliers
    columnas_numericas = df.select_dtypes(include=np.number).columns

    resultado = {}
    total_filas = len(df)

    # Recorremos cada columna numérica y calculamos sus outliers por los dos métodos
    for col in columnas_numericas:
        # Quitamos los nulos para los cálculos
        serie = df[col].dropna()

        # Si la columna no tiene datos válidos, devolvemos resultados vacíos para ella
        if serie.empty:
            resultado[col] = {
                'iqr': {'n_outliers': 0, 'porcentaje': 0.0, 'indices': []},
                'zscore': {'n_outliers': 0, 'porcentaje': 0.0, 'indices': []}
            }
            continue

        # --- Método 1: IQR (rango intercuartílico) ---
        q1 = serie.quantile(0.25)            # primer cuartil
        q3 = serie.quantile(0.75)            # tercer cuartil
        iqr = q3 - q1                        # rango intercuartílico
        limite_inferior = q1 - 1.5 * iqr     # frontera por debajo
        limite_superior = q3 + 1.5 * iqr     # frontera por encima
        # Es outlier todo lo que queda fuera de esas fronteras
        mascara_iqr = (serie < limite_inferior) | (serie > limite_superior)
        indices_iqr = serie[mascara_iqr].index.tolist()

        # --- Método 2: Z-score ---
        media = serie.mean()
        desviacion = serie.std()
        if desviacion == 0:
            # Si todos los valores son iguales no hay desviación y no hay outliers
            indices_z = []
        else:
            z_scores = (serie - media) / desviacion
            # Es outlier si se aleja de la media más de 'umbral_z' desviaciones
            mascara_z = z_scores.abs() > umbral_z
            indices_z = serie[mascara_z].index.tolist()

        # Guardamos los resultados de los dos métodos para esta columna
        resultado[col] = {
            'iqr': {
                'n_outliers': len(indices_iqr),
                'porcentaje': round(len(indices_iqr) / total_filas * 100, 2),
                'indices': indices_iqr
            },
            'zscore': {
                'n_outliers': len(indices_z),
                'porcentaje': round(len(indices_z) / total_filas * 100, 2),
                'indices': indices_z
            }
        }

    return resultado
