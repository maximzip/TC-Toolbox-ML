import pandas as pd
import numpy as np
from scipy.stats import pearsonr, mannwhitneyu, f_oneway
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
# _validar_regresion_categorica
# ---------------------------------------------------------------------------

def _validar_regresion_categorica(
    df: pd.DataFrame,
    target_col: str,
    pvalue: float
) -> bool:
    """Valida los argumentos comunes de las funciones categóricas de regresión."""
    # El input principal debe ser un DataFrame, si no, no podemos operar.
    if not isinstance(df, pd.DataFrame):
        print(f"Error: se esperaba un pd.DataFrame, se recibió {type(df)}")
        return False

    # La columna objetivo tiene que existir realmente en el DataFrame.
    if target_col not in df.columns:
        print(f"Error: target_col '{target_col}' no existe en el DataFrame")
        return False

    # Los tests comparan valores numéricos del target entre grupos: debe ser numérica.
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"Error: target_col '{target_col}' debe ser numérica")
        return False

    # El nivel de significancia es una probabilidad: float acotado entre 0 y 1.
    if not isinstance(pvalue, float) or not 0 <= pvalue <= 1:
        print("Error: pvalue debe ser un float entre 0 y 1")
        return False

    return True


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
    # Abortamos cuanto antes si la entrada no cumple las condiciones mínimas.
    if not _validar_regresion_categorica(df, target_col, pvalue):
        return None

    # Aquí iremos acumulando las columnas que pasen el test.
    columnas_significativas = []

    # Tratamos como categórica cualquier columna no numérica (object, category, bool...).
    columnas_categoricas = df.select_dtypes(exclude=np.number).columns

    for columna in columnas_categoricas:
        # Para cada categoría extraemos los valores del target asociados, sin nulos.
        # df[[columna, target_col]].dropna() elimina filas con NaN en cualquiera de las dos.
        grupos = [
            grupo[target_col].dropna().values
            for _, grupo in df[[columna, target_col]].dropna().groupby(columna, observed=True)
        ]

        # Descartamos categorías que se hayan quedado sin datos tras limpiar nulos.
        grupos = [g for g in grupos if len(g) > 0]

        # Un test de comparación necesita como mínimo dos grupos; si no, saltamos.
        if len(grupos) < 2:
            continue

        # Elegimos el test en función del número de categorías presentes:
        if len(grupos) == 2:
            # Dos grupos -> Mann-Whitney U (compara distribuciones, no asume normalidad).
            _, p_valor = mannwhitneyu(grupos[0], grupos[1])
        else:
            # Tres o más grupos -> ANOVA de un factor (compara las medias).
            # El operador * desempaqueta la lista de grupos como argumentos sueltos.
            _, p_valor = f_oneway(*grupos)

        # Si el p-valor es menor que el umbral, la relación es significativa.
        if p_valor < pvalue:
            columnas_significativas.append(columna)

    return columnas_significativas


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
    # Mismas validaciones que en _validar_regresion_categorica
    if not _validar_regresion_categorica(df, target_col, pvalue):
        return None

    # columns debe ser una lista para poder iterar y concatenar más abajo.
    if not isinstance(columns, list):
        print("Error: columns debe ser una lista")
        return None

    if columns:
        # Si el usuario pasa columnas, comprobamos que todas existan en el DataFrame.
        columnas_inexistentes = [col for col in columns if col not in df.columns]
        if columnas_inexistentes:
            print(f"Error: columns contiene columnas inexistentes: {columnas_inexistentes}")
            return None
        candidatas = columns
    else:
        # Si no se indican columnas, usamos todas las categóricas del DataFrame.
        candidatas = df.select_dtypes(exclude=np.number).columns.tolist()

    # Reutilizamos get_features_cat_regression sobre un subconjunto con solo las
    # candidatas y el target, evitando duplicar la lógica de los tests.
    # El condicional asegura que el target se incluya sin repetirlo si ya estaba.
    df_filtrado = df[candidatas + [target_col]] if target_col not in candidatas else df[candidatas]
    columnas_representadas = get_features_cat_regression(df_filtrado, target_col, pvalue)

    # Propagamos el None si la validación interna falló.
    if columnas_representadas is None:
        return None

    if with_individual_plot:
        # Modo individual: una figura independiente por cada variable significativa.
        for columna in columnas_representadas:
            plt.figure()
            # Superponemos un histograma del target por cada categoría.
            for categoria in df[columna].dropna().unique():
                subconjunto = df[df[columna] == categoria][target_col].dropna()
                # alpha=0.5 da transparencia para distinguir histogramas solapados.
                plt.hist(subconjunto, alpha=0.5, label=str(categoria))
            plt.title(f"{target_col} según {columna}")
            plt.xlabel(target_col)
            plt.ylabel("Frecuencia")
            plt.legend(title=columna)
            plt.show()
    elif columnas_representadas:
        # Modo agrupado: todas las variables en una única figura con subplots apilados.
        n = len(columnas_representadas)
        fig, axes = plt.subplots(n, 1, figsize=(8, 4 * n))

        # Con una sola variable, subplots devuelve un eje suelto y no un array;
        # lo envolvemos en lista para poder iterar de forma uniforme.
        if n == 1:
            axes = [axes]

        # Cada eje recibe los histogramas de una variable categórica.
        for ax, columna in zip(axes, columnas_representadas):
            for categoria in df[columna].dropna().unique():
                subconjunto = df[df[columna] == categoria][target_col].dropna()
                ax.hist(subconjunto, alpha=0.5, label=str(categoria))
            ax.set_title(f"{target_col} según {columna}")
            ax.set_xlabel(target_col)
            ax.set_ylabel("Frecuencia")
            ax.legend(title=columna)

        # Ajusta espaciados para que títulos y ejes no se solapen.
        plt.tight_layout()
        plt.show()

    return columnas_representadas


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
