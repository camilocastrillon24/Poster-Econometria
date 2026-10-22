# %% Importación de paquetes ============================

# Trabajar con rutas relativas en python 
from pathlib import Path

# Módulos de numpy, pandas, matplotlib y scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera, probplot

# Módulos de statsmodels
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss


# %% Cargar bases de datos en python usando rutas relativas =========================

# Obtener la ruta del directorio raíz
BASE_DIR = Path(__file__).resolve().parents[0]

# Obtener la ruta del directorio con los datos
DATA_DIR = BASE_DIR / "datos"

# Rutas de las bases de datos 
ruta_exp = DATA_DIR / "datos_ipc.csv" # Base de datos de ipc

# %% PRIMERA SERIE: IPC
# ============================

# Base de datos con la serie importada a python
ipc_base = pd.read_csv(
    ruta_exp,
    header=None,
    names=["ipc_variacion_mensual"]
)

# Ver el tipo de objeto de la base de datos (Pandas.DataFrame)
print(type(ipc_base))

# Ver primeras y últimas observaciones de la base de datos
print(ipc_base.head()) # Primeras observaciones
print(ipc_base.tail()) # Últimas observaciones 

# %% Creación del índice temporal de las series de tiempo

# Creación del índice temporal
fechas_expo_ipc = pd.date_range(
    start="2010-01-01",
    periods=len(ipc_base),
    freq="MS"
)

# Agregar el índice temporal a la base de datos de exportaciones tradicionales
ipc_base.index = fechas_expo_ipc
# El tipo de objeto de la base de datos sigue siendo Pandas.DataFrame
print(type(ipc_base))

# Ver primeras y últimas observaciones de la base de datos, ahora con índice temporal
print(ipc_base.head()) # Primeras observaciones
print(ipc_base.tail()) # Últimas observaciones 


# %% Creación de la serie de tiempo de "exportaciones"

# La nueva serie de tiempo se va a llamar "expo_serie" y va a tener valores numéricos
ipc_serie = ipc_base["ipc_variacion_mensual"].copy()
ipc_serie = pd.to_numeric(ipc_serie, errors="coerce")
ipc_serie = ipc_serie.dropna() # Borrar missing values

# Ver el principio y final de la serie de tiempo
print(ipc_serie.head())
print(ipc_serie.tail())

# # El tipo de objeto de la base de datos ahora es un Pandas.Series
print(type(ipc_serie))

# Ver algunas estadísticas descriptivas de la serie de tiempo 
print(ipc_serie.describe())


# %%  Paso 1: Identificación
# ============================

# Gráfica de la serie de tiempo "exportaciones tradicionales"
plt.figure(figsize=(10, 5))
plt.plot(ipc_serie)
plt.title("ipc, 2010-2026")
plt.xlabel("Fecha")
plt.ylabel("Valor")
plt.grid(True)
plt.show()

# %% FAC y FACP de la serie original

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_acf(
    ipc_serie,
    lags=24,
    alpha=0.05,
    bartlett_confint=False,
    ax=axes[0]
)

axes[0].set_title("FAC variacion mensual IPC")

plot_pacf(
    ipc_serie,
    lags=24,
    alpha=0.05,
    ax=axes[1]
)

axes[1].set_title("FACP de la serie original")

plt.tight_layout()
plt.show()

# %% Tests de Raíz Unitaria 

# Test de Augmented Dickey Fuller (ADF)
adf_result = adfuller(ipc_serie) # Test de Dickey Fuller sin intercepto y tendencia determínistica

# Nota: En el test de ADF si no rechazo la H0 la serie no es estacionaria
#       y si rechazo la H0 la serie es estacionaria. 

print("=== Test ADF ===")
print("Estadístico ADF:", adf_result[0])
print("p-valor:", adf_result[1])
print("Rezagos usados:", adf_result[2]) 
print("Observaciones:", adf_result[3])
print("Valores críticos:")
for nivel, valor in adf_result[4].items():
    print(f"{nivel}: {valor}")

if adf_result[1] < 0.05:
    print("ADF: Rechazamos H0. Según el test, la serie es estacionaria.")
else:
    print("ADF: No rechazamos H0. Según el test, la serie no es estacionaria.")

# Test KPSS
kpss_result = kpss(ipc_serie, regression="c", nlags="auto")

# Nota: En prueba KPSS se interpreta al contario que una prueba ADF.
#       Si no rechazo la H0 la serie es estacionaria
#       y si rechazo la H0 la serie es no estacionaria. 

print("\n=== Test KPSS ===")
print("Estadístico KPSS:", kpss_result[0])
print("p-valor:", kpss_result[1])
print("Rezagos usados:", kpss_result[2])
print("Valores críticos:")
for nivel, valor in kpss_result[3].items():
    print(f"{nivel}: {valor}")

if kpss_result[1] < 0.05:
    print("KPSS: rechazamos H0. Según el test, la serie es no estacionaria.")
else:
    print("KPSS: no rechazamos H0. Según el test, La serie es estacionaria.")

# %% Identificación del modelo usando FAC y FACP

# La serie parece estacionaria (oscila alrededor de una media constante),
# por lo que no se requiere diferenciación (d = 0).
# La FAC decae lentamente y presenta varios rezagos significativos
# (1, 2, 3, 4, 5, 8, 12 y 15), por lo que no corta de golpe como en un MA puro.
# La FACP muestra un pico dominante en el rezago 1 y picos marginales en
# los rezagos 2, 3 y 4, después de los cuales cae dentro de las bandas.
# Esto es característico de un proceso autorregresivo.
# Por tanto, se propone inicialmente un modelo AR(1), es decir,
# un ARIMA(1, 0, 0) sobre la serie, y como alternativas un AR(2)
# y un ARMA(1, 1).


# P.d. También se pueden usar criterios de información (AIC y BIC) para la
#      selección de los órdenes p y q del modelo ARIMA.

# %% # %% Paso 2: Estimación 
# ============================

# Se estimaran 3 modelos en éste caso, un ARIMA(1,0,0), un ARIMA(2,0,0) y un ARIMA(1,0,1)

# Se crea un diccionari de python especificando los ordenes (p,q) de cada uno de los modelos
# que se estimarán
modelos_serie_ipc = {
    "ARMA(1,0)": (1, 0, 0),
    "ARMA(2,0)": (2, 0, 0),
    "ARMA(1,1)": (1, 0, 1),
}

nombres_modelos = list(modelos_serie_ipc.keys())

# Diccionario que almacenara las estimaciones de cada uno de los modelos. Los "keys" del 
# diccionario son los nombres de los modelos, y los "values" son las estimaciones de los modelos
estimaciones_ipc_serie = {}

# Función para calcular la media de largo plazo del modelo, a partir de los coeficientes
# estimados en los modelos
def media_incondicional_sarimax(resultado):
    """Calcula E[y_t] para un modelo ARMA(p,q)"""
    params = resultado.params
    intercepto = params.get("intercept", 0)
    suma_ar = sum(
        valor
        for parametro, valor in params.items()
        if parametro.startswith("ar.L")
    )
    denominador = 1 - suma_ar

    if np.isclose(denominador, 0):
        return np.nan

    return intercepto / denominador


def se_media_incondicional_sarimax(resultado):
    """Calcula el error estándar de la media incondicional usando método delta."""
    params = resultado.params

    if "intercept" not in params.index:
        return np.nan

    parametros_ar = [
        parametro
        for parametro in params.index
        if parametro.startswith("ar.L")
    ]
    intercepto = params["intercept"]
    denominador = 1 - sum(params[parametro] for parametro in parametros_ar)

    if np.isclose(denominador, 0):
        return np.nan

    cov_params = resultado.cov_params()
    gradiente = pd.Series(0.0, index=params.index)
    gradiente["intercept"] = 1 / denominador

    for parametro in parametros_ar:
        gradiente[parametro] = intercepto / denominador**2

    varianza_mu = float(gradiente @ cov_params @ gradiente)

    if varianza_mu < 0:
        return np.nan

    return np.sqrt(varianza_mu)

# El loop estima los 3 modelos ARMA propusetos en "modelos_serie_te"

# Se itera sobre el diccionario "modelos_serie_te". "nombre" corresponde al nombre del modelo
# y "orden" indica el orden (p,d,q) del modelo ARIMA. En éste caso se itera 3 veces, porque
# el diccionario tiene 3 modelos. 
for nombre, orden in modelos_serie_ipc.items():
    
    # Especicación de cada uno de los modelos
    modelo = SARIMAX(
        ipc_serie,
        order=orden,
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False
    ) 
    
    # Nota: Se especifican en éste ejemplo particular, cada uno de los modelos en nivel

    # Se estiman los modelos, dada la especificación dada
    estimacion_modelo = modelo.fit(disp=False)
    
    # Nota: El método de estimación es máxima verosimilitud 
    #       sobre la representación del modelo en un espacio de estados
    
    # Se llena el diccionario de modelos estimados. Los "keys" del diccionario serán los nombres
    # de los modelos y los "values" del diccionario son las estimaciones de los modelos
    estimaciones_ipc_serie[nombre] = estimacion_modelo

    print("\n", nombre)
    print(estimacion_modelo.summary())
    print("\n")
    print(
        "Media incondicional implícita en SARIMAX "
        f"c / (1 - suma AR): {media_incondicional_sarimax(estimacion_modelo):.3f}"
    )

# %% Tabla resumen de modelos estimados


# Inicialmente la tabla de modelos es una lista de python
tabla_modelos = []

# Se itera sobre los keys y values del diccionario "estimaciones_te_serie" 
for nombre, resultado in estimaciones_ipc_serie.items():
    # nombre: es la variable que itera en el nombre de los modelos
    # resultado: es la variable que itera en la estimación de los modelos
    params = resultado.params # Da los estimaciones de los parámetros del modelo
    errores = resultado.bse # Da los errores estándar de las estimaciones de los parámetros del modelo

    # Estimacioón del intercepto del modelo y su error estándar
    intercepto_sarimax = params.get("intercept", np.nan)
    se_intercepto_sarimax = errores.get("intercept", np.nan)

    # Parámetros de la parte AR y MA
    ar1 = params.get("ar.L1", np.nan)
    ar2 = params.get("ar.L2", np.nan)
    ma1 = params.get("ma.L1", np.nan)

    # Error estándar de la parte AR y MA
    se_ar1 = errores.get("ar.L1", np.nan)
    se_ar2 = errores.get("ar.L2", np.nan)
    se_ma1 = errores.get("ma.L1", np.nan)

    # Media de largo plazo para los modelos AR
    ar1_mu = params.get("ar.L1", 0)
    ar2_mu = params.get("ar.L2", 0)

    # Cálculo de la media dependiendo del tipo de modelo 
    if "ar.L1" in params.index or "ar.L2" in params.index:
        mu = intercepto_sarimax / (1 - ar1_mu - ar2_mu)
    else:
        mu = intercepto_sarimax
    se_mu = se_media_incondicional_sarimax(resultado)

    # Se llena una a una la lista, con los resultados principales de las estimaciones
    # Dichos resultados quedan como un diccionario, es decir se obtiene una lista de diccionarios. 
    tabla_modelos.append({
        "Modelo": nombre,
        "intercepto_sarimax": intercepto_sarimax,
        "se_intercepto_sarimax": se_intercepto_sarimax,
        "media_incondicional": mu,
        "se_media_incondicional": se_mu,
        "a1": ar1,
        "se_a1": se_ar1,
        "a2": ar2,
        "se_a2": se_ar2,
        "b1": ma1,
        "se_b1": se_ma1,
        "AIC": resultado.aic,
        "BIC": resultado.bic,
    })

# Transformar la lista en un pandas DataFrame
tabla_modelos_te = pd.DataFrame(tabla_modelos)

# Nota: Hasta acá ya tenemos una tabla funcional. 

# Especificar los valores de las estimaciones en la tabla
def formato_estimacion(valor, decimales=3):
    if pd.isna(valor):
        return ""
    return f"{valor:.{decimales}f}"

# Especificar los valores de los errores estándar en la tabla
def formato_error_estandar(valor, decimales=3):
    if pd.isna(valor):
        return ""
    return f"({valor:.{decimales}f})"

# Acá se generá la tabla final de publicación. 

filas_tabla_publicacion = [
    ("a1", "a1", "se_a1"),
    ("", "se_a1", None),
    ("a2", "a2", "se_a2"),
    ("", "se_a2", None),
    ("b1", "b1", "se_b1"),
    ("", "se_b1", None),
    ("intercepto SARIMAX", "intercepto_sarimax", "se_intercepto_sarimax"),
    ("", "se_intercepto_sarimax", None),
    ("media incondicional", "media_incondicional", "se_media_incondicional"),
    ("", "se_media_incondicional", None),
    ("AIC", "AIC", None),
    ("BIC", "BIC", None),
]

# Primero la tabla final es una lista
tabla_publicacion = []

# loop que llena la lista para la tabla final 
for etiqueta, columna_valor, columna_error in filas_tabla_publicacion:
    fila = {"Parámetro": etiqueta}

    for nombre in nombres_modelos:
        modelo = tabla_modelos_te.loc[
            tabla_modelos_te["Modelo"] == nombre
        ].iloc[0]

        if columna_error is None and columna_valor.startswith("se_"):
            fila[nombre] = formato_error_estandar(modelo[columna_valor])
        elif columna_valor in ["AIC", "BIC"]:
            fila[nombre] = formato_estimacion(modelo[columna_valor], decimales=1)
        else:
            fila[nombre] = formato_estimacion(modelo[columna_valor])

    tabla_publicacion.append(fila)

# Genera la tabla final como un Pandas Dataframe
tabla_modelos_te_publicacion = pd.DataFrame(tabla_publicacion)

# Imprimir la tabla final en formato de texto
print("\nTabla resumen de modelos estimados")
print(tabla_modelos_te_publicacion.to_string(index=False))
print("\nErrores estándar entre paréntesis.")

# %%
