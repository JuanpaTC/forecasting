import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from clean_data import items, purchases, sales
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

'''
Consideraciones:
        (1) definir el intervalo de días/meses para el calculo --- problemas para pronosticar diariamente
        (2) definir el horizonte que se quiere pronosticar
        (3) investigar más sobre algoritmos para ajustar los parametros que sean más eficientes.
        (4) investigar sobre prueba Dickey-Fuller
'''

########################## Producto de prueba ##########################

test_item = sales.groupby('item_id')['quantity'].sum().idxmax() # más vendido, con estacionalidad pero con demanda cte. (se tienen varias observaciones)

########################################################################

item_name = items[items['item_id'] == test_item]['description'].iloc[0] # nombre item
test_item_sales = sales[sales['item_id'] == test_item].copy()           # copia de la info
test_item_sales = test_item_sales.sort_values(by='date')


# ------------> PARA FORECASTING MENSUAL <-------------
test_item_sales = test_item_sales.set_index('date').resample('W').sum()

test_item_sales = test_item_sales[test_item_sales.index < '2023-03-15'].copy()
#print(test_item_sales)

real_item_sales = sales[sales['item_id'] == test_item].copy()  # copia de las ventas de este producto
real_item_sales = real_item_sales[real_item_sales['date'] >= '2023-03-15']  # ventas después de la fecha OJOOOOOO
real_item_sales = real_item_sales.set_index('date').resample('W').sum()  # agrupadas por mes
#print(real_item_sales)

result = adfuller(test_item_sales['quantity'])                          # para prueba de estacionariedad (Dickey-Fuller)

if result[1] > 0.05:                                                    # si no es estacionaria, diferenciar los datos
    test_item_sales['quantity'] = test_item_sales['quantity'].diff().dropna()

sarima_model = SARIMAX(test_item_sales['quantity'],                     # definir y ajustar el modelo SARIMA con los mejores parámetros encontrados
                        order=(1, 0, 2),  
                       seasonal_order=(2, 2, 2, 12))
sarima_fit = sarima_model.fit(disp=False) ##### VER VALORES DEL FIT

forecast_steps = 52                                                     # generar pronóstico para prox. 12 meses
forecast = sarima_fit.get_forecast(steps=forecast_steps)                # se hace la asignación a un frame
forecast_values = forecast.predicted_mean                               # se obtiene el pronostico
forecast_conf_int = forecast.conf_int()                                 # intervalo de confianza

# convertir las listas de predicción y confianza a dataframes para graficar (al 95%)
forecast_values = pd.Series(forecast_values, index=pd.date_range(start=test_item_sales.index[-1] + pd.DateOffset(weeks=1), periods=forecast_steps, freq='W'))

########################## IMPORTANTE: CLEME (aquí esta el df para que puedas usar los datos) ------> forecast_SARIMA

forecast_SARIMA = pd.DataFrame({
    'date': pd.date_range(start=test_item_sales.index[-1] + pd.DateOffset(weeks=1), periods=forecast_steps, freq='W'), 
    'forecast': forecast_values
})
forecast_SARIMA.reset_index(drop=True, inplace=True)
#print(forecast_SARIMA)

# Graficar los resultados
plt.figure(figsize=(10, 6))
plt.plot(test_item_sales.index, test_item_sales['quantity'], label='Ventas reales')
plt.plot(forecast_values.index, forecast_values, label='Pronóstico SARIMA', color='green')
plt.plot(real_item_sales.index, real_item_sales['quantity'], label='Ventas reales (posteriores)', color='grey', linestyle='--')
#plt.fill_between(forecast_conf_int.index, forecast_conf_int.iloc[:, 0], forecast_conf_int.iloc[:, 1], color='green', alpha=0.2)
plt.axvline(x=test_item_sales.index[-1], color='red', linestyle='--', label='Inicio de pronósticos')
plt.title(f"Pronóstico SARIMA - Item ID: {test_item}: {item_name}")
plt.xlabel('Fecha')
plt.ylabel('Cantidad vendida')
plt.legend()
plt.grid(True)
plt.show()

'''
# ESTO ME DEJO LA EMBARRADA EN EL COMPU, CUIDADO AL CORRER EN LA TERMINAL ----> Calcula los mejores parametros
p = d = q = range(0, 3)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]  # Ajuste de estacionalidad a cada 12 meses (si es mensual)

# Búsqueda del mejor modelo usando AIC
best_aic = float("inf")
best_param = None
best_seasonal_param = None
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = SARIMAX(test_item_sales['quantity'],
                          order=param,
                          seasonal_order=param_seasonal,
                          enforce_stationarity=False,
                          enforce_invertibility=False)
            results = mod.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_param = param
                best_seasonal_param = param_seasonal
        except:
            continue

print(f"Mejor AIC: {best_aic} | Orden SARIMA: {best_param} | Orden estacional: {best_seasonal_param}")
# ---->Mejor AIC: 615.5691926783063 | Orden SARIMA: (1, 0, 2) | Orden estacional: (2, 2, 2, 12)
'''