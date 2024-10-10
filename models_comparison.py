from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from forecasting_xgboost import forecast_XGB, real_item_sales
from forecasting_tree import forecast_DT
from forecasting_mean import forecast_mean
from forecasting_ets import forecast_ETS
from forecasting_sarima import forecast_SARIMA
from forecasting_gaussian import forecast_GP
from forecasting_poisson import forecast_P
from forecasting_croston import forecast_croston


################################################
# INCLUIR UN GRAFICO CON TODOS LOS PRONOSTICOS #
################################################
'''
# Graficar las ventas reales
plt.figure(figsize=(12, 8))
plt.plot(real_item_sales['date'], real_item_sales['quantity'], label='Ventas reales', color='black', linewidth=2)

# Graficar los pronósticos de cada modelo
plt.plot(forecast_XGB['date'], forecast_XGB['forecast'], label='Pronóstico XGBoost', linestyle='--')
plt.plot(forecast_DT['date'], forecast_DT['forecast'], label='Pronóstico Árbol de Decisión', linestyle='--')
plt.plot(forecast_mean['date'], forecast_mean['forecast'], label='Pronóstico Media Móvil', linestyle='--')
plt.plot(forecast_ETS['date'], forecast_ETS['forecast'], label='Pronóstico ETS', linestyle='--')
plt.plot(forecast_SARIMA['date'], forecast_SARIMA['forecast'], label='Pronóstico SARIMA', linestyle='--')
plt.plot(forecast_GP['date'], forecast_GP['forecast'], label='Pronóstico Proceso Gaussiano', linestyle='--')

# Añadir leyenda y etiquetas
plt.title('Comparación de Pronósticos de Ventas - Diferentes Modelos')
plt.xlabel('Fecha')
plt.ylabel('Cantidad Vendida')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

# Mostrar el gráfico
plt.show()
'''

def mean_absolute_deviation(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def tracking_signal(y_true, y_pred):
    cumulative_error = np.cumsum(y_pred - y_true)
    mad = mean_absolute_deviation(y_true, y_pred)
    return cumulative_error / mad

# Resto de las funciones de métricas
def percentage_bias(y_true, y_pred):
    return 100 * np.sum(y_pred - y_true) / np.sum(y_true)

def mean_absolute_percentage_error(y_true, y_pred):
    mask = y_true != 0  # Ignorar los valores donde y_true es 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def hit_rate(y_true, y_pred, tolerance=0.2):
    y_true = np.where(y_true == 0, np.finfo(float).eps, y_true)
    hits = np.abs(y_true - y_pred) / y_true <= tolerance
    return np.mean(hits)

metricas = {
    'MAE': [],
    'MAPE': [],
    'PB': [],
    'HR': [],
    'MAD': [],
    'Tracking Signal': []
}

# Lista con los nombres de los modelos
modelos = ['XGBOOST', 'DECISION TREE', 'MOVING AVERAGE', 'SARIMA', 'ETS - HW', 'GAUSSIAN PROCESS', 'POISSON', 'CROSTON']

# Cálculo de las métricas para cada modelo
for modelo, forecast in zip(modelos, [forecast_XGB, forecast_DT, forecast_mean, forecast_SARIMA, forecast_ETS, forecast_GP, forecast_P, forecast_croston]):
    y_true = real_item_sales['quantity'].values
    y_pred = forecast['forecast'].values
    
    # Calcular cada métrica y agregarla al diccionario
    metricas['MAE'].append(mean_absolute_error(y_true, y_pred))
    metricas['MAPE'].append(mean_absolute_percentage_error(y_true, y_pred))
    metricas['PB'].append(percentage_bias(y_true, y_pred))
    metricas['HR'].append(hit_rate(y_true, y_pred))
    metricas['MAD'].append(mean_absolute_deviation(y_true, y_pred))
    metricas['Tracking Signal'].append(np.mean(tracking_signal(y_true, y_pred)))  # Promedio de la señal de rastreo


df_metricas = pd.DataFrame(metricas, index=modelos)


df_metricas.to_csv('metrics_561.csv')

print(df_metricas)