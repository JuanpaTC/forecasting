from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from forecasting_xgboost import forecast_XGB, real_item_sales
from forecasting_tree import forecast_DT
from forecasting_mean import forecast_mean
from forecasting_ets import forecast_ETS
from forecasting_sarima import forecast_SARIMA
from forecasting_gaussian import forecast_GP


################################################
# INCLUIR UN GRAFICO CON TODOS LOS PRONOSTICOS #
################################################

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


#############################################################################
#                  CHECHO, EMPIEZA A TRABAJAR DESDE AQUÍ                    #
#############################################################################

'''
    Revisar las metricas, cualquier duda pregúntame, trata de hacer un print para que se vea como una tabla. 
    Incluír las que dijo el profe.
    Se agregan 2 metodos más, Gaussiano y Poisson (pero yo aun sigo trabajando en elos asi que solo mencionalos).
'''



def percentage_bias(y_true, y_pred):
    return 100 * np.sum(y_pred - y_true) / np.sum(y_true)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.where(y_true == 0, np.finfo(float).eps, y_true)  # Reemplazar ceros por epsilon (valor muy pequeño)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def hit_rate(y_true, y_pred, tolerance=0.2):
    y_true = np.where(y_true == 0, np.finfo(float).eps, y_true)  # Reemplazar ceros por epsilon
    hits = np.abs(y_true - y_pred) / y_true <= tolerance
    return np.mean(hits)


print("\n   MAE:")
print(f"\tXGBOOST  --> {mean_absolute_error(real_item_sales['quantity'].values, forecast_XGB['forecast'].values)}")
print(f"\tDT       --> {mean_absolute_error(real_item_sales['quantity'].values, forecast_DT['forecast'].values)}")
print(f"\tM. AV.   --> {mean_absolute_error(real_item_sales['quantity'].values, forecast_mean['forecast'].values)}")
print(f"\tSARIMA   --> {mean_absolute_error(real_item_sales['quantity'].values, forecast_SARIMA['forecast'].values)}")
print(f"\tETS      --> {mean_absolute_error(real_item_sales['quantity'].values, forecast_ETS['forecast'].values)}")
print(f"\tGAUSSIAN --> {mean_absolute_error(real_item_sales['quantity'].values, forecast_GP['forecast'].values)}")

print("\n   MAPE:")
print(f"\tXGBOOST  --> {mean_absolute_percentage_error(real_item_sales['quantity'].values, forecast_XGB['forecast'].values)}%")
print(f"\tDT       --> {mean_absolute_percentage_error(real_item_sales['quantity'].values, forecast_DT['forecast'].values)}%")
print(f"\tM. AV.   --> {mean_absolute_percentage_error(real_item_sales['quantity'].values, forecast_mean['forecast'].values)}%")
print(f"\tSARIMA   --> {mean_absolute_percentage_error(real_item_sales['quantity'].values, forecast_SARIMA['forecast'].values)}%")
print(f"\tETS      --> {mean_absolute_percentage_error(real_item_sales['quantity'].values, forecast_ETS['forecast'].values)}%")
print(f"\tGAUSSIAN --> {mean_absolute_percentage_error(real_item_sales['quantity'].values, forecast_GP['forecast'].values)}")


print("\n   PB:")
print(f"\tXGBOOST  --> {percentage_bias(real_item_sales['quantity'].values, forecast_XGB['forecast'].values)}%")
print(f"\tDT       --> {percentage_bias(real_item_sales['quantity'].values, forecast_DT['forecast'].values)}%")
print(f"\tM. AV.   --> {percentage_bias(real_item_sales['quantity'].values, forecast_mean['forecast'].values)}%")
print(f"\tSARIMA   --> {percentage_bias(real_item_sales['quantity'].values, forecast_SARIMA['forecast'].values)}%")
print(f"\tETS      --> {percentage_bias(real_item_sales['quantity'].values, forecast_ETS['forecast'].values)}%")
print(f"\tGAUSSIAN --> {percentage_bias(real_item_sales['quantity'].values, forecast_GP['forecast'].values)}")


print("\n   HR:")
print(f"\tXGBOOST  --> {hit_rate(real_item_sales['quantity'].values, forecast_XGB['forecast'].values)}%")
print(f"\tDT       --> {hit_rate(real_item_sales['quantity'].values, forecast_DT['forecast'].values)}%")
print(f"\tM. AV.   --> {hit_rate(real_item_sales['quantity'].values, forecast_mean['forecast'].values)}%")
print(f"\tSARIMA   --> {hit_rate(real_item_sales['quantity'].values, forecast_SARIMA['forecast'].values)}%")
print(f"\tETS      --> {hit_rate(real_item_sales['quantity'].values, forecast_ETS['forecast'].values)}%")
print(f"\tGAUSSIAN --> {hit_rate(real_item_sales['quantity'].values, forecast_GP['forecast'].values)}")


print("\n")
#print(real_item_sales[['date','quantity']])

#print(forecast_XGB)
#print(forecast_DT)
#print(forecast_mean)
#print(forecast_SARIMA)
#print(forecast_ETS)

