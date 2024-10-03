import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from clean_data import items, sales

########################## Producto de prueba ##########################

test_item = sales.groupby('item_id')['quantity'].sum().idxmax()  # item más vendido... por ahora se probará con ese

########################################################################

item_name = items[items['item_id'] == test_item]['description'].iloc[0]  # nombre del item
test_item_sales = sales[sales['item_id'] == test_item].copy()  # copia de las ventas de este producto
test_item_sales = test_item_sales.sort_values(by='date')  # se ordenan por fecha

# cambiar la frecuencia a semanal para agrupar las ventas por semana
test_item_sales = test_item_sales.set_index('date').resample('W').sum()

# Datos hasta el 15 de marzo de 2023
test_item_sales = test_item_sales[test_item_sales.index < '2023-03-15'].copy()


real_item_sales = sales[sales['item_id'] == test_item].copy()  # copia de las ventas de este producto
real_item_sales = real_item_sales[real_item_sales['date'] >= '2023-03-15']  # ventas después de la fecha

real_item_sales = real_item_sales.set_index('date').resample('W').sum()



# Generar variables independientes (características)
test_item_sales['time'] = np.arange(len(test_item_sales))  # Índice de tiempo
test_item_sales['time_squared'] = test_item_sales['time'] ** 2  # Variable cuadrática
period = 13  # Aproximadamente 13 semanas (trimestral)
test_item_sales['sin_time'] = np.sin(2 * np.pi * test_item_sales['time'] / period)  # Estacionalidad
test_item_sales['cos_time'] = np.cos(2 * np.pi * test_item_sales['time'] / period)
test_item_sales['time_sin_interaction'] = test_item_sales['time'] * test_item_sales['sin_time']  # Interacción

# Preparar datos para el modelo
X_train = sm.add_constant(test_item_sales[['time', 'time_squared', 'sin_time', 'cos_time', 'time_sin_interaction']])
y_train = test_item_sales['quantity']

# Ajustar el modelo Poisson con términos adicionales
poisson_model = sm.GLM(y_train, X_train, family=sm.families.Poisson()).fit()

# Generar fechas futuras y características
fechas_futuras = pd.date_range(start=test_item_sales.index.max() + pd.DateOffset(weeks=1), periods=52, freq='W')
prediccion_futura = pd.DataFrame({'date': fechas_futuras})
prediccion_futura['time'] = np.arange(len(test_item_sales), len(test_item_sales) + len(fechas_futuras))
prediccion_futura['time_squared'] = prediccion_futura['time'] ** 2
prediccion_futura['sin_time'] = np.sin(2 * np.pi * prediccion_futura['time'] / period)
prediccion_futura['cos_time'] = np.cos(2 * np.pi * prediccion_futura['time'] / period)
prediccion_futura['time_sin_interaction'] = prediccion_futura['time'] * prediccion_futura['sin_time']

# Hacer predicciones
X_future = sm.add_constant(prediccion_futura[['time', 'time_squared', 'sin_time', 'cos_time', 'time_sin_interaction']])
y_pred = poisson_model.predict(X_future)

# Graficar
plt.figure(figsize=(10, 6))
plt.plot(test_item_sales.index, test_item_sales['quantity'], label='Ventas reales')
plt.plot(fechas_futuras, y_pred, label='Pronóstico Poisson', color='green')
plt.plot(real_item_sales.index, real_item_sales['quantity'], label='Ventas reales (posteriores)', color='grey', linestyle='--')
plt.axvline(x=test_item_sales.index.max(), color='red', linestyle='--', label='Inicio de predicciones')
plt.title(f"Pronóstico con Regresión de Poisson Mejorada - Item ID: {test_item}: {item_name}")
plt.xlabel('Fecha')
plt.ylabel('Cantidad vendida')
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.show()