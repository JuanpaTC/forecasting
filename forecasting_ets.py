import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from clean_data import items, sales
from statsmodels.tsa.holtwinters import ExponentialSmoothing

########################## Producto de prueba ##########################

test_item = sales.groupby('item_id')['quantity'].sum().idxmax()

########################################################################

item_name = items[items['item_id'] == test_item]['description'].iloc[0] # nombre item
test_item_sales = sales[sales['item_id'] == test_item].copy()           # copia de la info
test_item_sales = test_item_sales.sort_values(by='date')

# Resamplear los datos a frecuencia mensual o diaria según lo que prefieras
test_item_sales = test_item_sales.set_index('date').resample('W').sum()
#print(test_item_sales)
test_item_sales = test_item_sales[test_item_sales.index < '2023-03-15'].copy()

real_item_sales = sales[sales['item_id'] == test_item].copy()  # copia de las ventas de este producto
real_item_sales = real_item_sales[real_item_sales['date'] >= '2023-03-15']  # ventas después de la fecha FIJARS EN ESTO
real_item_sales = real_item_sales.set_index('date').resample('W').sum()  # agrupadas por mes
#print(real_item_sales)

# Definir el modelo ETS
ets_model = ExponentialSmoothing(test_item_sales['quantity'], 
                                 trend='add',  # 'add' para tendencia aditiva, también puede ser 'mul' para multiplicativa (?)
                                 seasonal='add',  # Estacionalidad aditiva, puede ser 'mul' para multiplicativa
                                 seasonal_periods=52)  # Aquí se ajusta a la estacionalidad anual (12 meses)

# Ajustar el modelo
ets_fit = ets_model.fit()

# Generar pronóstico
forecast_steps = 52  # Número de periodos futuros a predecir (por ejemplo, 12 meses)
forecast_ets = ets_fit.forecast(steps=forecast_steps)

# Crear DataFrame para pronóstico
forecast_ETS = pd.DataFrame({
    'date': pd.date_range(start=test_item_sales.index[-1] + pd.DateOffset(weeks=1), periods=forecast_steps, freq='W'),
    'forecast': forecast_ets
})
forecast_ETS.reset_index(drop=True, inplace=True)
#print(forecast_ETS)

# Graficar ventas reales y pronóstico
plt.figure(figsize=(10, 6))
plt.plot(test_item_sales.index, test_item_sales['quantity'], label='Ventas reales')
plt.plot(forecast_ets.index, forecast_ets, label='Pronóstico ETS', color='blue')
plt.plot(real_item_sales.index, real_item_sales['quantity'], label='Ventas reales (posteriores)', color='grey', linestyle='--')
plt.axvline(x=test_item_sales.index[-1], color='red', linestyle='--', label='Inicio de pronósticos')
plt.title(f"Pronóstico ETS - Item ID: {test_item}: {item_name}")
plt.xlabel('Fecha')
plt.ylabel('Cantidad vendida')
plt.legend()
plt.grid(True)
#plt.show()
