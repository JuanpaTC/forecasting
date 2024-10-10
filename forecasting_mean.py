import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from clean_data import items, sales

'''
Consideraciones:
        (1) definir el intervalo de días/meses para el calculo de la MM
        (2) definir el horizonte que se quiere pronosticar
        (3) NOTAR que en el largo plazo es inexacta (no considera patrones) .... REPLANTEAR CÓMO SE IMPLEMENTA?
'''

########################## Producto de prueba ##########################

test_item = sales.groupby('item_id')['quantity'].sum().idxmax()  # item más vendido... por ahora se probará con ese

########################################################################

item_name = items[items['item_id'] == test_item]['description'].iloc[0]  # nombre del item
test_item_sales = sales[sales['item_id'] == test_item].copy()  # copia de las ventas de este producto
test_item_sales = test_item_sales.sort_values(by='date')  # se ordenan por fecha

# cambiar la frecuencia a mensual para agrupar las ventas por mes
test_item_sales = test_item_sales.set_index('date').resample('W').sum()
#print(test_item_sales)

test_item_sales = test_item_sales[test_item_sales.index < '2023-03-15'].copy()
#print(test_item_sales)

real_item_sales = sales[sales['item_id'] == test_item].copy()  # copia de las ventas de este producto
real_item_sales = real_item_sales[real_item_sales['date'] >= '2023-03-15']  # ventas después de la fecha

real_item_sales = real_item_sales.set_index('date').resample('W').sum()  # agrupadas por mes
#print(real_item_sales)

# definir el intervalo en meses para la media móvil
intervalo = 2  

# calcular la media móvil (ahora es mensual)
test_item_sales['media movil'] = test_item_sales['quantity'].rolling(window=intervalo).mean()

# obtener la última fecha del registro
ultima_fecha = test_item_sales.index.max()
#print(ultima_fecha)
# generar fechas futuras mensuales para el pronóstico
fechas_futuras = pd.date_range(start=ultima_fecha + pd.DateOffset(weeks=1), periods=52, freq='W')  # Pronóstico a 12 meses

# crear DataFrame para las predicciones futuras (inicialmente vacías)
prediccion_futura = pd.DataFrame({
    'date': fechas_futuras,
    'quantity': np.nan,  
    'media movil': np.nan
})

# concatenar el histórico y las predicciones vacías
historico_y_pred = pd.concat([test_item_sales[['quantity', 'media movil']].reset_index(), prediccion_futura], ignore_index=True)

# calcular la media móvil para los meses futuros (usando valores históricos)
for i in range(len(test_item_sales), len(historico_y_pred)):
    window_data = historico_y_pred['media movil'].iloc[i - intervalo:i].dropna()  # solo tomar valores no nulos
    if len(window_data) == intervalo:
        historico_y_pred.loc[i, 'media movil'] = window_data.mean()

####################### IMPORTANTE: CLEME (aquí está el df para que puedas usar los datos) ------> forecast_mean

forecast_mean = historico_y_pred[historico_y_pred['date'] > ultima_fecha][['date', 'media movil']].copy()
forecast_mean.columns = ['date', 'forecast']  # Renombrar las columnas

print(forecast_mean)

# grafico: ventas reales, media móvil y predicciones
plt.figure(figsize=(10, 6))
plt.plot(historico_y_pred['date'], historico_y_pred['quantity'], label='Ventas reales')
plt.plot(historico_y_pred['date'], historico_y_pred['media movil'], label='Media móvil (predicciones dinámicas)', color='orange')
plt.plot(real_item_sales.index, real_item_sales['quantity'], label='Ventas reales (posteriores)', color='grey', linestyle='--')
plt.axvline(x=ultima_fecha, color='red', linestyle='--', label='Inicio de predicciones')
plt.title(f"Ventas y Forecasting Dinámico usando Media Móvil - Item ID: {test_item}: {item_name}")
plt.xlabel('Fecha')
plt.ylabel('Cantidad vendida')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
#plt.show()
