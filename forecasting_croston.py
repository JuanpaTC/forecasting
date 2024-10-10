import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from clean_data import items, sales

# Método de Croston con variabilidad añadida
def croston(ts, alpha=0.1):
    """
    Implementación del método de Croston con un pequeño componente de variabilidad.
    ts: Serie temporal de demanda (pandas Series).
    alpha: Parámetro de suavización (0 < alpha < 1).
    """
    demand = ts.values
    n = len(demand)
    
    # Inicializar los niveles de demanda y períodos
    forecast = np.zeros(n)
    demand_level = 0
    period_level = 0
    period = 1  # el primer período se cuenta como 1 para evitar dividir por 0
    
    for i in range(n):
        if demand[i] > 0:  # Cuando hay una venta
            if demand_level == 0:
                demand_level = demand[i]  # Primer valor no nulo inicializa el nivel de demanda
            else:
                demand_level = alpha * demand[i] + (1 - alpha) * demand_level  # Actualización exponencial
                
            period_level = alpha * period + (1 - alpha) * period_level  # Actualización del nivel de períodos
            period = 1  # Reiniciar período
        else:
            period += 1  # Incrementar período si no hay ventas
        
        forecast[i] = demand_level / period_level  # Pronóstico de demanda
    
    return pd.Series(forecast, index=ts.index)

########################## Producto de prueba ##########################

test_item = sales.groupby('item_id')['quantity'].sum().idxmax()  # item más vendido

########################################################################

item_name = items[items['item_id'] == test_item]['description'].iloc[0]  # nombre del item
test_item_sales = sales[sales['item_id'] == test_item].copy()  # copia de las ventas de este producto
test_item_sales = test_item_sales.sort_values(by='date')  # se ordenan por fecha

# cambiar la frecuencia a semanal
test_item_sales = test_item_sales.set_index('date').resample('W').sum()

# Tomar datos históricos para el cálculo
test_item_sales = test_item_sales[test_item_sales.index < '2023-03-15'].copy()

real_item_sales = sales[sales['item_id'] == test_item].copy()  # copia de las ventas de este producto
real_item_sales = real_item_sales[real_item_sales['date'] >= '2023-03-15']

real_item_sales = real_item_sales.set_index('date').resample('W').sum()

# Aplicar el método de Croston
test_item_sales['croston_forecast'] = croston(test_item_sales['quantity'])

# Obtener la última fecha del registro
ultima_fecha = test_item_sales.index.max()

# generar fechas futuras para el pronóstico
fechas_futuras = pd.date_range(start=ultima_fecha + pd.DateOffset(weeks=1), periods=52, freq='W')

# Crear DataFrame para predicciones futuras vacías
prediccion_futura = pd.DataFrame({
    'date': fechas_futuras,
    'quantity': np.nan,  
    'croston_forecast': np.nan
})

# Concatenar histórico y predicciones futuras
historico_y_pred = pd.concat([test_item_sales[['quantity', 'croston_forecast']].reset_index(), prediccion_futura], ignore_index=True)

# Usar el último pronóstico de Croston y agregar algo de variabilidad
ultimo_valor_croston = test_item_sales['croston_forecast'].iloc[-1]

# Agregar variabilidad aleatoria a las predicciones futuras
variabilidad = np.random.normal(loc=0, scale=ultimo_valor_croston * 0.1, size=len(fechas_futuras))
historico_y_pred.loc[len(test_item_sales):, 'croston_forecast'] = ultimo_valor_croston + variabilidad

forecast_croston = historico_y_pred[historico_y_pred['date'] > ultima_fecha][['date', 'croston_forecast']].copy()
forecast_croston.columns = ['date', 'forecast']

# Graficar ventas reales y predicciones de Croston
plt.figure(figsize=(10, 6))
plt.plot(historico_y_pred['date'], historico_y_pred['quantity'], label='Ventas reales')
plt.plot(historico_y_pred['date'], historico_y_pred['croston_forecast'], label='Croston (predicciones)', color='orange')
plt.plot(real_item_sales.index, real_item_sales['quantity'], label='Ventas reales (posteriores)', color='grey', linestyle='--')
plt.axvline(x=ultima_fecha, color='red', linestyle='--', label='Inicio de predicciones')
plt.title(f"Ventas y Forecasting Dinámico usando Croston - Item ID: {test_item}: {item_name}")
plt.xlabel('Fecha')
plt.ylabel('Cantidad vendida')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
#plt.show()
