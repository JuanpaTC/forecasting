import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from clean_data import items, sales
from sklearn.metrics import mean_squared_error

########################## Producto de prueba ##########################

test_item = sales.groupby('item_id')['quantity'].sum().idxmax()  # Producto más vendido

########################################################################

# Obtener datos del producto
item_name = items[items['item_id'] == test_item]['description'].iloc[0]  # Nombre del item
test_item_sales = sales[sales['item_id'] == test_item].copy()  # Copia de las ventas
test_item_sales = test_item_sales.sort_values(by='date')  # Ordenar por fecha

# Cambiar frecuencia a semanal
test_item_sales = test_item_sales.set_index('date').resample('W').sum().reset_index()
test_item_sales = test_item_sales[test_item_sales['date'] < '2023-03-15'].copy()

real_item_sales = sales[sales['item_id'] == test_item].copy()  # Copia de las ventas
real_item_sales = real_item_sales.sort_values(by='date')  # Ordenar por fecha
real_item_sales = real_item_sales.set_index('date').resample('W').sum().reset_index()
real_item_sales = real_item_sales[real_item_sales['date'] >= '2023-03-15'].copy()

# Crear características tiempo ---> mes, año, semana del año, y tendencia
test_item_sales['month'] = test_item_sales['date'].dt.month
test_item_sales['year'] = test_item_sales['date'].dt.year
test_item_sales['week_of_year'] = test_item_sales['date'].dt.isocalendar().week
test_item_sales['trend'] = np.arange(len(test_item_sales))  # Tendencia

# Definir X e y
X = test_item_sales[['month', 'year', 'week_of_year', 'trend']]
y = test_item_sales['quantity']

# División entrenamiento/prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Entrenar el modelo Árbol de Decisión
dt_model = DecisionTreeRegressor(max_depth=5)
dt_model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = dt_model.predict(X_test)

# Calcular y mostrar el error cuadrático medio
#mse = mean_squared_error(y_test, y_pred)
#print(f'Error cuadrático medio en el conjunto de prueba: {mse:.2f}')

# predicción final
ultima_fecha = test_item_sales['date'].max()
fechas_futuras = pd.date_range(start=ultima_fecha + pd.DateOffset(weeks=1), periods=52, freq='W')

# Crear DataFrame de las futuras fechas con características temporales
fechas_futuras_df = pd.DataFrame({
    'date': fechas_futuras,
    'month': fechas_futuras.month,
    'year': fechas_futuras.year,
    'week_of_year': fechas_futuras.isocalendar().week,
    'trend': np.arange(len(test_item_sales), len(test_item_sales) + len(fechas_futuras))  # Tendencia futura
})

# Pronosticar las cantidades futuras
forecast_dt = dt_model.predict(fechas_futuras_df[['month', 'year', 'week_of_year', 'trend']])
forecast_DT = pd.DataFrame({'date': fechas_futuras, 'forecast': forecast_dt})

# Gráfico: ventas reales y pronóstico con Árboles de Decisión
plt.figure(figsize=(10, 6))
plt.plot(test_item_sales['date'], test_item_sales['quantity'], label='Ventas reales')
plt.plot(forecast_DT['date'], forecast_DT['forecast'], label='Pronóstico Árbol de Decisión', color='green')
plt.plot(real_item_sales['date'], real_item_sales['quantity'], label='Ventas reales (posteriores)', color='grey', linestyle='--')
plt.axvline(x=ultima_fecha, color='red', linestyle='--', label='Inicio de predicciones')
plt.title(f"Pronóstico Árbol de Decisión - Item ID: {test_item}: {item_name}")
plt.xlabel('Fecha')
plt.ylabel('Cantidad vendida')
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.show()
