import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from clean_data import items, sales

#### OJO CON LOS VALORES NEGATIVOS ####

########################## Producto de prueba ##########################

test_item = sales.groupby('item_id')['quantity'].sum().idxmax()

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

# Crear características temporales (mes, año, semana del año, tendencia)
test_item_sales['month'] = test_item_sales['date'].dt.month
test_item_sales['year'] = test_item_sales['date'].dt.year
test_item_sales['week_of_year'] = test_item_sales['date'].dt.isocalendar().week.astype(int)  # Convertir a entero

# Definir características (X) y target (y)
X = test_item_sales[['month', 'year', 'week_of_year']]
y = test_item_sales['quantity']

# Separar los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Entrenar el modelo XGBoost
xgb_model = XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.05)
xgb_model.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = xgb_model.predict(X_test)

# Calcular y mostrar el error cuadrático medio
mse = mean_squared_error(y_test, y_pred)
print(f'Error cuadrático medio en el conjunto de prueba: {mse:.2f}')

# Predecir para el próximo año
ultima_fecha = test_item_sales['date'].max()
fechas_futuras = pd.date_range(start=ultima_fecha + pd.DateOffset(weeks=1), periods=52, freq='W')

# Crear DataFrame de las futuras fechas con características temporales
fechas_futuras_df = pd.DataFrame({
    'date': fechas_futuras,
    'month': fechas_futuras.month,
    'year': fechas_futuras.year,
    'week_of_year': fechas_futuras.isocalendar().week.astype(int),  # Convertir a entero
})

# Pronosticar las cantidades futuras
forecast_xgb = xgb_model.predict(fechas_futuras_df[['month', 'year', 'week_of_year']])
forecast_XGB = pd.DataFrame({'date': fechas_futuras, 'forecast': forecast_xgb})

#print(real_item_sales)
#print(forecast_XGB)

# Gráfico: ventas reales y pronóstico de XGBoost
plt.figure(figsize=(10, 6))
plt.plot(test_item_sales['date'], test_item_sales['quantity'], label='Ventas reales')
plt.plot(forecast_XGB['date'], forecast_XGB['forecast'], label='Pronóstico XGBoost', color='green')
plt.plot(real_item_sales['date'], real_item_sales['quantity'], label='Ventas reales (posteriores)', color='grey', linestyle='--')
plt.axvline(x=ultima_fecha, color='red', linestyle='--', label='Inicio de predicciones')
plt.title(f"Pronóstico XGBoost - Item ID: {test_item}: {item_name}")
plt.xlabel('Fecha')
plt.ylabel('Cantidad vendida')
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.show()
