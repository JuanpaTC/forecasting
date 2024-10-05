import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared, ConstantKernel as C
from clean_data import items, sales

seed = 28
np.random.seed(seed)

########################## Producto de prueba ##########################

test_item = sales.groupby('item_id')['quantity'].sum().idxmax()  # item más vendido... por ahora se probará con ese

########################################################################

item_name = items[items['item_id'] == test_item]['description'].iloc[0]  # nombre del item
test_item_sales = sales[sales['item_id'] == test_item].copy()  # copia de las ventas de este producto
test_item_sales = test_item_sales.sort_values(by='date')  # se ordenan por fecha

# cambiar la frecuencia a semanal para agrupar las ventas por semana
test_item_sales = test_item_sales.set_index('date').resample('W').sum()
#print(test_item_sales)

test_item_sales = test_item_sales[test_item_sales.index < '2023-03-15'].copy()
#print(test_item_sales)

real_item_sales = sales[sales['item_id'] == test_item].copy()  # copia de las ventas de este producto
real_item_sales = real_item_sales[real_item_sales['date'] >= '2023-03-15']  # ventas después de la fecha

real_item_sales = real_item_sales.set_index('date').resample('W').sum()  # agrupadas por semana
#print(real_item_sales)

# definir el intervalo en semanas para la media móvil
intervalo = 2  

# calcular la media móvil
test_item_sales['media movil'] = test_item_sales['quantity'].rolling(window=intervalo).mean()

# obtener la última fecha del registro
ultima_fecha = test_item_sales.index.max()
#print(ultima_fecha)

# generar fechas futuras para el pronóstico
fechas_futuras = pd.date_range(start=ultima_fecha + pd.DateOffset(weeks=1), periods=52, freq='W')  # Pronóstico a 52 semanas

# crear DataFrame para las predicciones futuras (inicialmente vacías)
prediccion_futura = pd.DataFrame({
    'date': fechas_futuras,
    'quantity': np.nan,  
    'media movil': np.nan
})

# concatenar el histórico y las predicciones vacías
historico_y_pred = pd.concat([test_item_sales[['quantity', 'media movil']].reset_index(), prediccion_futura], ignore_index=True)

# Crear el conjunto de datos para el entrenamiento
X_train = np.array((test_item_sales.index - test_item_sales.index[0]).days).reshape(-1, 1)  # Fechas como número de días
y_train = test_item_sales['quantity'].values  # Ventas históricas

# Definir el kernel: RBF para variaciones suaves y ExpSineSquared para la periodicidad

kernel = C(1.0, (1e-4, 1e1)) * RBF(length_scale=30.0, length_scale_bounds=(1e-2, 1e2)) \
         + C(1.0, (1e-4, 1e1)) * ExpSineSquared(length_scale=30.0, periodicity=12.0, length_scale_bounds=(1e-2, 1e2))

# Crear el modelo de proceso gaussiano
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)

# Ajustar el modelo a los datos históricos
gp.fit(X_train, y_train)

# Generar fechas futuras para pronosticar
X_future = np.array((fechas_futuras - test_item_sales.index[0]).days).reshape(-1, 1)

# Hacer las predicciones (con incertidumbre)
y_pred, sigma = gp.predict(X_future, return_std=True)

# Crear DataFrame con el pronóstico
forecast_gp = pd.DataFrame({
    'date': fechas_futuras,
    'forecast': y_pred,
    'lower_bound': y_pred - 1.96 * sigma,  # Intervalo de confianza del 95%
    'upper_bound': y_pred + 1.96 * sigma
})

forecast_GP = forecast_gp.copy()

# Graficar las ventas reales, predicciones y el intervalo de confianza
plt.figure(figsize=(10, 6))
plt.plot(historico_y_pred['date'], historico_y_pred['quantity'], label='Ventas reales')
plt.plot(forecast_gp['date'], forecast_gp['forecast'], label='Pronóstico GP', color='green')
plt.fill_between(forecast_gp['date'], forecast_gp['lower_bound'], forecast_gp['upper_bound'], color='lightgreen', alpha=0.3, label='Intervalo de confianza 95%')
plt.plot(real_item_sales.index, real_item_sales['quantity'], label='Ventas reales (posteriores)', color='grey', linestyle='--')
plt.axvline(x=ultima_fecha, color='red', linestyle='--', label='Inicio de predicciones')
plt.title(f"Pronóstico con Proceso Gaussiano - Item ID: {test_item}: {item_name}")
plt.xlabel('Fecha')
plt.ylabel('Cantidad vendida')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()