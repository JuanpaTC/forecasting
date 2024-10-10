import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from clean_data import items, sales

# Lista de productos
lista_productos = items['item_id'].unique().tolist()  # Ejemplo de lista de productos

# Definir el horizonte que se quiere pronosticar
horizonte_prediccion = 52  # número de semanas para el pronóstico (1 año)
intervalo = 2  # intervalo en semanas para la media móvil
ultima_fecha = sales['date'].max()
#print(ultima_fecha)

# DataFrame para almacenar todas las predicciones
forecast_basecase = pd.DataFrame()

# Iterar sobre la lista de productos
for test_item in lista_productos:

    # Nombre del producto
    item_name = items[items['item_id'] == test_item]['description'].iloc[0]  # nombre del item

    # Filtrar las ventas para este producto y ordenar por fecha
    test_item_sales = sales[sales['item_id'] == test_item].copy()
    test_item_sales = test_item_sales.sort_values(by='date')

    # Resamplear a frecuencia semanal
    test_item_sales = test_item_sales.set_index('date').resample('W').sum()

    # Tomar las ventas anteriores a la fecha de corte
    test_item_sales = test_item_sales[test_item_sales.index < '2024-03-04'].copy()

    # Calcular la media móvil
    test_item_sales['media movil'] = test_item_sales['quantity'].rolling(window=intervalo).mean()

    # Obtener la última fecha del registro

    # Generar las fechas futuras para las próximas 13 semanas
    fechas_futuras = pd.date_range(start=ultima_fecha + pd.DateOffset(weeks=0), periods=horizonte_prediccion, freq='W')

    # Crear DataFrame para las predicciones futuras
    prediccion_futura = pd.DataFrame({
        'date': fechas_futuras,
        'quantity': np.nan,
        'media movil': np.nan
    })

    # Concatenar el histórico con las predicciones vacías
    historico_y_pred = pd.concat([test_item_sales[['quantity', 'media movil']].reset_index(), prediccion_futura], ignore_index=True)

    # Calcular la media móvil para las semanas futuras
    for i in range(len(test_item_sales), len(historico_y_pred)):
        window_data = historico_y_pred['media movil'].iloc[i - intervalo:i].dropna()  # solo tomar valores no nulos
        if len(window_data) == intervalo:
            historico_y_pred.loc[i, 'media movil'] = window_data.mean()

    # Crear un DataFrame con las predicciones de este producto
    forecast_mean = historico_y_pred[historico_y_pred['date'] > ultima_fecha][['date', 'media movil']].copy()
    forecast_mean.columns = ['date', 'forecast']  # Renombrar las columnas

    # Añadir el id del producto para identificar las predicciones
    forecast_mean['item_id'] = test_item

    # Concatenar al DataFrame total
    forecast_basecase = pd.concat([forecast_basecase, forecast_mean], ignore_index=True)

# Arreglar datos NaN y NULL
forecast_basecase['forecast'] = forecast_basecase['forecast'].fillna(0)


####################### forecast_basecase

####################### PARA VISUALIZAR o si quieren trabajar desde un archivo ###############################

#forecast_basecase.to_csv('forecast_basecase.csv', index=False)
