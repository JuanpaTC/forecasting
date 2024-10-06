import pandas as pd

########################### DATA_ITEMS ###########################

data_items = pd.read_csv('data_items.csv', delimiter=',')

# Filtrado de productos descartados por tamaño o costo NA
descartados = data_items[(data_items['size_m3'] == 0) | (data_items['size_m3'].isna()) | 
                         (data_items['storage_cost (CLP)'].isna()) | 
                         (data_items['cost_per_purchase'].isna())]['item_id'].tolist()
items = data_items[~data_items['item_id'].isin(descartados)].copy()

# Corrección de nombres de descripciones y formatos
items['group_description'] = items['group_description'].replace({'medicamentos': 'medicamento', 'accesorios': 'accesorio'})
format_error = ['unit_sale_price (CLP)', 'cost (CLP)', 'storage_cost (CLP)', 'stock', 'cost_per_purchase']
items[format_error] = items[format_error].astype(int)

########################### DATA_SALES ###########################

data_sales = pd.read_csv('data_sales.csv', delimiter=';')

# Filtrar ventas que no estén en productos descartados
sales = data_sales[~data_sales['item_id'].isin(descartados)].copy()

# Formatear columnas y convertir fechas
sales['unit_sale_price (CLP)'] = sales['unit_sale_price (CLP)'].astype(int)
sales['date'] = pd.to_datetime(sales['date'], format='%Y-%m-%d')

# Agrupar ventas por producto
sales_per_item = sales.groupby('item_id')['date'].count()
total_periods = sales['date'].nunique()  # Cantidad de períodos (días o meses) en los datos

# Definir umbrales para alta, media y baja frecuencia
alta_frecuencia_threshold = 0.7 * total_periods
baja_frecuencia_threshold = 0.3 * total_periods

# Clasificar productos según frecuencia de ventas
items['frecuencia'] = sales_per_item.apply(
    lambda x: 1 if x >= alta_frecuencia_threshold else (-1 if x <= baja_frecuencia_threshold else 0))

# Paso 1: Estacionalidad (ventas significativamente más altas en algunos meses)
sales['month'] = pd.to_datetime(sales['date']).dt.month
monthly_sales = sales.groupby(['item_id', 'month'])['unit_sale_price (CLP)'].sum().unstack(fill_value=0)
items['estacionalidad'] = monthly_sales.apply(lambda row: 1 if row.max() >= 2 * row.mean() else 0, axis=1)

# Paso 2: Tendencia basada en el stock inicial (productos con stock inicial negativo)
items['tendencia'] = items['stock'].apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0)

# Paso 3: Intermitencia (productos con ventas esporádicas)
zero_sales_per_item = total_periods - sales_per_item  # Periodos sin ventas
items['intermitente'] = zero_sales_per_item.apply(lambda x: 1 if x > 0.5 * total_periods else 0)

# Mostrar el DataFrame con las nuevas columnas
print(items[['item_id', 'frecuencia', 'estacionalidad', 'tendencia', 'intermitente']])

# Guardar el nuevo DataFrame con las columnas adicionales
items.to_csv('items_with_forecasting_features.csv', index=False)
