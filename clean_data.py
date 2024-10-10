import pandas as pd

#################################################################################
#  CLEME Y JP, los DF con los que pueden trabajar son items, sales y purchases  #
#################################################################################


########################### DATA_ITEMS ###########################

'''
Cosas por arreglar:
    (1) Datos NA 
    (2) Atributos en plural y singular
    (3) Formato de atributos
'''

data_items = pd.read_csv('data_items.csv', delimiter=',')

descartados = data_items[(data_items['size_m3'] == 0) | (data_items['size_m3'].isna()) | 
                         (data_items['storage_cost (CLP)'].isna()) | 
                         (data_items['cost_per_purchase'].isna())]['item_id'].tolist()
items = data_items[~data_items['item_id'].isin(descartados)].copy()
#print(descartados)
#print(data_items[data_items['item_id'].isin(descartados)].copy()) ##### REVISAR

items.loc[:, 'group_description'] = items['group_description'].replace({'medicamentos': 'medicamento', 'accesorios': 'accesorio'})

format_error = ['unit_sale_price (CLP)', 'cost (CLP)', 'storage_cost (CLP)', 'stock', 'cost_per_purchase']
items[format_error] = items[format_error].astype(int)

########################### DATA_PURCHASES ###########################

'''
Cosas por arreglar:
    (1) Cantidades == 0
    (2) NO siempre hay correspondencia entre el costo descrito en esta tabla y data_items
    (3) Formato de atributos
    (4) Fechas
'''

data_purchases1 = pd.read_csv('data_purchases.csv', delimiter=';')
data_purchases2 = pd.read_csv('data_purchases_2.csv', delimiter=',')
data_purchases = pd.concat([data_purchases1, data_purchases2], ignore_index=True)

purchases = data_purchases[~data_purchases['item_id'].isin(descartados)].copy()
format_error = ['cost (CLP)']
purchases[format_error] = purchases[format_error].astype(int)

purchases['date'] = pd.to_datetime(purchases['date'], format='%Y-%m-%d')
purchases['delivery_date'] = pd.to_datetime(purchases['delivery_date'], format='%Y-%m-%d')

purchases['delivery_time (days)'] = (purchases['delivery_date'] - purchases['date']).dt.days

########################### DATA_SALES ###########################

'''
Cosas por arreglar:
    (1) Formato de atributos
    (2) Fechas
'''

data_sales = pd.read_csv('data_sales.csv', delimiter=';')

sales = data_sales[~data_sales['item_id'].isin(descartados)].copy()

format_error = ['unit_sale_price (CLP)', 'total (CLP)', 'client_id']
sales[format_error] = sales[format_error].astype(int)

sales['date'] = pd.to_datetime(sales['date'], format='%Y-%m-%d')

#items['Suma'] = items['size_m3'] * items['stock'] #Para ver el stock del archivo items
#print(items['Suma'].sum())

sales['date'] = pd.to_datetime(sales['date']) # productos descontinuados
cutoff_date = pd.Timestamp('2022-10-04')
last_sale_dates = sales.groupby('item_id')['date'].max()
discontinued_items = last_sale_dates[last_sale_dates < cutoff_date]
discontinued_item_ids = discontinued_items.index
discontinued_products = items[items['item_id'].isin(discontinued_item_ids)]
#print(discontinued_products)

items['descontinuado'] = 0
items.loc[items['item_id'].isin(discontinued_item_ids), 'descontinuado'] = 1


items = items[items['descontinuado'] == 0]
sales = sales[sales['item_id'].isin(items['item_id'])]
purchases = purchases[purchases['item_id'].isin(items['item_id'])]


#print(f"% datos perdidos:{1 - items['item_id'].count()/data_items['item_id'].count()}") # 0.17647058823529416
#print(f"% datos perdidos:{1 - purchases['item_id'].count()/data_purchases['item_id'].count()}") # 0.36868986424146566
#print(f"% datos perdidos:{1 - sales['item_id'].count()/data_sales['item_id'].count()}") # 0.7204518863551072


#############################################################################
#                  CHECHO, EMPIEZA A TRABAJAR DESDE AQUÍ                    #
#############################################################################

'''
    (1) clasificar la demanda, en el dataframe de 'items' crear columnas con los nombres: 
        [frecuencia, estacionalidad, tendencia, intermitente]
        La idea es rellenar con numeros 1, 0 o -1 (de ser el caso), 
        los criterios para cada caracteristica los defines tú (trata de tenerlos anotados)

    (2) Tratar los productos que tienen stock inicial negativo

    (3)

    (4) Solo en caso de tener tiempo (cuando termines el resto de arreglos de los archivos, 
        enfocate en esto), buscar la forma de reducir el nº de observaciones perdidas en todas las tablas
        (items, sales y purchases) sobre todo en las dos últimas
'''

sales_per_item = sales.groupby('item_id')['date'].count()

total_periods = sales['date'].nunique()  # Total de períodos únicos (días/meses)

alta_frecuencia_threshold = 0.7 * total_periods # por qué 30% y 70%
baja_frecuencia_threshold = 0.3 * total_periods

items['frecuencia'] = sales_per_item.apply(
    lambda x: 1 if x >= alta_frecuencia_threshold else (-1 if x <= baja_frecuencia_threshold else 0))

# Paso 1: Buscar productos con estacionalidad -----> mensual???? REVISAR NO ES CONFIABLE
sales['month'] = pd.to_datetime(sales['date']).dt.month
monthly_sales = sales.groupby(['item_id', 'month'])['unit_sale_price (CLP)'].sum().unstack(fill_value=0)

items['estacionalidad'] = monthly_sales.apply(lambda row: 1 if row.max() >= 2 * row.mean() else 0, axis=1)
#print(items[items['estacionalidad']==1])
# Paso 2: Tratar productos con stock inicial negativo
# Creamos la columna 'tendencia' basada en los datos de stock inicial
items['tendencia'] = items['stock'].apply(lambda x: 1 if x > 0 else 0)

# Paso 3: Definir si es intermitente
# Un producto es intermitente si tiene más de un 50% de períodos sin ventas, pero en los días que tiene ventas, estas son significativas
zero_sales_per_item = total_periods - sales_per_item  # Número de períodos sin ventas
items['intermitente'] = zero_sales_per_item.apply(lambda x: 1 if x > 0.5 * total_periods else 0)

# Mostrar el DataFrame con las nuevas columnas
#print(items[['frecuencia', 'estacionalidad', 'tendencia', 'intermitente']])

# Guardar el nuevo DataFrame con las columnas adicionales
items.to_csv('items_with_forecasting_features.csv', index=False)

# Extraer los primeros 5 productos de cada tipo de demanda basados en las columnas que hemos creado
alta_frecuencia = items[items['frecuencia'] == 1].head(5)
baja_frecuencia = items[items['frecuencia'] == -1].head(5)
continua = items[items['frecuencia'] == 0].head(5)
intermitente = items[items['intermitente'] == 1].head(5)
estacional = items[items['estacionalidad'] == 1].head(5)

# Guardar los nombres de los primeros 5 productos de cada tipo
tipos_de_demanda = {
    'Alta Frecuencia': alta_frecuencia['item_id'].tolist(),
    'Baja Frecuencia': baja_frecuencia['item_id'].tolist(),
    'Continua': continua['item_id'].tolist(),
    'Intermitente': intermitente['item_id'].tolist(),
    'Estacional': estacional['item_id'].tolist()
}

#print(tipos_de_demanda)

baja_frecuencia_items = items[items['item_id'].isin([1701, 275, 416, 1655, 247])]
continua_items = items[items['item_id'].isin([203, 1592, 208, 973, 845])]
intermitente_items = items[items['item_id'].isin([1701, 275, 416, 1655, 247])]
estacional_items = items[items['item_id'].isin([1701, 275, 416, 247, 383])]

productos_encontrados = {
    'Baja Frecuencia': baja_frecuencia_items[['item_id', 'description']].to_dict(orient='records'),
    'Continua': continua_items[['item_id', 'description']].to_dict(orient='records'),
    'Intermitente': intermitente_items[['item_id', 'description']].to_dict(orient='records'),
    'Estacional': estacional_items[['item_id', 'description']].to_dict(orient='records')
}

#print(productos_encontrados)

