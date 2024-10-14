import pandas as pd

#################################################################################
#  CLEME Y JP, los DF con los que pueden trabajar son items, sales y purchases  #
#################################################################################


########################### DATA_ITEMS ###########################

data_items = pd.read_csv('data_items.csv', delimiter=',')
'''
print(data_items[data_items['item_id']==1790])
descartados = data_items[(data_items['size_m3'] == 0) | (data_items['size_m3'].isna()) | 
                         (data_items['storage_cost (CLP)'].isna()) | 
                         (data_items['cost_per_purchase'].isna())]['item_id'].tolist()
items = data_items[~data_items['item_id'].isin(descartados)].copy()
'''

items = data_items.copy()
items.loc[:, 'group_description'] = items['group_description'].replace({'medicamentos': 'medicamento', 'accesorios': 'accesorio'})

########################### DATA_PURCHASES ###########################

data_purchases1 = pd.read_csv('data_purchases.csv', delimiter=';')
data_purchases2 = pd.read_csv('data_purchases_2.csv', delimiter=',')
data_purchases = pd.concat([data_purchases1, data_purchases2], ignore_index=True)

purchases = data_purchases.copy()

########################### DATA_SALES ###########################


data_sales = pd.read_csv('data_sales.csv', delimiter=';')
#print(data_sales[data_sales['item_id']==1790])

#sales = data_sales[~data_sales['item_id'].isin(descartados)].copy()
sales = data_sales.copy()

#items['Suma'] = items['size_m3'] * items['stock'] #Para ver el stock del archivo items
#print(items['Suma'].sum())

######################## Se sacan los productos descontinuados ########################

sales['date'] = pd.to_datetime(sales['date']) # productos descontinuados
cutoff_date = pd.Timestamp('2022-10-04') # son descontinuados aquellos que no se venden hace casi 14 meses
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

####################### Se recupera la inf de costo y precio de venta!!!! ####################### en promedio hay 23 valores por col que no se pudieron recuperar
'''
print(items.head())
print(items.columns.values)
print(sales.head())
print(purchases.head())
'''
purchases_no_duplicates = purchases.groupby('item_id').agg({
    'cost (CLP)': 'mean'  # O puedes usar 'max', 'min' según el caso
}).reset_index()

columns_to_replace = ['cost (CLP)']
for col in columns_to_replace:
    items[col] = items[col].fillna(items['item_id'].map(purchases_no_duplicates.set_index('item_id')[col]))

sales_no_duplicates = sales.groupby('item_id').agg({
    'unit_sale_price (CLP)': 'mean' # O puedes usar 'max', 'min' según el caso
}).reset_index()

columns_to_replace = ['unit_sale_price (CLP)']
for col in columns_to_replace:
    # Usamos .map() para obtener los valores de purchases que coinciden con item_id
    items[col] = items[col].fillna(items['item_id'].map(sales_no_duplicates.set_index('item_id')[col]))

####################### Falta recuperar inf. sobre storage_cost, size:m3 y cost_per_purchase ####################### 

columns_to_fill = ['storage_cost (CLP)', 'size_m3', 'cost_per_purchase']

for col in columns_to_fill:
    # Calcular el promedio de cada columna por categoría
    category_means = items.groupby('group_description')[col].transform('mean')
    
    # Reemplazar valores NaN con el promedio de la categoría correspondiente
    items[col] = items[col].fillna(category_means)

#print(items.isna().sum()) # hay 147 productos sin categoría, se podría poner a mano pero es mucho trabajo... se dejará por minetras como el promedio de toda la tienda para todas las columnas

columns_to_fill = ['storage_cost (CLP)', 'size_m3', 'cost_per_purchase', 'unit_sale_price (CLP)', 'cost (CLP)']
overall_means = items[columns_to_fill].mean()
for col in columns_to_fill:
    # Filtrar los productos sin categoría (group_description == NaN)
    items.loc[items['group_description'].isna(), col] = items.loc[items['group_description'].isna(), col].fillna(overall_means[col])

items['stock'] = items['stock'].fillna(0) # se supondrá que no hay productos..... REVISAR

#print(items.isna().sum())


############################### AHORA SE CAMBIAN LOS FORMATOS ###############################

format_error = ['unit_sale_price (CLP)', 'cost (CLP)', 'storage_cost (CLP)', 'stock', 'cost_per_purchase']
items[format_error] = items[format_error].astype(int)

format_error = ['cost (CLP)']
purchases[format_error] = purchases[format_error].astype(int)
# purchases['date'] = pd.to_datetime(purchases['date'], infer_datetime_format=True)
# purchases['delivery_date'] = pd.to_datetime(purchases['delivery_date'], infer_datetime_format=True)
# purchases['delivery_time (days)'] = (purchases['delivery_date'] - purchases['date']).dt.days

format_error = ['unit_sale_price (CLP)', 'total (CLP)']
sales[format_error] = sales[format_error].astype(int)
sales['date'] = pd.to_datetime(sales['date'], format='%Y-%m-%d')

############################### Añadir la columna 'lead_time_promedio' en la tabla 'items' ###############################
def calcular_lead_time_promedio(items, purchases):
    lead_time_promedio = purchases.groupby('item_id')['delivery_time (days)'].mean().reset_index()
    lead_time_promedio['delivery_time (days)'] = lead_time_promedio['delivery_time (days)'].fillna(0)
    lead_time_promedio = lead_time_promedio.rename(columns={'delivery_time (days)': 'lead_time_promedio'})
    items = items.merge(lead_time_promedio, on='item_id', how='left')
    items['lead_time_promedio'] = items['lead_time_promedio'].fillna(0)
    return items

# items = calcular_lead_time_promedio(items, purchases) # es un float

#print(purchases[purchases['quantity']==0]) ######### muchas compras por 0 productos????? por el momento no es relevante.
