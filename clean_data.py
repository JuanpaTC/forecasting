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

# 
discontinued_with_stock = items[(items['descontinuado'] == 1) & (items['stock'] != 0)]
num_discontinued_with_stock = discontinued_with_stock.shape[0]
print(f"Cantidad de productos descontinuados que aún ocupan espacio en bodega: {num_discontinued_with_stock}")


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


