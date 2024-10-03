import pandas as pd

################################################################################
# IMPORTANTE: AUN NO SE TRATAN LOS PRODUCTOS QUE TIENEN STOCK INICIAL NEGATIVO #
################################################################################

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

#print(f"% datos perdidos:{1 - items['item_id'].count()/data_items['item_id'].count()}") #cerca del 12.8%

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

#print(f"% datos perdidos:{1 - purchases['item_id'].count()/data_purchases['item_id'].count()}") #cerca del 34.7%

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


#print(f"% datos perdidos:{1 - sales['item_id'].count()/data_sales['item_id'].count()}") #cerca del 71%
#items['Suma'] = items['size_m3'] * items['stock'] #Para ver el stock del archivo items
#print(items['Suma'].sum())
