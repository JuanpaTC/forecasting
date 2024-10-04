import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from clean_data import items, purchases, sales
'''
# Las ventas disminuyen el stock
purchases['quantity'] = purchases['quantity'].astype(int)
sales['quantity'] = -sales['quantity'].astype(int)

# Combinar compras y ventas en un solo DataFrame
events_purchases = purchases[['item_id', 'date', 'quantity']]
events_sales = sales[['item_id', 'date', 'quantity']]
events = pd.concat([events_purchases, events_sales]).sort_values(by=['item_id', 'date'])

# Rehacer el stock por cada item_id
stock_evolution = []

for item in items['item_id'].unique():
    # Filtrar eventos para cada item
    item_events = events[events['item_id'] == item].copy()
    item_events = item_events.sort_values('date')
    
    # Stock final para este item (desde la tabla items)
    stock_final = items.loc[items['item_id'] == item, 'stock'].values[0]
    
    # Inicializar el stock final
    item_events['stock'] = stock_final
    
    # Rehacer el stock hacia atrás
    for i in range(len(item_events) - 2, -1, -1):
        item_events.iloc[i, item_events.columns.get_loc('stock')] = (
            item_events.iloc[i + 1]['stock'] - item_events.iloc[i + 1]['quantity']
        )
    
    stock_evolution.append(item_events)

# Concatenar todos los resultados
final_stock_evolution = pd.concat(stock_evolution)

# Generar el gráfico para el item_id 561
item_to_plot_1 = 561 #wanpy soft duck jerky stripos
stock_data_561 = final_stock_evolution[final_stock_evolution['item_id'] == item_to_plot_1]
item_to_plot_2 = 700 #tx 2 pelotas yute catnip
stock_data_700 = final_stock_evolution[final_stock_evolution['item_id'] == item_to_plot_2]


plt.figure(figsize=(10, 6))
plt.plot(stock_data_561['date'], stock_data_561['stock'], marker='o', color='b', markersize=2, label=f'Item {item_to_plot_1}')
plt.plot(stock_data_700['date'], stock_data_700['stock'], marker='o', color='b', markersize=2, label=f'Item {item_to_plot_2}')
plt.title(f'Evolución del Stock -')
plt.xlabel('Fecha')
plt.ylabel('Stock')
plt.grid(True)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
'''

########################## Producto de prueba 1 ##########################

test_item = sales.groupby('item_id')['quantity'].sum().idxmax()

########################################################################

item_name = items[items['item_id'] == test_item]['description'].iloc[0] # nombre item
test_item_sales = sales[sales['item_id'] == test_item].copy()           # copia de la info
test_item_sales = test_item_sales.sort_values(by='date')
test_item_sales = test_item_sales.set_index('date').resample('W').sum()

plt.figure(figsize=(10, 6))
plt.plot(test_item_sales.index, test_item_sales['quantity'], label='Ventas reales')
plt.title(f"Ventas y Forecasting Dinámico usando Media Móvil - Item ID: {test_item}: {item_name}")
plt.xlabel('Fecha')
plt.ylabel('Cantidad vendida')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
#plt.show()

########################## Producto de prueba ##########################

sales['date'] = pd.to_datetime(sales['date'])
cutoff_date = pd.Timestamp('2023-03-04')
last_sale_dates = sales.groupby('item_id')['date'].max()
discontinued_items = last_sale_dates[last_sale_dates < cutoff_date]
discontinued_item_ids = discontinued_items.index
discontinued_products = items[items['item_id'].isin(discontinued_item_ids)]
print(discontinued_products)

#############################################################################
#                  CHECHO, EMPIEZA A TRABAJAR DESDE AQUÍ                    #
#############################################################################

'''
    Encontrar productos con demandas distintas (tratar de que sean lo más diversas posibles).
    Trata de tener un gráfico de cada producto, identificalo con ID y nombre.
'''
