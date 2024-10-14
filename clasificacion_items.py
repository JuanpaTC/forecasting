import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from clean_data import items, sales

###############################  1a categoria: alta, baja o frecuencia media ###############################

# Calcular la cantidad total de ventas por producto
ventas_por_item = sales.groupby('item_id')['quantity'].sum().reset_index()
ventas_por_item.columns = ['item_id', 'total_ventas']

# Definir umbrales para categorizar los productos
# Usamos percentiles para agrupar los productos
umbral_mas_vendidos = ventas_por_item['total_ventas'].quantile(0.943)
umbral_menos_vendidos = ventas_por_item['total_ventas'].quantile(0.869)

# Crear una nueva columna para categorizar los productos
ventas_por_item['categoria'] = 'medianamente vendidos'  # Por defecto

# Los más vendidos son aquellos con ventas mayores al percentil 75
ventas_por_item.loc[ventas_por_item['total_ventas'] >= umbral_mas_vendidos, 'categoria'] = 'más vendidos'

# Los menos vendidos son aquellos con ventas menores al percentil 25
ventas_por_item.loc[ventas_por_item['total_ventas'] <= umbral_menos_vendidos, 'categoria'] = 'menos vendidos'

# Combinar esta información con el DataFrame de items
items = items.merge(ventas_por_item[['item_id', 'categoria', 'total_ventas']], on='item_id', how='left')

# Mostrar el resultado
print(items[['item_id', 'total_ventas', 'categoria', ]].head())

# Crear DataFrames separados para cada categoría
items_mas_vendidos = items[items['categoria'] == 'más vendidos']
items_medianamente_vendidos = items[items['categoria'] == 'medianamente vendidos']
items_menos_vendidos = items[items['categoria'] == 'menos vendidos']

# Mostrar los productos clasificados
print("\nMás vendidos:")
print(items_mas_vendidos[['item_id', 'total_ventas', 'categoria']].head())

print("\nMedianamente vendidos:")
print(items_medianamente_vendidos[['item_id', 'total_ventas', 'categoria']].head())

print("\nMenos vendidos:")
print(items_menos_vendidos[['item_id', 'total_ventas', 'categoria']].head())

# Reemplazar los valores NaN (aquellos productos que no están en ninguna categoría) con una nueva categoría "sin ventas"
items['categoria'] = items['categoria'].fillna('sin ventas')

# Filtrar los productos que están clasificados como "sin ventas"
items_sin_ventas = items[items['categoria'] == 'sin ventas']

# Mostrar los productos que no pertenecen a ninguna de las otras categorías
print("\nSin ventas:")
print(items_sin_ventas[['item_id', 'categoria']].head())

# Calcular el total de productos por categoría
total_por_categoria = items['categoria'].value_counts().reset_index()
total_por_categoria.columns = ['categoria', 'total_items']

# Mostrar el total de ítems por categoría
print("\nCategorias:")
print(total_por_categoria)

# Función para obtener el producto más y menos vendido por categoría
def obtener_mas_menos_vendido_por_categoria(categoria):
    # Filtrar los productos por categoría
    productos_categoria = items[items['categoria'] == categoria]
    
    # Obtener el producto más vendido
    producto_mas_vendido = productos_categoria.loc[productos_categoria['total_ventas'].idxmax()]
    
    # Obtener el producto menos vendido (excluyendo productos sin ventas, si deseas incluirlos, quita el filtro)
    producto_menos_vendido = productos_categoria[productos_categoria['total_ventas'] > 0].loc[productos_categoria['total_ventas'].idxmin()]
    
    return producto_mas_vendido, producto_menos_vendido

# Obtener los productos más y menos vendidos para cada categoría
categorias = ['más vendidos', 'medianamente vendidos', 'menos vendidos']

# for categoria in categorias:
#     print(f"\nCategoría: {categoria}")
#     producto_mas_vendido, producto_menos_vendido = obtener_mas_menos_vendido_por_categoria(categoria)
    
#     print(f"Producto más vendido: \n{producto_mas_vendido[['item_id', 'total_ventas']]}")
#     print(f"Producto menos vendido: \n{producto_menos_vendido[['item_id', 'total_ventas']]}")

# Seleccionar un producto de cada categoría
producto_mas_vendido = items[items['categoria'] == 'más vendidos']['item_id'].iloc[0]
producto_medianamente_vendido = items[items['categoria'] == 'medianamente vendidos']['item_id'].iloc[0]
producto_menos_vendido = items[items['categoria'] == 'menos vendidos']['item_id'].iloc[0]

# Filtrar las ventas de cada producto y agrupar por mes
ventas_mas_vendido = sales[sales['item_id'] == producto_mas_vendido].groupby(sales['date'].dt.to_period('M'))['quantity'].sum().reset_index()
ventas_medianamente_vendido = sales[sales['item_id'] == producto_medianamente_vendido].groupby(sales['date'].dt.to_period('M'))['quantity'].sum().reset_index()
ventas_menos_vendido = sales[sales['item_id'] == producto_menos_vendido].groupby(sales['date'].dt.to_period('M'))['quantity'].sum().reset_index()

# Convertir el período a timestamps para graficar correctamente
ventas_mas_vendido['date'] = ventas_mas_vendido['date'].dt.to_timestamp()
ventas_medianamente_vendido['date'] = ventas_medianamente_vendido['date'].dt.to_timestamp()
ventas_menos_vendido['date'] = ventas_menos_vendido['date'].dt.to_timestamp()

# Agregar una columna para la categoría
ventas_mas_vendido['categoria'] = 'más vendido'
ventas_medianamente_vendido['categoria'] = 'medianamente vendido'
ventas_menos_vendido['categoria'] = 'menos vendido'

# Combinar los tres DataFrames en uno solo para graficar
ventas_combined = pd.concat([ventas_mas_vendido, ventas_medianamente_vendido, ventas_menos_vendido])

# # Configurar el estilo del gráfico
# sns.set(style="whitegrid")

# # Graficar las ventas de los tres productos en un gráfico de líneas agrupado por mes
# plt.figure(figsize=(12, 6))
# sns.lineplot(x='date', y='quantity', hue='categoria', data=ventas_combined, marker='o')

# # Añadir título y etiquetas
# plt.title('Comportamiento de las ventas mensuales para tres productos de diferentes categorías')
# plt.xlabel('Fecha (Mes)')
# plt.ylabel('Cantidad Vendida')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

###############################  # 2a categoria: con estacionalidad marcada o no ###############################

# Asegúrate de que las fechas estén en formato datetime
sales['date'] = pd.to_datetime(sales['date'])

# Agrupar las ventas por fecha y producto, sumando las cantidades vendidas
ventas_por_item_fecha = sales.groupby(['item_id', 'date'])['quantity'].sum().reset_index()

# Función para determinar si un ítem tiene estacionalidad o no
def tiene_estacionalidad(producto, period):
    # Reindexar para asegurarse de que tengamos fechas completas
    producto = producto.set_index('date').asfreq('D', fill_value=0)
    
    try:
        # Descomponer la serie temporal en componentes
        decomposition = seasonal_decompose(producto['quantity'], model='additive', period=period)
        
        # Evaluamos la componente estacional: si la desviación estándar es mayor a un umbral, consideramos que tiene estacionalidad
        estacionalidad = decomposition.seasonal
        if estacionalidad.std() > 0.1:  # Umbral ajustable
            return True
        else:
            return False
    except:
        # Si la descomposición falla (ejemplo, no hay suficientes datos), asumimos que no hay estacionalidad
        return False


# Crear una función para clasificar los productos por estacionalidad
def clasificar_estacionalidad(items_categoria, categoria_nombre):
    items_categoria = items_categoria.copy()  # Asegurarnos de que trabajamos sobre una copia para evitar el warning
    items_categoria['estacionalidad_mensual'] = np.nan
    items_categoria['estacionalidad_semanal'] = np.nan
    
    for item_id in items_categoria['item_id'].unique():
        # Filtrar las ventas del producto actual por fecha
        producto = ventas_por_item_fecha[ventas_por_item_fecha['item_id'] == item_id]
        
        # Verificar si el producto tiene estacionalidad mensual
        if tiene_estacionalidad(producto, period=30):
            items_categoria.loc[items_categoria['item_id'] == item_id, 'estacionalidad_mensual'] = 'tiene estacionalidad'
        
        else:
            items_categoria.loc[items_categoria['item_id'] == item_id, 'estacionalidad_mensual'] = 'no tiene estacionalidad'

        # Verificar si el producto tiene estacionalidad semanal
        if tiene_estacionalidad(producto, period=7):
            items_categoria.loc[items_categoria['item_id'] == item_id, 'estacionalidad_semanal'] = 'tiene estacionalidad'
        
        else:
            items_categoria.loc[items_categoria['item_id'] == item_id, 'estacionalidad_semanal'] = 'no tiene estacionalidad'

    # # Exportar la tabla clasificada a un archivo CSV
    # file_name = f'items_{categoria_nombre}_con_estacionalidad.csv'
    # items_categoria.to_csv(file_name, index=False)
    
    return items_categoria

# Clasificar estacionalidad para cada categoría y guardar los resultados
items_alta_frecuencia = clasificar_estacionalidad(items_mas_vendidos, 'alta_frecuencia')
items_mediana_frecuencia = clasificar_estacionalidad(items_medianamente_vendidos, 'mediana_frecuencia')
items_baja_frecuencia = clasificar_estacionalidad(items_menos_vendidos, 'baja_frecuencia')

# Mostrar las tablas de cada categoría con estacionalidad
print("\nItems de alta frecuencia con estacionalidad:")
print(items_alta_frecuencia[['item_id', 'estacionalidad_mensual', 'estacionalidad_semanal']].head())
# print(items_alta_frecuencia[items_alta_frecuencia['estacionalidad_mensual'] == 'tiene estacionalidad'])
# print(items_alta_frecuencia[items_alta_frecuencia['estacionalidad_semanal'] == 'tiene estacionalidad'])

print("\nItems de mediana frecuencia con estacionalidad:")
print(items_mediana_frecuencia[['item_id', 'estacionalidad_mensual', 'estacionalidad_semanal']].head())
# print(items_mediana_frecuencia[items_mediana_frecuencia['estacionalidad_mensual'] == 'tiene estacionalidad'])
# print(items_mediana_frecuencia[items_mediana_frecuencia['estacionalidad_semanal'] == 'tiene estacionalidad'])

print("\nItems de baja frecuencia con estacionalidad:")
print(items_baja_frecuencia[['item_id', 'estacionalidad_mensual', 'estacionalidad_semanal']].head())
# print(items_baja_frecuencia[items_baja_frecuencia['estacionalidad_mensual'] == 'tiene estacionalidad'])
# print(items_baja_frecuencia[items_baja_frecuencia['estacionalidad_semanal'] == 'tiene estacionalidad'])

# Función para graficar ventas mensuales y semanales de un producto
def graficar_ventas(producto_id, categoria, frecuencia, titulo):
    # Filtrar las ventas del producto por item_id
    producto = ventas_por_item_fecha[ventas_por_item_fecha['item_id'] == producto_id].copy()
    
    # Reindexar por fecha para asegurarnos de que tenemos todas las fechas y rellenar días sin ventas con 0
    producto = producto.set_index('date').asfreq('D', fill_value=0)
    
    # Agrupar por frecuencia (Mensual o Semanal)
    if frecuencia == 'mensual':
        producto_agrupado = producto.resample('M').sum()  # Agrupar por meses
        xlabel = 'Mes'
    elif frecuencia == 'semanal':
        producto_agrupado = producto.resample('W').sum()  # Agrupar por semanas
        xlabel = 'Semana'
    
    # Graficar las ventas
    plt.figure(figsize=(12, 6))
    plt.plot(producto_agrupado.index, producto_agrupado['quantity'], marker='o', label='Ventas')
    plt.title(f'Ventas {titulo} para el producto {producto_id} en la categoría {categoria}')
    plt.xlabel(xlabel)
    plt.ylabel('Ventas')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Seleccionar un producto de cada categoría
producto_alta_frecuencia = items_alta_frecuencia['item_id'].iloc[0]  # Primer producto de alta frecuencia
producto_mediana_frecuencia = items_mediana_frecuencia['item_id'].iloc[0]  # Primer producto de mediana frecuencia
producto_baja_frecuencia = items_baja_frecuencia['item_id'].iloc[0]  # Primer producto de baja frecuencia

# Graficar ventas mensuales y semanales para cada categoría

# Alta frecuencia
# graficar_ventas(producto_alta_frecuencia, 'alta frecuencia', 'mensual', 'mensuales')
# graficar_ventas(producto_alta_frecuencia, 'alta frecuencia', 'semanal', 'semanales')

# # Mediana frecuencia
# graficar_ventas(producto_mediana_frecuencia, 'mediana frecuencia', 'mensual', 'mensuales')
# graficar_ventas(producto_mediana_frecuencia, 'mediana frecuencia', 'semanal', 'semanales')

# # Baja frecuencia
# graficar_ventas(producto_baja_frecuencia, 'baja frecuencia', 'mensual', 'mensuales')
# graficar_ventas(producto_baja_frecuencia, 'baja frecuencia', 'semanal', 'semanales')

###############################  # 3a categoria: tendencia positiva, negativa o sin tendencia. ###############################

# Función para determinar la tendencia de un producto
def determinar_tendencia(producto, period=30):
    # Reindexar para asegurarse de que tengamos fechas completas
    producto = producto.set_index('date').asfreq('D', fill_value=0)
    
    try:
        # Descomponer la serie temporal en componentes
        decomposition = seasonal_decompose(producto['quantity'], model='additive', period=period)
        
        # Extraer la componente de tendencia
        tendencia = decomposition.trend.dropna()
        
        # Verificar la pendiente de la tendencia
        if tendencia.iloc[-1] > tendencia.iloc[0]:
            return 'positiva'
        elif tendencia.iloc[-1] < tendencia.iloc[0]:
            return 'negativa'
        else:
            return 'no tiene'
    except:
        # Si la descomposición falla (ejemplo, no hay suficientes datos), asumimos que no hay tendencia
        return 'no tiene'

# Crear una función para clasificar los productos por tendencia
def clasificar_tendencia(items_categoria, categoria_nombre):
    items_categoria = items_categoria.copy()  # Trabajamos sobre una copia para evitar modificaciones no deseadas
    items_categoria['tendencia'] = np.nan
    
    for item_id in items_categoria['item_id'].unique():
        # Filtrar las ventas del producto actual por fecha
        producto = ventas_por_item_fecha[ventas_por_item_fecha['item_id'] == item_id]
        
        # Determinar la tendencia del producto
        tendencia = determinar_tendencia(producto, period=30)
        
        # Asignar la tendencia al producto
        items_categoria.loc[items_categoria['item_id'] == item_id, 'tendencia'] = tendencia
    
    return items_categoria

# Clasificar tendencia para cada categoría y guardar los resultados
items_alta_frecuencia_tendencia = clasificar_tendencia(items_alta_frecuencia, 'alta_frecuencia')
items_mediana_frecuencia_tendencia = clasificar_tendencia(items_mediana_frecuencia, 'mediana_frecuencia')
items_baja_frecuencia_tendencia = clasificar_tendencia(items_baja_frecuencia, 'baja_frecuencia')

# Mostrar las tablas de cada categoría con la clasificación de tendencia
print("\nItems de alta frecuencia con tendencia:")
print(items_alta_frecuencia_tendencia[['item_id', 'tendencia']].head())

print("\nItems de mediana frecuencia con tendencia:")
print(items_mediana_frecuencia_tendencia[['item_id', 'tendencia']].head())

print("\nItems de baja frecuencia con tendencia:")
print(items_baja_frecuencia_tendencia[['item_id', 'tendencia']].head())

###############################  # 4a categoria: si tiene intermitencia o no. ###############################

# Función para determinar si un producto tiene intermitencia
def determinar_intermitencia(producto):
    # Reindexar para asegurarse de que tengamos fechas completas
    producto = producto.set_index('date').asfreq('D', fill_value=0)
    
    # Calcular la proporción de días con ventas
    dias_con_ventas = (producto['quantity'] > 0).sum()
    total_dias = len(producto)
    
    # Calcular el porcentaje de días con ventas
    porcentaje_dias_con_ventas = dias_con_ventas / total_dias
    
    # Definir un umbral para intermitencia (por ejemplo, menos del 50% de los días tienen ventas)
    if porcentaje_dias_con_ventas < 0.5:
        return 'intermitente'
    else:
        return 'no intermitente'

# Crear una función para clasificar los productos por intermitencia
def clasificar_intermitencia(items_categoria, categoria_nombre):
    items_categoria = items_categoria.copy()  # Trabajamos sobre una copia para evitar modificaciones no deseadas
    items_categoria['intermitencia'] = np.nan
    
    for item_id in items_categoria['item_id'].unique():
        # Filtrar las ventas del producto actual por fecha
        producto = ventas_por_item_fecha[ventas_por_item_fecha['item_id'] == item_id]
        
        # Determinar la intermitencia del producto
        intermitencia = determinar_intermitencia(producto)
        
        # Asignar la intermitencia al producto
        items_categoria.loc[items_categoria['item_id'] == item_id, 'intermitencia'] = intermitencia
    
    return items_categoria

# Clasificar intermitencia para cada categoría y guardar los resultados
items_alta = clasificar_intermitencia(items_alta_frecuencia_tendencia, 'alta_frecuencia')
items_mediana = clasificar_intermitencia(items_mediana_frecuencia_tendencia, 'mediana_frecuencia')
items_baja = clasificar_intermitencia(items_baja_frecuencia_tendencia, 'baja_frecuencia')

# Mostrar las tablas de cada categoría con la clasificación de intermitencia
print("\nItems de alta frecuencia con intermitencia:")
print(items_alta[['item_id', 'intermitencia']].head())

print("\nItems de mediana frecuencia con intermitencia:")
print(items_mediana[['item_id', 'intermitencia']].head())

print("\nItems de baja frecuencia con intermitencia:")
print(items_baja[['item_id', 'intermitencia']].head())

print("\nClasificacion items alta demanda")
print(items_alta[['item_id', 'estacionalidad_mensual', 'estacionalidad_semanal', 'tendencia', 'intermitencia']])
print("\nnClasificacion items mediana demanda")
print(items_mediana[['item_id', 'estacionalidad_mensual', 'estacionalidad_semanal', 'tendencia', 'intermitencia']])
print("\nnClasificacion items baja demanda")
print(items_baja[['item_id', 'estacionalidad_mensual', 'estacionalidad_semanal', 'tendencia', 'intermitencia']])