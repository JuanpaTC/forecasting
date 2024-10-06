'''
Para no perder tantas observaciones en tu análisis de ventas, podemos aplicar varias estrategias. Algunas posibles mejoras incluyen el manejo de los valores faltantes y la revisión de los umbrales utilizados para descartar productos. Aquí te propongo varias estrategias que puedes seguir para reducir el porcentaje de ventas perdidas y aprovechar la mayor cantidad de datos posible:

### Estrategias a aplicar:

1. **Imputación de datos faltantes**:
    En lugar de descartar productos con valores nulos en las columnas `size_m3`, `storage_cost (CLP)` y `cost_per_purchase`, puedes considerar imputar estos valores. Existen varios métodos para imputar valores faltantes:
    
    - **Promedio o mediana**: Imputar valores faltantes con la media o mediana de los valores conocidos en la columna.
    - **Imputación basada en categorías**: Si los productos pertenecen a diferentes grupos (`group_description`), puedes imputar los valores faltantes en función de los promedios o medianas dentro de cada grupo.
    - **Modelos predictivos**: Usar modelos como regresión para predecir los valores faltantes basados en otras características del producto.
   
   Ejemplo de código para imputar con la mediana:

   ```python
   items['size_m3'].fillna(items['size_m3'].median(), inplace=True)
   items['storage_cost (CLP)'].fillna(items['storage_cost (CLP)'].median(), inplace=True)
   items['cost_per_purchase'].fillna(items['cost_per_purchase'].median(), inplace=True)
   ```

2. **Revisar los umbrales para descartar productos**:
   En lugar de descartar todos los productos que tienen un `size_m3` igual a cero, podrías permitir aquellos productos con `size_m3` igual a cero pero que tienen otros atributos válidos. Por ejemplo, podrías ajustar el criterio a:

   ```python
   descartados = data_items[(data_items['storage_cost (CLP)'].isna()) | 
                            (data_items['cost_per_purchase'].isna())]['item_id'].tolist()
   ```

   Esto reduce el número de productos descartados en función de un solo atributo en lugar de múltiples.

3. **Mantener productos con ventas pero sin otros datos**:
   Algunos productos pueden tener información faltante en `size_m3` o costos, pero es posible que tengas ventas importantes. Podrías mantener esos productos, imputando los datos faltantes o asignándoles un valor por defecto.

4. **Revisar las fechas de ventas y productos descontinuados**:
   En lugar de filtrar todos los productos cuya última venta fue antes de una fecha límite, podrías ajustar este filtro considerando productos con ventas intermitentes o estacionales. Por ejemplo, podrías ampliar el `cutoff_date` para incluir productos que se vendieron hace más tiempo pero aún tienen valor en el inventario.

   ```python
   # Ajustar la fecha de corte o eliminar este filtro temporalmente para productos con ventas intermitentes
   cutoff_date = pd.Timestamp('2022-01-01')  # Puedes ampliar este rango si los productos aún son relevantes.
   ```

5. **Incluir productos con bajo stock o ventas intermitentes**:
   Si algunos productos se venden solo de forma intermitente, podrías considerar no descartarlos, especialmente si el análisis de ventas depende de productos con patrones de ventas irregulares pero importantes. Mantén estos productos con ventas bajas o esporádicas para mejorar el análisis de tendencias.

6. **Verificar si los productos descartados tienen ventas importantes**:
   Revisa los productos que has descartado para ver si tienen un volumen de ventas significativo. Si encuentras productos con ventas importantes que fueron descartados por errores en los atributos, podrías reconsiderar su inclusión en el análisis imputando o corrigiendo esos atributos.

   ```python
   discarded_sales = sales[sales['item_id'].isin(descartados)]
   discarded_sales_summary = discarded_sales.groupby('item_id')['total (CLP)'].sum()
   print(discarded_sales_summary.sort_values(ascending=False).head(10))  # Revisar los productos con ventas más importantes
   ```

### Aplicación de algunas de estas estrategias:

```python
# 1. Imputar valores faltantes
items['size_m3'].fillna(items['size_m3'].median(), inplace=True)
items['storage_cost (CLP)'].fillna(items['storage_cost (CLP)'].median(), inplace=True)
items['cost_per_purchase'].fillna(items['cost_per_purchase'].median(), inplace=True)

# 2. Revisar umbrales para descartar productos, solo descartar si ambos datos críticos son NaN
descartados = data_items[(data_items['storage_cost (CLP)'].isna()) & 
                         (data_items['cost_per_purchase'].isna())]['item_id'].tolist()

# Filtrar los productos de las tablas sales y purchases
items = data_items[~data_items['item_id'].isin(descartados)].copy()

# Revisar si los productos descartados tienen ventas importantes
discarded_sales = sales[sales['item_id'].isin(descartados)]
discarded_sales_summary = discarded_sales.groupby('item_id')['total (CLP)'].sum()
print(discarded_sales_summary.sort_values(ascending=False).head(10))

# Ajustar fechas de corte para productos descontinuados
cutoff_date = pd.Timestamp('2022-01-01')  # Ampliar el rango si es necesario
```

### Resultado esperado:
- **Reducción del porcentaje de datos perdidos**: Con la imputación de datos faltantes y el ajuste de umbrales de descarte, deberías observar una reducción significativa en el número de productos descartados y, por tanto, en las ventas perdidas.
- **Mejora en la calidad del análisis**: Al incluir más productos con ventas significativas, tu análisis será más completo y representará mejor las tendencias reales de ventas e inventario.

Estas estrategias te ayudarán a conservar más datos y a hacer un análisis más robusto.

'''

