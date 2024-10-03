from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from forecasting_xgboost import forecast_XGB, real_item_sales
from forecasting_tree import forecast_DT
from forecasting_mean import forecast_mean
from forecasting_ets import forecast_ETS
from forecasting_sarima import forecast_SARIMA

################################################
# INCLUIR UN GRAFICO CON TODOS LOS PRONOSTICOS #
################################################


def percentage_bias(y_true, y_pred):
    return 100 * np.sum(y_pred - y_true) / np.sum(y_true)

def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.where(y_true == 0, np.finfo(float).eps, y_true)  # Reemplazar ceros por epsilon (valor muy pequeño)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def hit_rate(y_true, y_pred, tolerance=0.2):
    y_true = np.where(y_true == 0, np.finfo(float).eps, y_true)  # Reemplazar ceros por epsilon
    hits = np.abs(y_true - y_pred) / y_true <= tolerance
    return np.mean(hits)


print("\n   MAE:")
print(f"\tXGBOOST  --> {mean_absolute_error(real_item_sales['quantity'].values, forecast_XGB['forecast'].values)}")
print(f"\tDT       --> {mean_absolute_error(real_item_sales['quantity'].values, forecast_DT['forecast'].values)}")
print(f"\tM. AV.   --> {mean_absolute_error(real_item_sales['quantity'].values, forecast_mean['forecast'].values)}")
print(f"\tSARIMA   --> {mean_absolute_error(real_item_sales['quantity'].values, forecast_SARIMA['forecast'].values)}")
print(f"\tETS      --> {mean_absolute_error(real_item_sales['quantity'].values, forecast_ETS['forecast'].values)}")

print("\n   MAPE:")
print(f"\tXGBOOST  --> {mean_absolute_percentage_error(real_item_sales['quantity'].values, forecast_XGB['forecast'].values)}%")
print(f"\tDT       --> {mean_absolute_percentage_error(real_item_sales['quantity'].values, forecast_DT['forecast'].values)}%")
print(f"\tM. AV.   --> {mean_absolute_percentage_error(real_item_sales['quantity'].values, forecast_mean['forecast'].values)}%")
print(f"\tSARIMA   --> {mean_absolute_percentage_error(real_item_sales['quantity'].values, forecast_SARIMA['forecast'].values)}%")
print(f"\tETS      --> {mean_absolute_percentage_error(real_item_sales['quantity'].values, forecast_ETS['forecast'].values)}%")

print("\n   PB:")
print(f"\tXGBOOST  --> {percentage_bias(real_item_sales['quantity'].values, forecast_XGB['forecast'].values)}%")
print(f"\tDT       --> {percentage_bias(real_item_sales['quantity'].values, forecast_DT['forecast'].values)}%")
print(f"\tM. AV.   --> {percentage_bias(real_item_sales['quantity'].values, forecast_mean['forecast'].values)}%")
print(f"\tSARIMA   --> {percentage_bias(real_item_sales['quantity'].values, forecast_SARIMA['forecast'].values)}%")
print(f"\tETS      --> {percentage_bias(real_item_sales['quantity'].values, forecast_ETS['forecast'].values)}%")

print("\n   HR:")
print(f"\tXGBOOST  --> {hit_rate(real_item_sales['quantity'].values, forecast_XGB['forecast'].values)}%")
print(f"\tDT       --> {hit_rate(real_item_sales['quantity'].values, forecast_DT['forecast'].values)}%")
print(f"\tM. AV.   --> {hit_rate(real_item_sales['quantity'].values, forecast_mean['forecast'].values)}%")
print(f"\tSARIMA   --> {hit_rate(real_item_sales['quantity'].values, forecast_SARIMA['forecast'].values)}%")
print(f"\tETS      --> {hit_rate(real_item_sales['quantity'].values, forecast_ETS['forecast'].values)}%")

print("\n")
#print(real_item_sales[['date','quantity']])

#print(forecast_XGB)
#print(forecast_DT)
#print(forecast_mean)
#print(forecast_SARIMA)
#print(forecast_ETS)