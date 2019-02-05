import papermill as pm

list_n_estimators = [10, 20, 40, 80]
list_max_edges = [10000, 20000, 40000, 80000, 160000]
list_features = ['all']

param_list = [
    {"max_edges": i, "n_estimators": j, "features": k}
    for k in list_features
    for j in list_n_estimators
    for i in list_max_edges
]

for params in param_list:
    pm.execute_notebook(
       'price_1_basic_analysis.ipynb',
       'results/price_1_basic_analysis_%s_%s_%s.ipynb' % (params['max_edges'], params['n_estimators'], params['features']),
       parameters=params
    )
