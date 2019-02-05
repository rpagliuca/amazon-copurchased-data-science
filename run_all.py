import papermill as pm

# Experiment 1 - Price - Execution time

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

# Experiment 2 - Price - Features

list_n_estimators = [20]
list_max_edges = [0]
list_features = ['none', 'all', 'all_except_network_metrics']

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

# Experiment 3 - Link - Features

for params in param_list:
    pm.execute_notebook(
       'predict_links.ipynb',
       'results/predict_links_%s_%s_%s.ipynb' % (params['max_edges'], params['n_estimators'], params['features']),
       parameters=params
    )
