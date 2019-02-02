# -*- coding: utf-8 -*-

from __future__ import division
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import itertools
import networkx as nx
import pandas as pd
import hashlib
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_validate
from sklearn.tree import export_graphviz
import pydot
import test_data

def plot_confusion_matrix(cm, classes, std=False,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        normalization = cm.sum(axis=1)[:, np.newaxis]
        cm = cm.astype('float') / normalization
        if std is not False:
            std = std.astype('float') / normalization
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = (cm.max() + cm.min()) / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if std is not False:
            text = '%.2f\n(%.2f)' % (cm[i, j], std[i, j])
        else:
            if normalize:
                text = '%.2f' % (cm[i, j])
            else:
                text = '%i' % (cm[i, j])
        plt.text(j, i, text,
                 horizontalalignment="center",
                 verticalalignment='center',
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Classe verdadeira')
    plt.xlabel('Classe prevista')

def read_G(edges_csv_file, max_edges):
    lines = []
    total_line_count = 0
    with open(edges_csv_file, 'rb') as f:
        f.readline()   # skip first line / header
        while True:
            line = f.readline()
            if not line:
                break
            if not (max_edges > 0 and len(lines) >= max_edges):
                lines.append(line)
            total_line_count += 1
    G = nx.parse_edgelist(lines, delimiter=',', nodetype=int)
    print "Using %d edges out of %d available (%.2f%% of data)" % (len(lines), total_line_count, len(lines)/total_line_count * 100)
    return G

# Add columns to dataframe
def merge_columns(dataframe, data):
    df = dataframe.copy()
    for col in data:
        rows = []
        for item in data[col].items():
            rows.append({"id": item[0], col: item[1]})
        df = df.merge(pd.DataFrame(rows))
    return df

def centrality_measures(G):
    centrality_measures = {}
    centrality_measures["degree"] = nx.degree(G)
    centrality_measures["eigenvector_centrality"] = nx.eigenvector_centrality_numpy(G)
    centrality_measures["betweenness_centrality"] = nx.approximate_current_flow_betweenness_centrality(G)
    # Very slow!
    #centrality_measures["closeness_centrality"] = nx.closeness_centrality(G)
    #centrality_measures["betweenness_centrality"] = nx.betweenness_centrality(G)
    return centrality_measures

def add_sha256_column_from_id(df):
    sha256_ids = df['id'].map(lambda x: int(hashlib.sha256(str(x)).hexdigest()[0:8], 16))
    df = merge_columns(df, {'sha256_id': sha256_ids})
    return df

def accumulated_distribution(histogram):
    accumulated = []
    total = 0.0
    for value in histogram:
        total += float(value)
        accumulated.append(total)
    return accumulated

def prepare_data(df, numeric_features):
    df = df.replace("<<MISSING_DATA>>", np.NaN)
    df[numeric_features] = df[numeric_features].apply(pd.to_numeric)
    df[['price']] = df[['price']].apply(pd.to_numeric)
    for feature in numeric_features:
        df[feature].fillna(df[feature].mean(), inplace = True)
    return df

def prepare_datasets(df, numeric_features, categorical_features, target_column):

    print "Numeric features: ", numeric_features
    print "Categorical features: ", categorical_features
    print "Target column: ", target_column

    id = ['sha256_id'] if 'sha256_id' in df else []

    df_with_dummies = pd.get_dummies(
            df[id + numeric_features + categorical_features + [target_column]],
            columns=categorical_features,
            drop_first=True
    )
    feature_list = list(df_with_dummies.drop(columns = [target_column]))

    features_values = np.array(df_with_dummies.drop(columns = [target_column]))
    target_value = np.array(df_with_dummies[target_column])

    features_and_target = np.concatenate((features_values, target_value.reshape(len(target_value), 1)), axis = 1)

    # Shuffle features_and_target
    np.random.shuffle(features_and_target)

    limit = int(round(len(target_value) * 0.8))
    print "Test percentage: ", (len(target_value)-limit)/len(target_value)

    features = features_and_target[:limit, :-1]
    target = features_and_target[:limit, -1]

    test_features = features_and_target[limit:, :-1]
    test_target = features_and_target[limit: , -1]

    print "Train features shape: ", features.shape
    print "Train target shape: ", target.shape
    print "Test features shape: ", test_features.shape
    print "Test target shape: ", test_target.shape

    return [target, features, feature_list, test_features, test_target]

def plot_relative_error_distribution(predicted_df):
    sets = [
        [u'Modelo', predicted_df.error_relative],
        [u'Baseline 1 - Média', predicted_df.error_baseline_relative],
        [u'Baseline 2 - Mediana', predicted_df.error_baseline_median_relative]
    ]
    centers, hists = plot_histogram(sets, xlabel=u'Erro relativo', ymax=2)
    return [centers, hists[0], hists[1], hists[2]]

def plot_accumulated_relative_error(centers, normalized_hist_predicted, normalized_hist_baseline, normalized_hist_baseline_median):
    plt.figure(figsize=(8,8), dpi=130)
    
    plt.plot(centers, accumulated_distribution(normalized_hist_predicted), marker='o', label=u'Modelo')
    plt.plot(centers, accumulated_distribution(normalized_hist_baseline), marker='o', label=u'Baseline 1 - Média')
    plt.plot(centers, accumulated_distribution(normalized_hist_baseline_median), marker='o', label=u'Baseline 2 - Mediana')

    plt.xlabel(u'Erro relativo')
    plt.ylabel(u'Frequência relativa acumulada')
    plt.legend(loc='lower right')
    plt.show()

def print_score_summary(scores):
    print "=== Relative"
    print "RF relative abs mean: ", np.mean(np.abs(scores['test_relative']))
    print "RF relative abs std: ", np.std(scores['test_relative'])

    #print "Baseline: ", scores['test_baseline']
    print "Baseline (average) relative mean: ", np.mean(scores['test_baseline_relative'])
    print "Baseline (average) relative std: ", np.std(scores['test_baseline_relative'])

    print "Baseline (median) relative mean: ", np.mean(scores['test_baseline_median_relative'])
    print "Baseline (median) relative std: ", np.std(scores['test_baseline_median_relative'])

    print "=== Absolute"

    #print "Abs: ", scores['test_abs']
    print "RF abs mean: ", np.mean(np.abs(scores['test_abs']))
    print "RF abs std: ", np.std(scores['test_abs'])

    #print "Baseline: ", scores['test_baseline']
    print "Baseline (average) mean: ", np.mean(scores['test_baseline'])
    print "Baseline (average) std: ", np.std(scores['test_baseline'])

    #print "Baseline: ", scores['test_baseline']
    print "Baseline (median) mean: ", np.mean(scores['test_baseline_median'])
    print "Baseline (median) std: ", np.std(scores['test_baseline_median'])

def run_cross_validation_classification(features, target):

    X = features
    y = target

    average_target = 1
    group_counts = []

    def class_weight(label):
        return len([i for i in target if i == label])

    class_weights = {
        0: (class_weight(1)/len(target))*1E0,
        1: (class_weight(0)/len(target))*1E7
    }

    cv = sklearn.model_selection.StratifiedKFold(n_splits=10)
    splits = list(cv.split(X, y))

    rf = RandomForestClassifier(n_estimators = 500, class_weight=class_weights, n_jobs = -1)
    scores = cross_validate(estimator=rf, X=features, y=target, cv=splits,
                            scoring = {
                                'abs': 'neg_mean_absolute_error',
                            },
                            return_train_score=False, return_estimator = True)

    return [scores['estimator'], splits, scores]

def run_cross_validation_regression(features, target):

    X = features
    y = target
    average_target = np.average(target)
    median_target = np.median(target)

    def baseline_score_function (y_true, y_pred):
        errors_baseline = abs(average_target - y_true)
        return np.mean(errors_baseline)

    def baseline_median_error_function (y_true, y_pred):
        errors_baseline = abs(median_target - y_true)
        return np.mean(errors_baseline)

    def relative_error_function (y_true, y_pred):
        errors_absolute = abs(y_pred - y_true)
        errors_relative = errors_absolute / y_true
        return np.mean(errors_relative)

    def baseline_relative_error_function (y_true, y_pred):
        errors_absolute = abs(average_target - y_true)
        errors_relative = errors_absolute / y_true
        return np.mean(errors_relative)

    def baseline_median_relative_error_function (y_true, y_pred):
        errors_absolute = abs(median_target - y_true)
        errors_relative = errors_absolute / y_true
        return np.mean(errors_relative)


    rf = RandomForestRegressor(n_estimators = 500, n_jobs = -1)

    scorer = make_scorer(baseline_score_function)
    scorer2 = make_scorer(relative_error_function)
    scorer3 = make_scorer(baseline_relative_error_function)
    scorer4 = make_scorer(baseline_median_relative_error_function)
    scorer5 = make_scorer(baseline_median_error_function)

    cv = sklearn.model_selection.KFold(n_splits=10)
    splits = list(cv.split(X, y))

    scores = cross_validate(estimator=rf, X=features, y=target, cv=splits,
                            scoring = {
                                'abs': 'neg_mean_absolute_error',
                                'baseline': scorer,
                                'baseline_median': scorer5,
                                'relative': scorer2,
                                'baseline_relative': scorer3,
                                'baseline_median_relative': scorer4
                            },
                            return_train_score=False, return_estimator = True)

    # Use best estimator to do some visual reports
    rf = scores['estimator'][0]
    return [scores['estimator'], splits, scores]

def plot_predicted_vs_real_price(test_target, test_predictions, target, title, xlabel, ylabel, legend, labels = {}, zoomY = False):

    try:
        max = np.max(target)
    except:
        max = None

    plot_predicted_vs_real_price_start(max, title, xlabel, ylabel, legend, labels)
    plt.scatter(test_target, test_predictions, 100, alpha=0.05, edgecolors="none")
    plot_predicted_vs_real_price_end()

    if zoomY:
        plot_predicted_vs_real_price_start(max, title, xlabel, ylabel, legend, labels, zoomY)
        plt.scatter(test_target, test_predictions, 100, alpha=0.05, edgecolors="none")
        plot_predicted_vs_real_price_end()

def plot_predicted_vs_real_price_start(maxval, title, xlabel, ylabel, legend, labels = {}, zoomY = False):
    plt.figure(figsize=(8,8), dpi=130)
    if (len(labels)):
        plt.xticks(labels.keys(), labels.values())

    baseline = [0, maxval]
    if legend:
        plt.plot(baseline, baseline, "--", color="green", label = legend)
    ax = plt.gca()
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if legend:
        ax.legend()
    plt.title(title)
    if zoomY:
        plt.xlim(0, zoomY) 
        plt.ylim(0, zoomY)
    if not labels:
        plt.axes().set_aspect('equal', 'datalim')
    else:
        plt.xlim(-1, 2)

def plot_predicted_vs_real_price_end():
    plt.show()

def print_mean_absolute_error(test_predictions, test_target, average_target, median_target):

    # Calculate the absolute errors
    errors = abs(test_predictions - test_target)
    errors_baseline = abs(average_target - test_target)
    errors_baseline_median = abs(median_target - test_target)

    # Relative errors
    errors_relative = errors/test_target
    errors_baseline_relative = errors_baseline/test_target
    errors_baseline_median_relative = errors_baseline_median/test_target

    # Print out the mean absolute error (mae)
    print "== Absolute"
    print('Mean absolute prediction error: R$', round(np.mean(errors), 2))
    print('Std prediction error: R$', round(np.std(errors), 2))
    print('Mean absolute error using average: R$', round(np.mean(errors_baseline), 2))
    print('Std prediction error using average: R$', round(np.std(errors_baseline), 2))
    print('Mean absolute error using median: R$', round(np.mean(errors_baseline_median), 2))
    print('Std prediction error using median: R$', round(np.std(errors_baseline_median), 2))

    print "== Relative"
    print('Mean relative absolute prediction error: ', round(np.mean(errors_relative), 2))
    print('Std relative prediction error: ', round(np.std(errors_relative), 2))
    print('Mean relative absolute error using average: ', round(np.mean(errors_baseline_relative), 2))
    print('Std relative prediction error using average: ', round(np.std(errors_baseline_relative), 2))
    print('Mean absolute error using median: R$', round(np.mean(errors_baseline_median_relative), 2))
    print('Std prediction error using median: R$', round(np.std(errors_baseline_median_relative), 2))

    return [errors, errors_baseline, errors_relative, errors_baseline_relative, errors_baseline_median, errors_baseline_median_relative]

def join_predicted_df(
    df,
    test_features,
    test_target,
    test_predictions,
    errors,
    errors_relative,
    errors_baseline,
    errors_baseline_relative,
    errors_baseline_median,
    errors_baseline_median_relative
):
    data = {
        "all_features": test_features.tolist(),
        "sha256_id": test_features[:, 0],
        "target": test_target,
        "prediction": test_predictions,
        "error": errors,
        "error_relative": errors_relative,
        "error_baseline": errors_baseline,
        "error_baseline_relative": errors_baseline_relative,
        "error_baseline_median": errors_baseline_median,
        "error_baseline_median_relative": errors_baseline_median_relative
    }
    predicted_df = pd.DataFrame(data = data)
    joined_predicted_df = predicted_df
    joined_predicted_df = predicted_df.set_index("sha256_id").join(df.set_index("sha256_id"))
    return [predicted_df, joined_predicted_df]

def render_image_first_decision_tree(rf, feature_list, output_filename):
    # Pull out one tree from the forest
    tree = rf.estimators_[0]
    # Export the image to a dot file
    export_graphviz(tree, out_file = 'tree.dot',
                    feature_names = feature_list, rounded = True)
    # Use dot file to create a graph
    (graph, ) = pydot.graph_from_dot_file('tree.dot')
    # Write graph to a png file
    graph.write_png(output_filename)
    print "Output image: ", output_filename

def has_link_to_node(G, node):
    return dict(map(lambda n: [n, reduce(lambda t, i: (t+1)%2 if i == node else t, nx.all_neighbors(G, n), 0)], G.nodes_iter()))

def get_most_important_features(estimators, feature_list):
    rows = []
    for estimator in estimators:
        rows.append(estimator.feature_importances_)
    df = pd.DataFrame(rows, columns=feature_list)
    std = df.std() * 100
    mean = df.mean() * 100
    relative_std = std/mean
    importance = zip(feature_list, mean, std, relative_std)
    importance.sort(key=lambda x:-x[1])
    return pd.DataFrame(importance, columns=['feature', 'mean importance', 'std', 'std/mean']).head(20)

def plot_splits_confusion_matrices(X, y, splits, estimators, threshold = 0.5):

    rows = []

    print 'Splits quantity: ', len(splits)
    print 'Splits lenghts: ', [len(split[1]) for split in splits]
    print 'X shape: ', X.shape
    print 'y shape: ', y.shape

    confusion_matrices = []
    for i in range(len(splits)):
        test_indices = splits[i][1]
        y_pred = estimators[i].predict_proba(X[test_indices])
        cnf_matrix = sklearn.metrics.confusion_matrix(
            y[test_indices],
            [1 if p[1] >= threshold else 0 for p in y_pred]
        )
        confusion_matrices.append(cnf_matrix)

        plt.figure(figsize=(8,8), dpi=130)
        title = u'Split %i - Matriz de confusão\npara limiar de classificação p = %.2f' % (i+1, threshold)
        plot_confusion_matrix(cnf_matrix, normalize=False, classes=[u'Sem ligação', u'Com ligação'], title=title)
        plt.show()

    cf2d = [row.reshape(4) for row in confusion_matrices]

    df = pd.DataFrame(cf2d, columns=['tn', 'fp', 'fn', 'tp'])
    cnf_matrix_mean = df.mean().values.reshape(2, 2)
    cnf_matrix_std = df.std().values.reshape(2, 2)

    plt.figure(figsize=(8,8), dpi=130)
    plot_confusion_matrix(cnf_matrix_mean, std = cnf_matrix_std, normalize=True, classes=[u'Sem ligação', u'Com ligação'],
                          title=(u'Matriz de confusão agregada (média)\npara limiar de classificação p = %.2f' % (threshold)))
    plt.show()

def get_all_predictions_from_splits(X, y, splits, estimators):
    rows = []

    print 'Splits quantity: ', len(splits)
    print 'Splits lenghts: ', [len(split[1]) for split in splits]
    print 'X shape: ', X.shape
    print 'y shape: ', y.shape

    all_preds = []
    for i in range(len(splits)):
        test_indices = splits[i][1]
        if 'predict_proba' in dir(estimators[i]):
            y_pred = [p[1] for p in estimators[i].predict_proba(X[test_indices])]
        else:
            y_pred = estimators[i].predict(X[test_indices])
        all_preds = all_preds + zip(test_indices, y_pred)

    all_preds.sort(key = lambda x: x[0])

    return [x[1] for x in all_preds]

def plot_splits_predicted_vs_real(y, y_pred, title, xlabel, ylabel, legend, labels = {}, zoomY = False):
    plot_predicted_vs_real_price(y, y_pred, y, title, xlabel, ylabel, legend, labels, zoomY)

def plot_histogram(sets, xlabel='', ymax = False, nbins=10):

    if not ymax:
        ymax = max([max(x[1]) for x in sets])

    plt.figure(figsize=(8,8), dpi=130)
    max_frequency = 0

    hists = []
    for dataset in sets:
        y = dataset[1]
        count = len(y)
        hist = np.histogram(y, np.linspace(0, ymax, nbins))
        normalized_hist = hist[0]/count
        hists.append(normalized_hist)
        centers = np.convolve(hist[1], [0.5, 0.5])
        centers = centers[1:-1]
        #plt.bar(centers, normalized_hist, width=(centers[1]-centers[0])*0.95)
        plt.plot(centers, normalized_hist, label = dataset[0], marker='o')
        max_frequecy = max([max_frequency, max(hist[0]/count) * 1.1])

    ylim = max_frequency
    plt.legend(loc='lower right')
    plt.ylabel(u'Frequência relativa')
    plt.xlabel(xlabel)
    plt.show()
    return [centers, hists]


def print_classification_probability_distribution(y, y_pred):
    y_pred = np.array(y_pred)
    print y_pred.shape
    class0 = [x == 0 for x in y]
    class1 = [x == 1 for x in y]
    plot_histogram([[u'Sem ligação', y_pred[class0]], [u'Com ligação', y_pred[class1]]], xlabel=u'Probabilidade estimada de ligação pelo modelo')

def plot_roc_curve(target, y_pred):
    steps = 1001
    x_data = []
    y_data = []
    closest_to_optimal_distance = 99999
    closest_to_optimal_probability = None
    closest_to_optimal_point = None

    for r in range(0, steps):
        s = 1/(steps - 1) * r
        m = sklearn.metrics.confusion_matrix(target, map(lambda p: 1 if p >= s else 0, y_pred))
        tn = m[0][0]
        fp = m[0][1]
        fn = m[1][0]
        tp = m[1][1]
        tpr = tp/(tp+fn)
        fpr = fp/(fp+tn)
        y_data.append(tpr)
        x_data.append(fpr)
        distance = ((fpr - 0)**2.0 + (tpr - 1)**2.0)**0.5
        if distance < closest_to_optimal_distance:
            closest_to_optimal_point = [fpr, tpr]
            closest_to_optimal_distance = distance
            closest_to_optimal_probability = s
    plt.figure(figsize=(8,8), dpi=130)
    plt.plot(x_data, y_data)
    plt.plot([0, 1], [0, 1])
    plt.plot(closest_to_optimal_point[0], closest_to_optimal_point[1], "bx", label=u"Limiar de probabilidade = " + str(closest_to_optimal_probability))
    plt.axes().set_aspect('equal')
    plt.axes().set_ylabel(u"Taxa de verdadeiros positivos")
    plt.axes().set_xlabel(u"Taxa de falsos positivos")
    plt.axes().legend()
    plt.title(u"Curva ROC da variação do limiar de\nprobabilidade para classificação de positivos")
    plt.axes().set_aspect('equal')
    plt.show()
    return closest_to_optimal_probability
