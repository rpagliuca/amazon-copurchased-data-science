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

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

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
    total = 0
    for value in histogram:
        total += value
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

    df_with_dummies = pd.get_dummies(
            df[['sha256_id'] + numeric_features + categorical_features + [target_column]],
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
    count = len(predicted_df.error_relative)
    hist_predicted = np.histogram(predicted_df.error_relative, np.linspace(0, 2, 20))
    normalized_hist_predicted = hist_predicted[0]/count
    hist_baseline = np.histogram(predicted_df.error_baseline_relative, np.linspace(0, 2, 20))
    normalized_hist_baseline = hist_baseline[0]/count
    centers = np.convolve(hist_predicted[1], [0.5, 0.5])
    centers = centers[1:-1]
    plt.bar(centers, normalized_hist_predicted, width=(centers[1]-centers[0])*0.95)
    ylim = max(hist_predicted[0]/count) * 1.1
    plt.ylim(0, ylim)
    plt.show()
    plt.bar(centers, normalized_hist_baseline, width=(centers[1]-centers[0])*0.95)
    plt.ylim(0, ylim)
    plt.show()
    return [centers, normalized_hist_predicted, normalized_hist_baseline]

def plot_accumulated_relative_error(centers, normalized_hist_predicted, normalized_hist_baseline):
    plt.plot(centers, accumulated_distribution(normalized_hist_predicted), label='Predicted')
    plt.plot(centers, accumulated_distribution(normalized_hist_baseline), label='Baseline')
    plt.xlabel('Relative error')
    plt.ylabel('Accumulated fraction')
    plt.legend(loc='lower right')
    plt.show()

def print_score_summary(scores):
    print "=== Relative"
    print "RF relative abs mean: ", np.mean(np.abs(scores['test_relative']))
    print "RF relative abs std: ", np.std(scores['test_relative'])

    #print "Baseline: ", scores['test_baseline']
    print "Baseline relative mean: ", np.mean(scores['test_baseline_relative'])
    print "Baseline relative std: ", np.std(scores['test_baseline_relative'])

    print "=== Absolute"

    #print "Abs: ", scores['test_abs']
    print "RF abs mean: ", np.mean(np.abs(scores['test_abs']))
    print "RF abs std: ", np.std(scores['test_abs'])

    #print "Baseline: ", scores['test_baseline']
    print "Baseline mean: ", np.mean(scores['test_baseline'])
    print "Baseline std: ", np.std(scores['test_baseline'])

def print_score_summary_classification(scores):
    return True

def run_cross_validation_classification(features, target):

    def class_weight(label):
        return len([i for i in target if i == label])

    class_weights = {
        0: (class_weight(1)/len(target))*1E0,
        1: (class_weight(0)/len(target))*1E7
    }

    rf = RandomForestClassifier(n_estimators = 500, class_weight=class_weights)
    scores = cross_validate(estimator=rf, X=features, y=target, cv=10,
                            scoring = {
                                'abs': 'neg_mean_absolute_error'
                            },
                            return_train_score=False, return_estimator = True)
    # Use best estimator to do some visual reports
    rf = scores['estimator'][0]
    return [rf, scores]

def run_cross_validation_regression(features, target):
    average_target = np.average(target)

    def baseline_score_function (y_true, y_pred):
        errors_baseline = abs(average_target - y_true)
        return np.mean(errors_baseline)

    def relative_error_function (y_true, y_pred):
        errors_absolute = abs(y_pred - y_true)
        errors_relative = errors_absolute / y_true
        return np.mean(errors_relative)

    def baseline_relative_error_function (y_true, y_pred):
        errors_absolute = abs(average_target - y_true)
        errors_relative = errors_absolute / y_true
        return np.mean(errors_relative)


    rf = RandomForestRegressor(n_estimators = 50, n_jobs = -1)

    scorer = make_scorer(baseline_score_function)
    scorer2 = make_scorer(relative_error_function)
    scorer3 = make_scorer(baseline_relative_error_function)

    scores = cross_validate(estimator=rf, X=features, y=target, cv=10,
                            scoring = {
                                'abs': 'neg_mean_absolute_error',
                                'baseline': scorer,
                                'relative': scorer2,
                                'baseline_relative': scorer3
                            },
                            return_train_score=False, return_estimator = True)

    # Use best estimator to do some visual reports
    rf = scores['estimator'][0]
    return [rf, scores]

def plot_predicted_vs_real_price(test_target, test_predictions, target):
    plt.figure(figsize=(8,8), dpi=130)
    plt.scatter(test_target, test_predictions, 100, alpha=0.05, edgecolors="none")
    baseline = [0, np.max(target)]
    plt.plot(baseline, baseline, "--", color="green", label = u"Preço previsto = Preço real")
    ax = plt.gca()
    ax.set_ylabel(u"Preço previsto (R$)")
    ax.set_xlabel(u"Preço real (R$)")
    ax.legend()
    plt.title(u"Preço previsto vs. Preço real")
    plt.axes().set_aspect('equal', 'datalim')
    #plt.xlim(0, 150)
    #plt.ylim(0, 150)
    plt.show()

def print_mean_absolute_error(test_predictions, test_target, average_target):
    # Calculate the absolute errors
    errors = abs(test_predictions - test_target)
    errors_baseline = abs(average_target - test_target)
    errors_relative = errors/test_target
    errors_baseline_relative = errors_baseline/test_target
    # Print out the mean absolute error (mae)
    print "== Absolute"
    print('Mean absolute prediction error: R$', round(np.mean(errors), 2))
    print('Std prediction error: R$', round(np.std(errors), 2))
    print('Mean absolute error using average: R$', round(np.mean(errors_baseline), 2))
    print('Std prediction error using average: R$', round(np.std(errors_baseline), 2))
    print "== Relative"
    print('Mean relative absolute prediction error: ', round(np.mean(errors_relative), 2))
    print('Std relative prediction error: ', round(np.std(errors_relative), 2))
    print('Mean relative absolute error using average: ', round(np.mean(errors_baseline_relative), 2))
    print('Std relative prediction error using average: ', round(np.std(errors_baseline_relative), 2))
    return [errors, errors_baseline, errors_relative, errors_baseline_relative]

def join_predicted_df(
    df,
    test_features,
    test_target,
    test_predictions,
    errors,
    errors_relative,
    errors_baseline,
    errors_baseline_relative
):
    data = {
        "all_features": test_features.tolist(),
        "sha256_id": test_features[:, 0],
        "target": test_target,
        "prediction": test_predictions,
        "error": errors,
        "error_relative": errors_relative,
        "error_baseline": errors_baseline,
        "error_baseline_relative": errors_baseline_relative
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
