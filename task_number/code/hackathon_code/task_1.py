# Cancellation prediction - Classification problem
import csv

import numpy as np
import plotly.graph_objects as go

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA

from .utils import *
from .preproccer_task1 import *

models = [
    ("Nearest Neighbors", KNeighborsClassifier(5)),
    ("LDA", LinearDiscriminantAnalysis(store_covariance=True)),
    # ("RBF SVM", SVC(gamma=2, C=1)),
    # ("Gaussian Process", GaussianProcessClassifier(1.0 * RBF(1.0))),
    ("Decision Tree 10", DecisionTreeClassifier(max_depth=10)),
    ("Decision Tree 5", DecisionTreeClassifier(max_depth=5)),
    ("Random Forest 5", RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)),
    ("AdaBoost", AdaBoostClassifier()),
    # ("Naive Bayes", GaussianNB()),
    # ("QDA", QuadraticDiscriminantAnalysis()),
    ('Logistic Regression', LogisticRegression(max_iter=1000)),
    ('Generic Random Forest', RandomForestClassifier()),
    # ('Support Vector Machines', SVC()),
    ('Gradient Boosting', GradientBoostingClassifier()),
    ("Decision Tree", DecisionTreeClassifier()),

    ("Neural Net", MLPClassifier(alpha=1, max_iter=1000)),
    ("Default Logistic Regression", LogisticRegression()),
    ("Linear SVM", SVC(kernel="linear"))

]


def find_best_model(df_results):
    # Sort the results DataFrame by the desired evaluation metric (e.g., accuracy)
    df_results_sorted = df_results.sort_values(by='Accuracy', ascending=False)

    # Get the best model (the one with the highest accuracy)
    return df_results_sorted.iloc[0][0]


def plot_results(df_results, title):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(mode="markers", marker=dict(size=10), name='F1-score', x=df_results[0],
                   y=df_results['F1-score'])
    )
    # Update the layout of the plot
    fig.update_layout(
        title=title,
        xaxis_title='Model',
        yaxis_title='Score'
    )
    fig.show()


def plot_2_measures(df_results, title):
    fig = go.Figure()
    fig.add_traces(data=[
        go.Scatter(mode="markers", marker=dict(size=10), name='Accuracy', x=df_results['Model'],
                   y=df_results['Accuracy']),
        go.Scatter(mode="markers", marker=dict(size=10), name='F1-score', x=df_results['Model'],
                   y=df_results['F1-score'])
    ])
    # Update the layout of the plot
    fig.update_layout(
        title=title,
        xaxis_title='Model',
        yaxis_title='Score'
    )
    # Show the plot
    fig.show()
    return fig


def fit_and_save_ensemble_3(X_train, y_train, prefix_name):
    print(prefix_name+'ensemble_3')
    # Define the individual models
    model1 = DecisionTreeClassifier(max_depth=10)
    model2 = AdaBoostClassifier()
    model3 = GradientBoostingClassifier()
    # Create the ensemble by combining the models using VotingClassifier
    ensemble = VotingClassifier(estimators=[('dt', model1), ('ab', model2), ('gbc', model3)], voting='hard')
    # Train the ensemble model
    ensemble.fit(X_train, y_train)
    save_model(ensemble, prefix_name+'ensemble_3')
    return ensemble


def load_models_and_predict(X_test: np.ndarray, y_test: np.ndarray):
    results = []
    for name, _ in models.copy():
        model = load_model(name)
        accuracy, f1, precision, recall = model_predict(model, name, X_test, y_test)
        results.append((name, accuracy, precision, recall, f1))

        # Create a DataFrame to store the results
    df_results = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-score'])

    best_model = find_best_model(df_results)

    # Print the best model
    print(f"The best model is: {best_model}")

    # Plot the results
    plot_2_measures(df_results, 'Performance of Different Classification Models')

    return best_model


def model_predict(model, name, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"{name} scores: \naccuracy: {accuracy},\nprecision: {precision},\nrecall: {recall},\nf1: {f1}\n\n")
    return accuracy, f1, precision, recall


def fit_and_save_model(X_train: np.ndarray, y_train: np.ndarray):
    for name, model in models.copy():
        print(f"saving {name}")
        model.fit(X_train, y_train)
        save_model(model, name)


def write_file(path, id_array, cancellation_array):
    # Combining the id and cancellation arrays into a single array
    pd.DataFrame({"ID": id_array, "cancellation": cancellation_array})\
        .to_csv(path, index=False)

    print("CSV file created successfully.")


def run_task_1(path_to_input):
    ids = pd.read_csv(path_to_input)['h_booking_id']
    model = load_model('agoda_all_ensemble_3')

    test_X = preprocess_predict_task1(path_to_input)
    test_y = model.predict(test_X)

    write_file('../../predictions/agoda_cancellation_prediction.csv', ids, test_y)


# if __name__ == "__main__":
#     run_task_1("agoda_data/Agoda_Test_1.csv")
# #
#     from preproccer_task1 import load_train_data_task1, load_validation_data_task1, load_train_agoda_data_task1
# #
#     # X, y = load_train_data_task1()
# #     X_test, y_test = load_validation_data_task1()
# #
# #     # fit_and_save_model(X_train, y_train)
# #     # load_model_and_predict(X_test, y_test)
# #     # fit_and_save_ensemble_3(X, y, "train_")
# #     # model_predict(load_model("train_"+'ensemble_3'), "validation "+'ensemble_3', X_test, y_test)
# #
#     X_all, y_all = preprocess_t1.load_train_agoda_data_task1(True)
#     fit_and_save_ensemble_3(X_all, y_all, "agoda_all_")

    # # ensemble_classification_model(X, y)
    # plot_train_result().write_image("classification_models_performance.png")
