# Cancellation prediction - Classification problem
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

from utils import save_model, load_model

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

# models = [
#
#     ("Decision Tree 1", DecisionTreeClassifier(max_depth=5))
#
# ]


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


def choose_cross_validation_classification_model(X: np.ndarray, y: np.ndarray) -> BaseEstimator:

    # Train and evaluate models
    results = []
    for name, model in models.copy():
        if name == 'Generic Random Forest':
            continue
        scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')

        print(f"{name} scores: \naccuracy: {scores.mean()},\n\n")
        results.append((name, scores.mean()))


    # Create a DataFrame to store the results
    df_results = pd.DataFrame(results, columns=['Model', 'F1-score'])

    # Get the best model (the one with the highest accuracy)
    best_model = find_best_model(df_results)

    # Print the best model
    print(f"The best model is: {best_model}")

    # Plot the results
    plot_results(df_results, 'Performance of CV-5 of Different Classification Models')

    return best_model


def ensemble_classification_model(X: np.ndarray, y: np.ndarray) -> BaseEstimator:

    # Train and evaluate models
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ensemble = fit_and_save_ensemble_3('splitted_', X_train, y_train)
    # Make predictions
    y_pred = ensemble.predict(X_test)
    print("ensemble accuracy: ", accuracy_score(y_test, y_pred))
    return ensemble


def fit_and_save_ensemble_3(X_train, y_train, prefix_name):
    print(prefix_name+'ensemble_3')
    # Define the individual models
    model1 = DecisionTreeClassifier(max_depth=5)
    model2 = AdaBoostClassifier()
    model3 = GradientBoostingClassifier()
    # Create the ensemble by combining the models using VotingClassifier
    ensemble = VotingClassifier(estimators=[('dt', model1), ('ab', model2), ('gbc', model3)], voting='hard')
    # Train the ensemble model
    ensemble.fit(X_train, y_train)
    save_model(ensemble, prefix_name+'ensemble_3')
    return ensemble


def load_model_and_predict(X_test: np.ndarray, y_test: np.ndarray):
    results = []
    for name, _ in models.copy():
        model = load_model(name)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        print(f"{name} scores: \naccuracy: {accuracy},\nprecision: {precision},\nrecall: {recall},\nf1: {f1}\n\n")
        results.append((name, accuracy, precision, recall, f1))

        # Create a DataFrame to store the results
    df_results = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-score'])

    best_model = find_best_model(df_results)

    # Print the best model
    print(f"The best model is: {best_model}")

    # Plot the results
    plot_2_measures(df_results, 'Performance of Different Classification Models')

    return best_model


def fit_and_save_model(X_train: np.ndarray, y_train: np.ndarray):
    for name, model in models.copy():
        print(f"saving {name}")
        model.fit(X_train, y_train)
        save_model(model, name)


if __name__ == "__main__":

    from preproccer_task1 import load_train_data_task1
    X, y = load_train_data_task1()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #
    # # plot_data(data,y)
    # # choose_classification_model(X, y)
    # # choose_cross_validation_classification_model(X, y)
    # # ensemble_classification_model(X, y)
    # # kernel_methods_classifcation(X,y)
    # # search_best_hyperparameters(X, y)
    # # evaluate_different_models(data, y)
    fit_and_save_model(X_train, y_train)
    load_model_and_predict(X_test, y_test)
    # fit_and_save_ensemble_3(X, y, "full_agoda_")
    # # ensemble_classification_model(X, y)
    # plot_train_result().write_image("classification_models_performance.png")
