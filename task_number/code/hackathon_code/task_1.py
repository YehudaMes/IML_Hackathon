# Cancellation prediction - Classification problem
import numpy as np
import plotly.graph_objects as go

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA

from task_number.code.hackathon_code.utils import save_model, load_model

models = [
    # ("Nearest Neighbors", KNeighborsClassifier(5)),
    # ("Linear SVM", SVC(kernel="linear", probability=True)),
    # ("RBF SVM", SVC(gamma=2, C=1)),
    # ("Gaussian Process", GaussianProcessClassifier(1.0 * RBF(1.0))),
    ("Decision Tree", DecisionTreeClassifier(max_depth=5)),
    # ("Random Forest 5", RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)),
    # ("Neural Net", MLPClassifier(alpha=1, max_iter=1000)),
    ("AdaBoost", AdaBoostClassifier()),
    # ("Naive Bayes", GaussianNB()),
    # ("QDA", QuadraticDiscriminantAnalysis()),
    ('Logistic Regression', LogisticRegression(max_iter=1000)),
    ('Generic Random Forest', RandomForestClassifier()),
    ('Support Vector Machines', SVC()),
    ('Gradient Boosting', GradientBoostingClassifier())
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


def plot_4_measures(df_results, title):
    fig = go.Figure()
    fig.add_traces(data=[
        go.Scatter(mode="markers", marker=dict(size=10), name='Accuracy', x=df_results['Model'],
                   y=df_results['Accuracy']),
        go.Scatter(mode="markers", marker=dict(size=10), name='Precision', x=df_results['Model'],
                   y=df_results['Precision']),
        go.Scatter(mode="markers", marker=dict(size=10), name='Recall', x=df_results['Model'], y=df_results['Recall']),
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


def choose_classification_model(X: np.ndarray, y: np.ndarray)-> BaseEstimator:
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate models
    results = []
    for name, model in models.copy():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"{name} scores: \naccuracy: {accuracy},\nprecision: {precision},\nrecall: {recall},\nf1: {f1}\n\n")
        results.append((name, accuracy, precision, recall, f1))

    # Create a DataFrame to store the results
    df_results = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-score'])

    best_model = find_best_model(df_results)

    # Print the best model
    print(f"The best model is: {best_model}")

    # Plot the results
    plot_4_measures(df_results, 'Performance of Different Classification Models')

    return best_model


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


def kernel_methods_classification(X: np.ndarray, y: np.ndarray):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Perform PCA for dimensionality reduction
    pca = PCA(n_components=2)  # Adjust the number of components as desired
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Define the kernel methods to try
    kernel_methods = ['linear', 'poly', 'rbf', 'sigmoid']

    # Train and evaluate SVM models with different kernel methods
    results = []
    for kernel in kernel_methods:
        model = SVC(kernel=kernel)
        model.fit(X_train_pca, y_train)
        y_pred = model.predict(X_test_pca)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test,y_pred)
        results.append((kernel, accuracy, f1))

    # Create a DataFrame to store the results
    df_results = pd.DataFrame(results, columns=['Kernel Method', 'Accuracy', 'F1-score'])

    best_model = find_best_model(df_results)
    # Print the best model
    print(f"The best model is: {best_model}")
    plot_results(df_results, 'Performance of kernel-methods SVC')

    return best_model


def load_model_and_predict(X: np.ndarray, y: np.ndarray):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = []
    for name, _ in models.copy():
        model = load_model(name)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"{name} scores: \naccuracy: {accuracy},\nprecision: {precision},\nrecall: {recall},\nf1: {f1}\n\n")
        results.append((name, accuracy, precision, recall, f1))

        # Create a DataFrame to store the results
    df_results = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-score'])

    best_model = find_best_model(df_results)

    # Print the best model
    print(f"The best model is: {best_model}")

    # Plot the results
    plot_4_measures(df_results, 'Performance of Different Classification Models')

    return best_model


def fit_and_save_model(X: np.ndarray, y: np.ndarray):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    for name, model in models.copy():
        print(f"saving {name}")
        model.fit(X_train, y_train)
        save_model(model, name)


if __name__ == "__main__":
    from preproccer_task1 import load_data
    df = load_data()
    X, y = df.loc[:, df.columns != "cancellation_indicator"], df.cancellation_indicator
    # plot_data(data,y)
    # choose_classification_model(X, y)
    # choose_cross_validation_classification_model(X, y)
    # ensemble_classification_model(X, y)
    # kernel_methods_classifcation(X,y)
    # search_best_hyperparameters(X, y)
    # evaluate_different_models(data, y)
    fit_and_save_ensemble_3(X, y, "full_train_")
    # ensemble_classification_model(X, y)
