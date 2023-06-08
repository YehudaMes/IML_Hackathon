# Cost of cancellation - Estimate the expected money loss of order cancellation
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, \
    VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from utils import save_model
from preproccer_task2 import preprocess_train_task2
from transform_task_2 import ClassifierTransformer
from work_temp import naive_preprocess


CLASSIFIER_NAME="ClassificationRowAdder"
REGRESSOR_NAME="Regressor"
TRAIN_PATH="./data/train.csv"

def task_2_train_preprocess(path):
    df = preprocess_train_task2(path)
    train_y = df.original_selling_amount
    classifier_y=df.cancellation_indicator
    train_X = df.drop(columns=['original_selling_amount', 'cancellation_indicator'])

    return train_X, train_y, classifier_y


def get_pipeline(classifier, regressor):
    return Pipeline([(CLASSIFIER_NAME,ClassifierTransformer(classifier)),
                     (REGRESSOR_NAME, regressor)])

def fit(path):
    model1 = DecisionTreeClassifier(max_depth=5)
    model2 = AdaBoostClassifier()

    model3 = GradientBoostingClassifier()
    # Create the ensemble by combining the models using VotingClassifier
    classifier = VotingClassifier(estimators=[('DecisionTree', model1), ('AdaBoost', model2), ('GradientBoosting', model3)], voting='hard')
    regressor=RandomForestRegressor()
    pipeline=get_pipeline(classifier=classifier, regressor=regressor)
    train_X, train_y, classifier_y=task_2_train_preprocess(path)
    classifier.fit(train_X, classifier_y)
    save_model(classifier, "task_2_train_classifier")
    print("classifier fitted")
    pipeline.fit(train_X,train_y,**{f"{CLASSIFIER_NAME}__y_classifier":classifier_y, f"{CLASSIFIER_NAME}__fit_again":False})
    save_model(regressor, "task_2_train_regressor")
    save_model(pipeline, "task_2_train_pipline")



def visualize_data_pca_response(data, response):
    # Perform PCA
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)

    # Extract the coordinates from the PCA-transformed data
    x = data_pca[:, 0]
    y = data_pca[:, 1]
    z = response

    # Create the scatter plot with different colors for response = -1
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=x[response == -1], y=y[response == -1], z=z[response == -1],
        mode='markers', marker=dict(color='red'), name='Response = -1'
    ))
    fig.add_trace(go.Scatter3d(
        x=x[response != -1], y=y[response != -1], z=z[response != -1],
        mode='markers', marker=dict(color='blue'), name='Response != -1'
    ))

    # Set layout and axis labels
    fig.update_layout(scene=dict(xaxis_title='PC1', yaxis_title='PC2', zaxis_title='Response'),
                      title='Data Matrix (PCA) with Response')

    # Show the plot
    fig.show()


def best_ridge(train_X, train_y, plot=True):
    alpha_values = np.linspace(1e-2, 5, 15)
    # Create Ridge regression model with all the alphas
    models = [Ridge(alpha) for alpha in alpha_values]
    # Perform cross-validation
    scores = np.array([np.mean(cross_val_score(
        model, train_X, train_y, cv=5, scoring='neg_mean_squared_error')) for model in models])
    if plot:
        go.Figure(
            go.Scatter(x=alpha_values, y=scores, name="-RMSE score vs alpha")).update_layout(
            title="Ridge selection").update_yaxes(title_text="alpha").update_yaxes(title_text="score")
    max_ind = np.argmax(scores)
    return models[max_ind], alpha_values[max_ind]


def evaluate_models(train_X, train_y, models):
    mean_neg_rmse_scores = np.zeros(len(models))
    for i, (model_name, pipeline) in enumerate(models):
        neg_rmse_scores = cross_val_score(pipeline, train_X, train_y, cv=6, scoring='neg_root_mean_squared_error')
        print(f'{model_name}:')
        print(f'  RMSE: {neg_rmse_scores}')
        print(f'  Average RMSE: {neg_rmse_scores.mean()}\n')
        mean_neg_rmse_scores[i] = neg_rmse_scores.mean()
    return models[np.argmax(mean_neg_rmse_scores)]






def pca_visualization():
    train_X, train_y = task_2_train_preprocess()
    visualize_data_pca_response(train_X, train_y)


if __name__ == '__main__':
    fit(TRAIN_PATH)
    # pca_visualization()
    raise Exception()
    train_X, train_y = task_2_train_reprocess()
    decision_models = [("RandomForest",
                        Pipeline([('scaler', StandardScaler()), (
                        "BaggingRegressor", BaggingRegressor(base_estimator=RandomForestRegressor()))])),
        ("BaggingRegressor-decision tree 10",
        Pipeline([('scaler', StandardScaler()), ("BaggingRegressor", BaggingRegressor(base_estimator=DecisionTreeRegressor(max_depth=10)))])),
                       ("BaggingRegressor-decision tree",
                        Pipeline([('scaler', StandardScaler()), (
                        "BaggingRegressor", BaggingRegressor(base_estimator=DecisionTreeRegressor()))])),
                       ("RandomForest",
                        Pipeline([('scaler', StandardScaler()), (
                        "BaggingRegressor", BaggingRegressor(base_estimator=RandomForestRegressor(max_depth=10)))]))
                       ]
        # [('adaboost',
        # Pipeline([('scaler', StandardScaler()), ('adaboost', AdaBoostRegressor())]))]
    #     (f'decision tree regressor {depth}', Pipeline(
    #     [('scaler', StandardScaler()), (f'decision tree regressor {depth}', RandomForestRegressor(max_depth=depth))]))
    #                    for depth in range(4, 9)]
    # decision_models.extend([
    #     ('Random Forest', Pipeline([('scaler', StandardScaler()), ('Random Forest', RandomForestRegressor())])),
    #     ('ElasticNet Regression 0.5',
    #      Pipeline([('scaler', StandardScaler()), ('ElasticNet Regression 0.5', ElasticNet(alpha=0.5))])),
    #     ('ElasticNet Regression 1',
    #      Pipeline([('scaler', StandardScaler()), ('ElasticNet Regression 1', ElasticNet(alpha=1))])),
    #     ('ElasticNet Regression 1.5',
    #      Pipeline([('scaler', StandardScaler()), ('ElasticNet Regression 1.5', ElasticNet(alpha=1.5))])),
    #     ('Decision Tree', Pipeline([('scaler', StandardScaler()), ('Decision Tree', DecisionTreeRegressor())])),
    #     ('Gradient Boosting',
    #      Pipeline([('scaler', StandardScaler()), ('Gradient Boosting', GradientBoostingRegressor())])),
    #     ('K-Nearest Neighbors',
    #      Pipeline([('scaler', StandardScaler()), ('K-Nearest Neighbors', KNeighborsRegressor())])),
    #     ('Laso Regression 0.5', Pipeline([('scaler', StandardScaler()), ('Laso Regression 0.5', Lasso(alpha=0.5))])),
    #     ('Ridge Regression 0.5', Pipeline([('scaler', StandardScaler()), ('Ridge Regression 0.5', Ridge(alpha=0.5))])),
    #     ('Laso Regression 1', Pipeline([('scaler', StandardScaler()), ('Laso Regression 1', Lasso(alpha=1))])),
    #     ('Ridge Regression 1', Pipeline([('scaler', StandardScaler()), ('Ridge Regression 1', Ridge(alpha=1))])),
    #     ('Laso Regression 1.5', Pipeline([('scaler', StandardScaler()), ('Laso Regression 1.5', Lasso(alpha=1.5))])),
    #     ('Ridge Regression 1.5', Pipeline([('scaler', StandardScaler()), ('Ridge Regression 1.5', Ridge(alpha=1.5))])),
    #     ('Linear Regression', Pipeline([('scaler', StandardScaler()), ('Linear Regression', LinearRegression())])),
    #     ('Support Vector Regression', Pipeline([('scaler', StandardScaler()), ('Support Vector Regression', SVR())]))
    # ])
    models=decision_models
    model_name, model = evaluate_models(train_X, train_y, models)
    print(f"chosen model: {model_name}")
# np.random.seed(0)
# num_points = 1000
# data = np.random.randn(num_points, 5)
# response = np.random.choice([-1, np.random.gamma(3, 5, num_points)], size=num_points)
#
#
# visualize_data_pca_response(data, response)
