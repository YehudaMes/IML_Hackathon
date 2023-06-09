# Cost of cancellation - Estimate the expected money loss of order cancellation
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from .task_1 import fit_and_save_ensemble_3
from .preproccer_task2 import preprocess_test_task2, task_2_train_preprocess
from .transform_task_2 import ClassifierTransformer
from .utils import save_model, load_model

CLASSIFIER_NAME = "ClassificationRowAdder"
REGRESSOR_NAME = "Regressor"
TRAIN_PATH = "./data/train.csv"

TEST_CLASSIFIER_NAME = "task_2_test_classifier"
TEST_REGRESSOR_NAME = "task_2_test_regressor"


def get_regressors_models():
    decision_models = [
        (f'decision tree regressor {depth}', Pipeline(
            [('scaler', StandardScaler()),
             (f'decision tree regressor {depth}', RandomForestRegressor(max_depth=depth))]))
        for depth in range(4, 9)]
    decision_models.extend([
        ('Random Forest', Pipeline([('scaler', StandardScaler()), ('Random Forest', RandomForestRegressor())])),
        ('ElasticNet Regression 0.5',
         Pipeline([('scaler', StandardScaler()), ('ElasticNet Regression 0.5', ElasticNet(alpha=0.5))])),
        ('ElasticNet Regression 1',
         Pipeline([('scaler', StandardScaler()), ('ElasticNet Regression 1', ElasticNet(alpha=1))])),
        ('ElasticNet Regression 1.5',
         Pipeline([('scaler', StandardScaler()), ('ElasticNet Regression 1.5', ElasticNet(alpha=1.5))])),
        ('Decision Tree', Pipeline([('scaler', StandardScaler()), ('Decision Tree', DecisionTreeRegressor())])),
        ('Gradient Boosting',
         Pipeline([('scaler', StandardScaler()), ('Gradient Boosting', GradientBoostingRegressor())])),
        ('K-Nearest Neighbors',
         Pipeline([('scaler', StandardScaler()), ('K-Nearest Neighbors', KNeighborsRegressor())])),
        ('Laso Regression 0.5', Pipeline([('scaler', StandardScaler()), ('Laso Regression 0.5', Lasso(alpha=0.5))])),
        ('Ridge Regression 0.5', Pipeline([('scaler', StandardScaler()), ('Ridge Regression 0.5', Ridge(alpha=0.5))])),
        ('Laso Regression 1', Pipeline([('scaler', StandardScaler()), ('Laso Regression 1', Lasso(alpha=1))])),
        ('Ridge Regression 1', Pipeline([('scaler', StandardScaler()), ('Ridge Regression 1', Ridge(alpha=1))])),
        ('Laso Regression 1.5', Pipeline([('scaler', StandardScaler()), ('Laso Regression 1.5', Lasso(alpha=1.5))])),
        ('Ridge Regression 1.5', Pipeline([('scaler', StandardScaler()), ('Ridge Regression 1.5', Ridge(alpha=1.5))])),
        ('Linear Regression', Pipeline([('scaler', StandardScaler()), ('Linear Regression', LinearRegression())])),
        ('adaboost', Pipeline([('scaler', StandardScaler()), ('adaboost', AdaBoostRegressor())])),
        ('Support Vector Regression', Pipeline([('scaler', StandardScaler()), ('Support Vector Regression', SVR())]))
    ])
    models = decision_models
    return models


def get_pipeline(classifier, regressor):
    return Pipeline([(CLASSIFIER_NAME, ClassifierTransformer(classifier)),
                     (REGRESSOR_NAME, regressor)])


def fit(data_path, classifier_name=None, fit_classifier=True, regressor_name="regressor", fit_regressor=True):
    train_X, train_y, classifier_y = task_2_train_preprocess(data_path)
    if not fit_classifier:
        classifier = load_model(classifier_name + "ensemble_3")
    else:
        classifier = fit_and_save_ensemble_3(train_X, classifier_y, classifier_name)
    print("classifier fitted")
    if fit_regressor:
        regressor = Pipeline([('scaler', StandardScaler()), ('Decision Tree', DecisionTreeRegressor())])
        regressor.fit(train_X.loc[classifier_y == 1], train_y.loc[classifier_y == 1])
        save_model(regressor, regressor_name)
    else:
        regressor = load_model(regressor_name, regressor_name)
    return classifier, regressor


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


def predict(X, classifier, regressor, ids=None, output_path=None, save=False):
    cancel = classifier.predict(X)
    loss = -np.ones(len(X))
    price_inds = cancel == 1
    price = regressor.predict(X.loc[price_inds])
    loss[price_inds] = price
    if save:
        pd.DataFrame({"ID": ids, "predicted_selling_amount": loss}).to_csv(output_path, index=False)
    else:
        return loss


def read_models(base_path):
    return load_model(base_path + CLASSIFIER_NAME), load_model(base_path + REGRESSOR_NAME)


def check_against_validation(validation_path, classifier, regressor):
    X, y, _ = task_2_train_preprocess(validation_path, save_columns=False)
    print(f"validation, RMSE: {RMSE(X, y, classifier, regressor)}")


def RMSE(X, y, classifier, regressor):
    rmse = np.sqrt(np.sum((predict(X, classifier, regressor) - y) ** 2) / len(y))
    return rmse


def full_validation(fit_classifier=True, fit_regressor=True):
    classifier, regressor = fit("./data/train.csv", classifier_name="task_2_validation_classifier", fit_classifier=fit_classifier,
                                regressor_name="validation_regressor", fit_regressor=fit_regressor)
    check_against_validation("./data/validation.csv", classifier, regressor)


def run_task_2(test_data_path, output_path, train=False):
    if train:
        classifier, regressor = fit("./agoda_data/agoda_cancellation_train.csv", TEST_CLASSIFIER_NAME,train=True,
                                    regressor_name=TEST_REGRESSOR_NAME)
    else:
        classifier, regressor = load_model(TEST_CLASSIFIER_NAME + "ensemble_3"), load_model(TEST_REGRESSOR_NAME)
    ids, test_X = preprocess_test_task2(test_data_path)
    predict(test_X, classifier, regressor, ids=ids, output_path=output_path, save=True)


# if __name__ == '__main__':
#     task_2_test("./agoda_data/Agoda_Test_2.csv", train=False, output_path="../../predictions/agoda_cost_of_cancellation.csv")
    # full_validation(False, False)
    # fit(TRAIN_PATH)
    # pca_visualization()

    # model_name, model = evaluate_models(train_X.loc[classifier_y==1][:10000], train_y.loc[classifier_y==1][:10000], models)
    # print(f"chosen model: {model_name}")
# np.random.seed(0)
# num_points = 1000
# data = np.random.randn(num_points, 5)
# response = np.random.choice([-1, np.random.gamma(3, 5, num_points)], size=num_points)
#
#
# visualize_data_pca_response(data, response)
