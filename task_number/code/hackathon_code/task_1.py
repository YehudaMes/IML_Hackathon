# Cancellation prediction - Classification problem
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.datasets import make_moons
from utils import *
import numpy as np
import plotly.graph_objects as go

title = "Agoda"
np.random.seed(1)
m = 250
symbols = np.array(["circle", "x"])
(X, y) = []
lims = np.array([X.min(axis=0), X.max(axis=0)]).T + np.array([-.4, .4])
def execute_models():
    models = [
        LogisticRegression(penalty="none"),
        DecisionTreeClassifier(max_depth=5),
        KNeighborsClassifier(n_neighbors=5),
        SVC(kernel='linear', probability=True),
        LinearDiscriminantAnalysis(store_covariance=True),
        QuadraticDiscriminantAnalysis(store_covariance=True)
    ]
    model_names = ["Logistic regression", "Desicion Tree (Depth 5)", "KNN", "Linear SVM", "LDA", "QDA"]
    plot_predictions(model_names, models)

    plot_evaluations(model_names, models)


def plot_evaluations(model_names, models):
    fig = go.Figure(
        layout=go.Layout(title=rf"$\textbf{{(3) ROC Curves Of Models - {title} Dataset}}$", margin=dict(t=100)))
    for i, model in enumerate(models):
        fpr, tpr, th = metrics.roc_curve(y, model.predict_proba(X)[:, 1])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=model_names[i]))
    fig.show()


def plot_predictions(model_names, models):
    fig = make_subplots(rows=2, cols=3, subplot_titles=[rf"$\textbf{{{m}}}$" for m in model_names],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    for i, m in enumerate(models):
        fig.add_traces([decision_surface(m.fit(X, y).predict, lims[0], lims[1], showscale=False),
                        go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=y, symbol=symbols[y], colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))],
                       rows=(i // 3) + 1, cols=(i % 3) + 1)
    fig.update_layout(title=rf"$\textbf{{(2) Decision Boundaries Of Models - {title} Dataset}}$", margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)


