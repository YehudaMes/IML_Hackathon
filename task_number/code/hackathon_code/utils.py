import joblib
from sklearn.base import BaseEstimator


def save_model(model: BaseEstimator, name):
    path = 'trained_model/'+name+'.joblib'
    joblib.dump(model, path)


def load_model(name):
    path = 'trained_model/' + name + '.joblib'
    return joblib.load(path)
