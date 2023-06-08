from copy import deepcopy

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted


# used for the pipeline in part 2 with extensive questioning of chatgpt
class ClassifierTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier
        self.fitted=False

    def fit(self, X, y,y_classifier, fit_again):
        if not self.fitted or fit_again:
            self.classifier.fit(X, y_classifier)
            self.fitted=True
        return self

    def transform(self, X):
        # Get the predicted labels from the classifier
        labels = self.classifier.predict(X)

        # Reshape the predicted labels into a column vector
        labels = labels.reshape(-1,1)

        # Concatenate the predicted labels to the original feature matrix
        transformed_X = deepcopy(X)
        print(f"before adding column: {type(transformed_X)}")

        transformed_X["cancellation_indicator"] = labels
        print(f"after adding column: {type(transformed_X)}")
        return transformed_X

    def predict(self, X):
        return self.classifier.predict(X)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def score(self, X, y):
        return self.classifier.score(X, y)
