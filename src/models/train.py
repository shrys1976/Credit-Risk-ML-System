"""Train credit risk models."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score


def train_logistic_model(preprocessor, X_train, y_train):
    model_pipeline  = Pipeline(steps=[

        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000))

    ])

    
    model_pipeline.fit(X_train, y_train)
    return model_pipeline

    



