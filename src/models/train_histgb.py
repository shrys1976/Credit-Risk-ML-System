from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline


def train_hist_gradient_boosting(preprocessor, X_train, y_train):

    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=6,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=42
        ))
    ])

    model_pipeline.fit(X_train, y_train)

    return model_pipeline
