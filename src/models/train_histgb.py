from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline


def train_hist_gradient_boosting(preprocessor, X_train, y_train):

    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.03,
            max_depth=7,
            min_samples_leaf=30,
            l2_regularization=2.0,
            random_state=42
        ))
    ])

    model_pipeline.fit(X_train, y_train)

    return model_pipeline
