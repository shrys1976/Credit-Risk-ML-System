from pyexpat import model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline


def train_gb(preprocessor, X_train, y_train):
    model_pipeline = Pipeline(steps=[

        ("preprocessor", preprocessor),
        ("model", GradientBoostingClassifier(

            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42

        ))


    ])

    model_pipeline.fit(X_train, y_train)
    return model_pipeline