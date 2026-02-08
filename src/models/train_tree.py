from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def train_random_forest(preprocessor, X_train, y_train):

    model_pipeline = Pipeline(steps=[

        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(

            n_estimators=300,
            max_depth=None,
            min_samples_leaf = 2,
            min_samples_split = 5,
            n_jobs=-1,
            random_state=42

        ))

    ])

    model_pipeline.fit(X_train, y_train)
    return model_pipeline