try:
    from xgboost import XGBClassifier
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "xgboost is not installed in the current Python environment. "
        "Install with: pip install xgboost. "
        "In a Jupyter notebook, run in a cell: %pip install xgboost"
    ) from None
from sklearn.pipeline import Pipeline

def train_xgb(preprocessor,X_train, y_train):
    model_pipeline  = Pipeline(steps=[

        ("preprocessor",preprocessor),
        ("model", XGBClassifier(

            n_estimators=600,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            reg_alpha=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric="auc",
            tree_method="hist"

        ))
    ])

    model_pipeline.fit(X_train, y_train)
    return model_pipeline