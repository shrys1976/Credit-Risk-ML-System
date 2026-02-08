"""Evaluate trained models."""
<<<<<<< HEAD
=======
from sklearn.model_selection import cross_val_score
import numpy as np

def evaluate_model_cv(pipeline,X,y,cv=5):
    scores = cross_val_score(

        pipeline,
        X,
        y,
        cv=cv,
        scoring = "roc_auc",
        n_jobs = -1
    )

    return{
        "mean_auc": np.mean(scores),
        "std_auc": np.std(scores),
        "all_scores": scores
    }
>>>>>>> dev
