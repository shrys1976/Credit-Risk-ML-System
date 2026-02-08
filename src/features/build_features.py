"""Build features for credit risk models."""
<<<<<<< HEAD
=======

import numpy as np
import pandas as pd


def basic_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365

   
    df["CREDIT_TO_INCOME"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_TO_INCOME"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]

    
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)


    drop_cols = [

        "SK_ID_CURR",
        "DAYS_BIRTH"
    ]

    df = df.drop(columns = [c for c in drop_cols if c in df.columns])
    return df
>>>>>>> dev
