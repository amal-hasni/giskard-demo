from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import train_test_split



def churn_datasets(test_size: float = 0.3, seed: int =42)-> tuple[pd.DataFrame, pd.DataFrame]:
    data_path = "data/online_retail_customer_churn.csv"

    categorical_cols = ['Gender', 'Promotion_Response']
    df = pd.read_csv(data_path, index_col='Customer_ID', dtype={col: 'category' for col in categorical_cols})
    
    df = _feature_engineering(df)
    
    
    return train_test_split(df, test_size=test_size, random_state=seed)


def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Age Group
    age_groups = ["{0} - {1}".format(i, i + 9) for i in range(0, 99, 10)]
    cats = pd.Categorical(age_groups)
    df["Age group"] = pd.cut(df.Age, range(0, 101, 10), right=False, labels=cats).astype('category')
    
    # Spending Habits
    df['spending_habits'] = df['Total_Spend'].apply(lambda x: "High" if x>1000 else "Low").astype('category')
    
    # Customer Seniority
    seniority_groups = ["{0} - {1}".format(i, i + 3) for i in range(0, 19, 3)]
    cats = pd.Categorical(seniority_groups)
    df["customer_seniority"] = pd.cut(df.Years_as_Customer, range(0, 22, 3), right=False, labels=cats).astype('category')
    return df



def get_trained_model(df: pd.DataFrame) -> Pipeline:
    target_col = 'Target_Churn'
    feature_cols = [col for col in df.columns if col != target_col]
    X_train, y_train = df.loc[:, feature_cols], df[target_col]
    
    ct = make_column_transformer(
        (OneHotEncoder(), make_column_selector(dtype_include='category')),
        remainder='passthrough'
    )

    clf = make_pipeline(ct, RandomForestClassifier())
    return clf.fit(X_train, y_train)
