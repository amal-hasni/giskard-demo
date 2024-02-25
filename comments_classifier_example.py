from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression




def comments_datasets(sample_size=10000, test_size: float = 0.3, seed: int = 0)-> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_csv('data/comments.csv', usecols=["comment_text", "target"]).sample(sample_size, random_state=seed)
    data['target'] = (data["target"]>0.75).astype(int)

    return train_test_split(data, test_size=test_size, random_state=seed)


def get_trained_model(df: pd.DataFrame) -> tuple[Callable[[pd.DataFrame], np.array], list[str]]:
    
   
    X_train, y_train = df["comment_text"], df["target"]
    vectorizer = CountVectorizer()
    X_features = vectorizer.fit_transform(X_train)
    
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_features, y_train)
    
    def classify_closure(strings: pd.DataFrame) -> np.array:
        x = vectorizer.transform(strings.comment_text)
        return clf.predict_proba(x)

    return classify_closure, clf.classes_
