import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def evaluate_feature_subset(X, y, selected):
    if sum(selected) == 0:
        return 0.0
    cols = [i for i,b in enumerate(selected) if b==1]
    Xs = X[:, cols]
    clf = RandomForestClassifier(n_estimators=40)
    scores = cross_val_score(clf, Xs, y, cv=3, scoring="f1_macro")
    return scores.mean()
