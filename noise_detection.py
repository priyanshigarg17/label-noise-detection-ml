from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def detect_suspicious_samples(X, y, n_splits=5):

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    suspicious = []

    for train_idx, val_idx in skf.split(X, y):

        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = RandomForestClassifier(random_state=42)
        model.fit(X_tr, y_tr)

        probs = model.predict_proba(X_val)

        for i, idx in enumerate(val_idx):
            true_class = y_val.iloc[i]
            confidence = probs[i][true_class]

            if confidence < 0.4:   # threshold
                suspicious.append(idx)

    return suspicious
