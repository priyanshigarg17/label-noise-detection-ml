from sklearn.ensemble import RandomForestClassifier

def get_model():
    return RandomForestClassifier(random_state=42)

def train_model(model, X, y, sample_weight=None):
    model.fit(X, y, sample_weight=sample_weight)
    return model
