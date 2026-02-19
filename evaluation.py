from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate(model, X_test, y_test):

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    return {
        "accuracy": acc,
        "confusion_matrix": cm
    }
