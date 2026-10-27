import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

d = np.load("data/prepared.npz")
X_train, y_train = d["X_train"], d["y_train"]
X_val, y_val = d["X_val"], d["y_val"]
X_test, y_test = d["X_test"], d["y_test"]

# Dumbest possible, always guess the most common genre
majority = np.bincount(y_train).argmax()
acc_majority = accuracy_score(y_test, np.full_like(y_test, majority))
print(f"Always-guess-one-genre accuracy: {acc_majority:.3f}")

# Baseline 1: logistic regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
print(f"Logistic regression — val: {accuracy_score(y_val, logreg.predict(X_val)):.3f}  "
    f"test: {accuracy_score(y_test, logreg.predict(X_test)):.3f}")

# Baseline 2: random forest
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
print(f"Random forest      — val: {accuracy_score(y_val, rf.predict(X_val)):.3f}  "
    f"test: {accuracy_score(y_test, rf.predict(X_test)):.3f}")
