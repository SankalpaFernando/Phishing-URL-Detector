from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import pandas as pd

dataset = fetch_ucirepo(id=967)

X = dataset.data.features.copy()
y = dataset.data.targets.copy()
y = y.iloc[:, 0].values

X = X.apply(pd.to_numeric, errors="coerce")

X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(model, "phishing_model.pkl")
print("Model trained and saved successfully")
