from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("phishing_model.pkl")

FEATURE_NAMES = model.feature_names_in_

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        data = []
        for feature in FEATURE_NAMES:
            value = float(request.form[feature])
            data.append(value)

        X = pd.DataFrame([data], columns=FEATURE_NAMES)
        result = model.predict(X)[0]

        prediction = "Phishing Website" if result == 1 else "Legitimate Website"

    return render_template("index.html", features=FEATURE_NAMES, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
