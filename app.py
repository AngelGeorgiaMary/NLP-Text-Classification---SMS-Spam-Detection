from flask import Flask, request, jsonify, render_template
import joblib
import re

app = Flask(__name__)

# Load the model and vectorizer
model = joblib.load("spam_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Show index.html form
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    spam_prob = None
    if request.method == "POST":
        message = request.form["message"]
        processed = preprocess(message)
        vector = tfidf.transform([processed])
        result = model.predict(vector)[0]
        prediction = "Spam" if result == 1 else "Ham"
        probas = model.predict_proba(vector)[0]
        spam_prob = round(probas[1], 2)
    return render_template("index.html", prediction=prediction, spam_prob=spam_prob)

# Handle API call with JSON data
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    message = data.get("message", "")
    processed = preprocess(message)
    vector = tfidf.transform([processed])
    prediction = model.predict(vector)[0]
    label = "spam" if prediction == 1 else "ham"
    probas = model.predict_proba(vector)[0]
    spam_prob = round(probas[1], 2)
    return jsonify({
        "label": label,
        "spam_prob": spam_prob
    })

if __name__ == "__main__":
    app.run(debug=True)
