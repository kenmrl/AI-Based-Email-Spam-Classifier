from flask import Flask, request, jsonify
import joblib
import re

app = Flask(__name__)
model = joblib.load("spam_model.pkl")  # Pre-trained Spam Model
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    email = request.json["email"]
    email_clean = re.sub(r'\W+', ' ', email.lower())  # Clean text
    email_vectorized = vectorizer.transform([email_clean])
    
    prediction = model.predict(email_vectorized)[0]
    result = "Spam" if prediction == 1 else "Not Spam"
    
    return jsonify({"classification": result})

if __name__ == "__main__":
    app.run(debug=True)
