# app.py
from flask import Flask, render_template, request
import os
from model import train_and_save, load_model, predict_url, DEFAULT_MODEL_PATH

app = Flask(__name__)

# Load or train model
if not os.path.exists(DEFAULT_MODEL_PATH):
    train_and_save()
model = load_model()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        url = request.form["url"]
        label = predict_url(url, model)
        if label == 1:
            result = "⚠️ Phishing Website Detected!"
        else:
            result = "✅ Legitimate Website"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
