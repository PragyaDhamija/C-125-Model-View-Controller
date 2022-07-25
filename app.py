from flask import Flask, request, jsonify
from classifier import getPrediction

app = Flask(__name__)

@app.route("/")
def HomePage():
    return("Welcome to the home page... :)")

@app.route("/predict-digit", methods = ["POST"])
def PostImg():
    img = request.files.get("digit")
    pred = getPrediction(img)
    return jsonify({"Prediction":pred})

if(__name__ == "__main__"):
    app.run(debug = True)
