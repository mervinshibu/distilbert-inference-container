from flask import Flask, request, jsonify
from transformers import pipeline
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""


app = Flask(__name__)

classifier = None

def get_model():
    global classifier
    if classifier is None:
        classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1  # force CPU
        )
    return classifier

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    model = get_model()
    result = model(text)[0]

    return jsonify({
        "label": result["label"],
        "score": result["score"]
    })
