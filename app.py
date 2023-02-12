from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    # Load the saved model
    model = load_model("mnist.h5")
    
    # Parse the image from the POST request
    image = request.get_json()["image"]
    image = np.array(image, dtype="float32")
    image = np.expand_dims(image, axis=0)
    
    # Use the model to make a prediction
    prediction = model.predict(image)
    prediction = np.argmax(prediction, axis=1)
    
    # Return the prediction as a JSON response
    response = {"prediction": int(prediction[0])}
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")
