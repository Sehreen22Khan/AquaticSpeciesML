
from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler


# Initialise the Flask app
app = Flask(__name__)

# Use pickle to load in the pre-trained model
filename = "models/finalized_model.sav"
model = pickle.load(open(filename, "rb"))

# Set up the main route
@app.route('/', methods=["GET", "POST"])
def main():

    if request.method == "POST":
        loaded_model = pickle.load(open('./models/finalized_model.sav', 'rb'))

        # Extract the input from the form
        weight = request.form.get("weight")
        length1 = request.form.get("length1")
        length2 = request.form.get("length2")
        length3 = request.form.get("length3")
        height = request.form.get("height")
        width = request.form.get("width")

        # Convert user input into a numpy array and reshape it
        user_input = np.array([weight, length1, length2, length3, height, width]).reshape(1, -1)

        # Standardize user input
        scaler = StandardScaler()
        scaler.fit(user_input)
        user_input_scaled = scaler.transform(user_input)

        # Use the loaded model to make predictions
        prediction = loaded_model.predict(user_input_scaled)
    
        # We now pass on the input from the form and the prediction to the index page
        return render_template("index.html",
                                     original_input={'Weight': weight,
                                                     'Length1': length1,
                                                     'Length2': length2,
                                                     'Length3': length3,
                                                     'Height': height,
                                                     'Width': width},
                                     result=prediction
                                     )
    # If the request method is GET
    return render_template("index.html")
