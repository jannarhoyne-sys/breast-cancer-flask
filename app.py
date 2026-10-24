from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model and scaler once
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


# show the input form
@app.route('/')
def index():
    return render_template('index.html')


# user input
@app.route('/predict', methods=['POST'])
def predict():
    # Read the 9 values from the form (they arrive as text, so convert to numbers)
    clump_thickness   = float(request.form['clump_thickness'])
    cell_size         = float(request.form['cell_size'])
    cell_shape        = float(request.form['cell_shape'])
    marginal_adhesion = float(request.form['marginal_adhesion'])
    epithelial_size   = float(request.form['epithelial_size'])
    bare_nuclei       = float(request.form['bare_nuclei'])
    bland_chromatin   = float(request.form['bland_chromatin'])
    normal_nucleoli   = float(request.form['normal_nucleoli'])
    mitoses           = float(request.form['mitoses'])

    features = np.array([[clump_thickness, cell_size, cell_shape, marginal_adhesion,
                          epithelial_size, bare_nuclei, bland_chromatin,
                          normal_nucleoli, mitoses]])

    # Scale the input exactly like the training data, then predict
    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)[0]

    #2 = benign, 4 = malignant
    result = 'Malignant' if pred == 4 else 'Benign'

    # Send the result back to the same page
    return render_template('index.html', predict=result)


if __name__ == '__main__':
    app.run(debug=True)