from flask import Flask, render_template, request
import numpy as np
import os
from tensorflow.keras.models  import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)  # Define the Flask app

model = load_model(r'D:\Python Projects\Plant Pathology\Models\apple2.h5')  # Loading the model

def model_prediction(img_path, model):
    test_image = image.load_img(img_path, target_size=(224, 224))  # Load test image for prediction
    test_image = image.img_to_array(test_image)  # Convert image to array of numbers
    test_image_scaled = test_image / 255  # Scaled every pixel from 0 to 1
    test_image_scaled = np.expand_dims(test_image_scaled, axis=0)
    result = model.predict(test_image_scaled)
    return result

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save file to the upload folder
        basepath = os.path.dirname(os.path.realpath('__file__'))
        file_path = os.path.join(basepath, 'Uploads', secure_filename((f.filename)))
        f.save(file_path)

        # Make the prediction
        result = model_prediction(file_path, model)
        categories_of_deseases = ['Healthy', 'Multiple_deseases', 'Rust', 'Scab']

        # Process result for human
        class_prediction = categories_of_deseases[result.argmax()]
        return class_prediction
    return None

if __name__ == '__main__':
    app.run(debug=True, port=5926)


