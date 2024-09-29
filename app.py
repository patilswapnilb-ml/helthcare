from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import gdown
import os

app = Flask(__name__)

# Download model from Google Drive
url = "https://drive.google.com/uc?id=124UhbC0-gsRuHGPi_6cVRCQwKO_yj0mZ"
output = "your_model_file.h5"
if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

# Load the trained model
model = load_model(output)

# Define the class names based on your dataset
class_names = ['Bacterial leaf blight', 'Brown spot', 'Healthy', 'Leaf Blast', 'Leaf scald', 'Narrow Brown Spot']

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_class_name = None  # Initialize as None to pass to the template
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded", predicted_class_name=None)

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No selected file", predicted_class_name=None)

        if file:
            # Load the image
            image = load_img(file, target_size=(299, 299))

            # Preprocess the image
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = image / 255.0

            # Make predictions
            predictions = model.predict(image)
            predicted_class = np.argmax(predictions, axis=1)
            predicted_class_name = class_names[predicted_class[0]]

    return render_template('index.html', predicted_class_name=predicted_class_name)

if __name__ == "__main__":
    app.run(debug=True)
