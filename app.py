from flask import Flask, request, render_template
from tensorflow import keras
from keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
model = load_model('pneumonia_detection_model.h5')

# Define the predict route
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:  # Check if file part is in the request
        return render_template('index.html', result="No file uploaded")

    file = request.files['file']
    
    if file.filename == '':  # Check if file has a name
        return render_template('index.html', result="No file selected")
    
    if file:
        # Convert the file to an image and preprocess
        img = Image.open(file.stream)
        img = img.resize((150, 150))  # Resize image to match model input size
        img = np.array(img) / 255.0  # Normalize the image

        # Add an extra batch dimension as the model expects a batch of images
        img = np.expand_dims(img, axis=0)

        # Get prediction
        prediction = model.predict(img)
        
        # Check the prediction
        if prediction[0] >=0.8:
            result = "Pneumonia is detected"
        elif (prediction[0]>0.6 and prediction[0]<0.8):
            result="There is a strong possiblility that you have contracted Pneumonia"
        elif(prediction[0]>0.4 and prediction[0]<0.6):
            result="Mild stage of Pnuemonia is detected"
        else:
            result = "Normal"

        return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
