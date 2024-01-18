from flask import Flask, render_template, request, redirect, url_for
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'wav', 'mp3', 'mp4'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_image_model(model_path):
    return load_model(model_path)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)  # Apply model-specific preprocessing
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(img_path):
    model = load_image_model('new_image_model.h5')
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    return prediction

# Define routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Get selected model
        selected_model = request.form.get('model_select')

        if selected_model == 'image':
            prediction = predict_image(file_path)
            if prediction[0][0] > 0.5:  # Adjust the threshold as needed
                result = "Fake"
            else:
                result = "Real"
                
            # Use url_for to generate the correct URL for the image
            # image_url = url_for('static', filename=file.filename)
            image_url = url_for('static', filename=os.path.join('uploads', file.filename))


            return render_template('result.html', result=result, image_url=image_url, file=file)

        elif selected_model == 'audio':
            # Implement audio model prediction logic
            result = "Audio prediction not implemented yet."

        elif selected_model == 'video':
            # Implement video model prediction logic
            result = "Video prediction not implemented yet."

        else:
            result = "Invalid model selection."

        return render_template('result.html', result=result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
