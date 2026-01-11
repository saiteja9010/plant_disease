import io
from flask import Flask, request, render_template, flash
from werkzeug.utils import secure_filename
from another import transform_image, get_prediction
from PIL import Image

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.secret_key = "THisisSECRET_KEY"


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# LABELS
CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]


@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':

        if 'file' not in request.files:
            flash("No file uploaded")
            return render_template('index.html')

        file = request.files['file']

        if file.filename == '':
            flash("No selected file")
            return render_template('index.html')

        if file and allowed_file(file.filename):
            # READ IMAGE DIRECTLY FROM MEMORY (NO DISK)
            img = Image.open(io.BytesIO(file.read())).convert("RGB")

            # TRANSFORM & PREDICT
            tensor = transform_image(img)
            prediction = get_prediction(tensor)
            result = CLASSES[prediction]

            return render_template('prediction.html', result=result)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
