import os 
from flask import Flask, flash, request, redirect, url_for,render_template
from werkzeug.utils import secure_filename 
from another import transform_image, get_prediction
from PIL import Image
 
UPLOAD_FOLDER = 'F:/copied/static/uploads' 
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif']) 
 
app = Flask(__name__) 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 
app.config['SECRET_KEY']='THisisSECRET_KEY'
 
def allowed_file(filename): 
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
         
 
@app.route('/', methods=['GET', 'POST']) 
def upload_file(): 
    if request.method == 'POST': 
        # check if the post request has the file part 
        if 'file' not in request.files: 
            flash('No file part') 
            return redirect(request.url) 
        file = request.files['file'] 
        # if user does not select file, browser also 
        # submit an empty part without filename 
        if file.filename == '': 
            flash('No selected file') 
            return redirect(request.url) 
        if file and allowed_file(file.filename): 
            filename = secure_filename(file.filename) 
            flash('file {} saved'.format(file.filename)) 
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename)) 
            return render_template('index.html',filename=filename)
    return render_template('index.html')
    ''' 
    <!doctype html> 
    <title>Upload new File</title> 
    <h1>Upload new File</h1> 
    <form method=post enctype=multipart/form-data> 
      <input type=file name=file> 
      <input type=submit value=Upload> 
    </form> 
    ''' 
# @app.route('/uploaded',methods=['GET'])
# def new():
# 	return render_template('prediction.html')
# @app.route('/display/<filename>')
# def display_image(filename):
#     #print('display_image filename: ' + filename)
#     file='uploads/' + filename
#     try:
#         img_bytes = file.read()
#         tensor = transform_image(img_bytes)
#         prediction = get_prediction(tensor)
#         data = {'prediction': prediction.item(), 'class_name': str(prediction.item())}
#         return jsonify(data)
#         # print(data)
#     except:
#     	pass
#     	# return jsonify({'error': 'error during prediction'})
#     return redirect(url_for('static', filename=file, code=301))


@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    file='uploads/' + filename
    # image = Image.open('uploads/' + filename)
    # image = image.resize((256,256),Image.ANTIALIAS)
    # image.save(file)
    try:
        img_bytes = file.read()
        tensor = transform_image(img_bytes)
        prediction = get_prediction(tensor)
        data = {'prediction': prediction.item(), 'class_name': str(prediction.item())}
        # return jsonify(data)
        # print(data)
    except:
    	pass
    	# return jsonify({'error': 'error during prediction'})
    return redirect(url_for('static', filename=file, code=301))


# @app.route('/display')
@app.route('/predict/<filename>',methods=['GET','POST'])
def display_img(filename):
    #print('display_image filename: ' + filename)
    l=['Strawberry___healthy', 'Tomato___Late_blight', 
   'Orange___Haunglongbing_(Citrus_greening)',
   'Corn_(maize)___Northern_Leaf_Blight', 'Tomato___Septoria_leaf_spot', 
   'Corn_(maize)___Common_rust_', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
   'Potato___Early_blight', 'Tomato___Early_blight', 'Strawberry___Leaf_scorch',
   'Potato___Late_blight', 'Raspberry___healthy', 'Tomato___healthy',
   'Pepper,_bell___Bacterial_spot', 'Cherry_(including_sour)___Powdery_mildew', 
   'Cherry_(including_sour)___healthy', 'Apple___Black_rot',
   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Blueberry___healthy',
   'Apple___Cedar_apple_rust', 'Apple___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
   'Tomato___Tomato_mosaic_virus', 'Peach___Bacterial_spot', 'Squash___Powdery_mildew',
   'Grape___Esca_(Black_Measles)', 'Tomato___Bacterial_spot', 'Peach___healthy', 
   'Tomato___Leaf_Mold', 'Tomato___Target_Spot', 'Corn_(maize)___healthy', 
   'Soybean___healthy', 'Grape___Black_rot', 'Pepper,_bell___healthy', 'Grape___healthy', 
   'Tomato___Spider_mites Two-spotted_spider_mite', 'Apple___Apple_scab', 'Potato___healthy']

    l.sort()

    file="F:/copied/static/uploads/" + filename
    file=open(file,'rb')
    img_bytes = file.read()
    tensor = transform_image(img_bytes)
    prediction = get_prediction(tensor)
    data = {'prediction': prediction.item(), 'class_name': str(prediction.item())}
    data=l[data['prediction']]
    return render_template('prediction.html',filename=data)
    
    	# return jsonify({'error': 'error during prediction'})
    # return redirect(url_for('static', filename=data, code=301))

   


if __name__ == '__main__':
	app.run(debug=True) 