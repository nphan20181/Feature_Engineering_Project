import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, flash, request, redirect, url_for, jsonify, Markup, send_file
from werkzeug.utils import secure_filename
import boto3
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model



UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 

# Entry point for the website
@app.route("/", methods=['POST', 'GET'])
def index():
    return render_template('index.html')


# handle file upload
@app.route('/upload', methods=['GET', 'POST'])
def upload():
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
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('display_result',
                                    filename=filename))
    return render_template('upload.html')


@app.route('/url', methods=['GET', 'POST'])
def check_url():
    return render_template('url.html')


@app.route('/api/images', methods=['GET', 'POST'])
def images():
     if request.method == 'GET':
        filename = request.args.get('filename')
        return send_file('uploads/'+filename, mimetype='image/gif') 

@app.route('/result', methods=['GET'])
def display_result():
    filename = request.args.get('filename')

    # build data frame that store result for image classification
    df_predictions = pd.DataFrame(columns=['Model', 'Predicted Flower Class', 'Probability'])

    # load uploaded image
    uploaded_image = plt.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # resize image
    resize_img = cv2.resize(uploaded_image, (150,150))

    #convert to np from image
    img_as_np = np.array(resize_img)

    df_predictions, x1 = gaussian_filter_cnn_classify(df_predictions, img_as_np)
    df_predictions, x2 = gabor_filter_cnn_classify(df_predictions, img_as_np)
    df_predictions, x3 = sharpening_filter_cnn_classify(df_predictions, img_as_np)
    df_predictions = merged_model_ccn(df_predictions, [x1, x2, x3])
    df_predictions = aws_rekognition_classify(filename, df_predictions)
    
    return render_template('result.html', url=filename, predictions=df_predictions)

def get_flower_name_from_class(class_number):

    flower_dict = {0: "Daisy", 1: "Dandelion", 2: "Rose", 3: "Sunflower", 4: "Tulip"}
    return flower_dict[int(class_number)]

def gaussian_filter_cnn_classify(df_predictions, img_np):

    #apply gaussian filter on resized image
    gauss_feature = cv2.GaussianBlur(img_np, (3, 33), 0)

    #normalize gaussian feature
    gaussx = np.array(gauss_feature)/255

    #convert to list, as model input is based on list.
    images_list = []
    images_list.append(gaussx)
    x = np.asarray(images_list)

    # invoke trained model
    model = load_model('modelgaussbest.h5')

    result = model.predict(x)

    print("----gaussian model result")
    print(result)

    predicted_class = np.asscalar(np.argmax(result,axis=1))

    print("predicted_class")
    print(predicted_class)

    prob = np.asscalar(result[0][predicted_class])
    print("probability")
    print(prob)

    # save prediction to data frame
    df_predictions = df_predictions.append({'Model': 'Gaussian Filter CNN', \
                                            'Predicted Flower Class': get_flower_name_from_class(predicted_class),\
                                            'Probability': str(prob)}, ignore_index=True)
    return df_predictions, x

def gabor_filter_cnn_classify(df_predictions, img_np):


    #apply gabor filter on resized image
    g_kernel = cv2.getGaborKernel((13, 13), 4.0, 56.2, 10.0, 1, 0, ktype=cv2.CV_32F)
    gabor_feature = cv2.filter2D(img_np, cv2.CV_8UC3, g_kernel)

    #normalize gabor feature
    gaborx = np.array(gabor_feature)/255

    #convert to list, as model input is based on list.
    images_list = []
    images_list.append(gaborx)
    x = np.asarray(images_list)

    # invoke trained model
    model = load_model('modelgaborbest.h5')

    result = model.predict(x)

    print("----gabor model result")
    print(result)

    predicted_class = np.asscalar(np.argmax(result,axis=1))

    print("predicted_class")
    print(predicted_class)

    prob = np.asscalar(result[0][predicted_class])
    print("probability")
    print(prob)

    # save prediction to data frame
    df_predictions = df_predictions.append({'Model': 'Gabor Filter CNN', \
                                            'Predicted Flower Class': get_flower_name_from_class(predicted_class),\
                                            'Probability': str(prob)}, ignore_index=True)
    return df_predictions, x

def sharpening_filter_cnn_classify(df_predictions, img_np):

    #apply sharp filter on resized image
    skernel = np.array([[-1,-1,-1], [-1,10,-1], [-1,-1,-1]])
    sharp_feature = cv2.filter2D(img_np, -1, skernel)

    #normalize gaussian feature
    sharpx = np.array(sharp_feature)/255

    #convert to list, as model input is based on list.
    images_list = []
    images_list.append(sharpx)
    x = np.asarray(images_list)

    # invoke trained model
    model = load_model('modelsharpbest.h5')

    result = model.predict(x)

    print("----Sharpening model result")
    print(result)

    predicted_class = np.asscalar(np.argmax(result,axis=1))

    print("predicted_class")
    print(predicted_class)

    prob = np.asscalar(result[0][predicted_class])
    print("probability")
    print(prob)

    # save prediction to data frame
    df_predictions = df_predictions.append({'Model': 'Sharpening Filter CNN', \
                                            'Predicted Flower Class': get_flower_name_from_class(predicted_class),\
                                            'Probability': str(prob)}, ignore_index=True)
    return df_predictions, x


def merged_model_ccn(df_predictions, model_input):

    # invoke trained model
    model = load_model('merged_model.h5')

    result = model.predict(model_input)


    print("----Merged model result")
    print(result)

    predicted_class = np.asscalar(np.argmax(result,axis=1))

    print("predicted_class")
    print(predicted_class)

    prob = np.asscalar(result[0][predicted_class])
    print("probability")
    print(prob)

    # save prediction to data frame
    df_predictions = df_predictions.append({'Model': 'Merged Model CNN', \
                                            'Predicted Flower Class': get_flower_name_from_class(predicted_class),\
                                            'Probability': str(prob)}, ignore_index=True)
    return df_predictions


def aws_rekognition_classify(filename, df_predictions):

    #initialize rekogniton client
    rekognition = boto3.client(
    'rekognition',
    aws_access_key_id='ASIA2DV7S2FXT3N7NOFE',
    aws_secret_access_key='tKMW7heeCiLRvIdu+NtaZ0ws00u/NFfTrsIiywo1',
    aws_session_token='IQon1a1g1a1Jb3JpZ2luX2VjEGMaCXVzLWVhc3QtMSJHMEUCIA/VatdN7fYDnPT9XfgeVqus1F9j3lrK1pKVgI6rSI5NAiEA6xRrV/IJjGBf00VuJHNe2iD/ZCSUqrs50OWKHVqlScwq7wEIOxAAGgw2OTUxMTI3NDk0MjMiDHd+90ki7ARnuYSZoyrMAb5+r+8dekdQO6VIfTaodaEgVSLdlm6ol5eI7Qolo/RNqa5bpb9CZoEOd0PcTAlVjBC0FpAuZm+FGlPLd8X1clODol3j3kiQvkY82dUiTmO8rM53V5D7euOW202PKsnNYHFp92sc9eHywQU7nhhF1bI9on0AIHl0gjx6BRCadSIEBY1LceTkNxIvj8FoH+KWrnmr06pT4v+gw7q45dbRAYjASSMNhS7qO7Lg3P9+aJMSyLpPe4Fuy5CPY75DJzXah6ChuvE+Y9PbJmKcSTCY+6WNBjqYAbT4oJ/PykFBAIZ2ykEuS0chupLkYWtP3FEmJjhAHZGmtqGY5nyD7sudksG3XnhvuIhbNXSsLWfrGzBKmai0NqYnxhe/S+efaTM2n6jjo9hwaj6PYy2CdoSPVe/B04Y4I5x56xgqyUbOBcEnQHWGHoj2vvONLA0/yEMuD+Ec7eVFKsdnhTiWlVOoZ2DnS+wt89YBp8ohxBYZ')

    #invoke aws rekognition api
    with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'rb') as image_data:
        response_content = image_data.read()
        rekognition_response = rekognition.detect_labels(Image={'Bytes':response_content})


    # Extract labels that we are interested in for flowers of type from dataset used.
    predicted_type = 'Unknown'
    confidence_value = 0
    for label in rekognition_response['Labels']:
        if label['Name'] in ['Dandelion', 'Daisy','Rose', 'Sunflower', 'Tulip']:
            predicted_type = label['Name']
            confidence_value = label['Confidence']



    # save prediction to data frame
    df_predictions = df_predictions.append({'Model': 'Amazon Rekognition API', \
                                            'Predicted Flower Class': predicted_type,\
                                            'Probability': confidence_value * 0.01}, ignore_index=True)
    return df_predictions