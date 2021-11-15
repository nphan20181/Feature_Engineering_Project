import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, flash, request, redirect, url_for, jsonify, Markup, send_file
from werkzeug.utils import secure_filename
import boto3



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

    df_predictions = gabor_filter_cnn_classify(filename, df_predictions)
    df_predictions = gaussian_filter_cnn_classify(filename, df_predictions)
    df_predictions = sharpening_filter_cnn_classify(filename, df_predictions)
    df_predictions = aws_rekognition_classify(filename, df_predictions)
    
    return render_template('result.html', url=filename, predictions=df_predictions)


def gabor_filter_cnn_classify(filename, df_predictions):



    # save prediction to data frame
    df_predictions = df_predictions.append({'Model': 'Gabor Filter CNN', \
                                            'Predicted Flower Class': 'TO DO',\
                                            'Probability': 'TO DO'}, ignore_index=True)
    return df_predictions

def gaussian_filter_cnn_classify(filename, df_predictions):



    # save prediction to data frame
    df_predictions = df_predictions.append({'Model': 'Gaussian Filter CNN', \
                                            'Predicted Flower Class': 'TO DO',\
                                            'Probability': 'TO DO'}, ignore_index=True)
    return df_predictions


def sharpening_filter_cnn_classify(filename, df_predictions):



    # save prediction to data frame
    df_predictions = df_predictions.append({'Model': 'Sharpening Filter CNN', \
                                            'Predicted Flower Class': 'TO DO',\
                                            'Probability': 'TO DO'}, ignore_index=True)
    return df_predictions



def aws_rekognition_classify(filename, df_predictions):

    #initialize rekogniton client
    rekognition = boto3.client(
    'rekognition',
    aws_access_key_id='ASIAVZKUOEYSHABXNY6B',
    aws_secret_access_key='9n6rR4OeU4ieG0UvIIJr2o09Xc++Un0aVkfsnw18',
    aws_session_token='IQoJb3JpZ2luX2VjELH//////////wEaCXVzLWVhc3QtMSJHMEUCIQD3gyPATfERd10/GVAR8tTv8rUgku9g9PJRP7vD+pKuDgIgXnlOM1e9OfNuSyZUnnWtmn8JeScxQj6Soj72v1ljbDcq7wEIahAAGgwzOTc5OTg0MzM4MjgiDOnT4gI52uBtYovtQyrMAdPawGT+RSgWDspWaxcK55XzQ5dDnmBnum003u10A5soPwEz7mSNcJH75MhSOZ2PHv2wBX1UPVgXYluvV6w7ppzMGsCMcNKHtN2dSXtJzWoh3ARtDOfgejBtvHZhSvpnwdpA5qV9VrGn3Bj9WeASgAL8AE8At/8ToF1PUST5FQlO1Wys8Dl5gBl4uPT/0zIAzQn54pwMQ2cq3xFyByGuqFfNdb2JiJuZczFwt3aryldWtg+7YyuywG9EMlU02xRhP6vas/65q+G69YmuJTD20MaMBjqYAT6e9Sc2hxKPovQpLUd7y3K1IC94e8nH8w8+QL0udROj/LCEhQEu4nvGDuUsc2YE8CnXD7wfY86whAOOJde9eXU4M1FUcQDDxlqCcL+sEplPd0aYKGlp4/8BPVzsG15RCE5Lntpo7TvS83nIOQzhVGMNOAzZquBcrwTcEeviQWk8lB7BuhlYdZfqD11Z1KpYSIwh2kf2FZrt')

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