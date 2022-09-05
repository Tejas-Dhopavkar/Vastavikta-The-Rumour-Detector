# -*- coding: utf-8 -*-


from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

import tensorflow as tf
from tf.keras.preprocessing.text import Tokenizer
from tf.keras.preprocessing.sequence import pad_sequences
from tf.keras.layers import Embedding
from sklearn.model_selection import train_test_split

import tweepy
import pandas as pd
consumer_key="WvSq4XfQyZTDraeRCaNMaaAyl"
consumer_secret="my8DLwacftk67fx0hUPmwneZFmb3SwfqcYlzunOnnJfXFq7Zl0"
access_token="1272839707731214336-hbdVBxstuqv0NjFXosdE6meQPw1uJR"
access_token_secret="bkOu6sZukeoy0bfXdHrjs6gnQu5VjVYFLPnSwUGBmZSKG"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token,access_token_secret)
api = tweepy.API(auth)

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

#from gevent.pywsgi import WSGIServer

from PIL import Image, ImageChops, ImageEnhance
import os
import itertools

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='model_casia_run1.h5'
MODEL_PATH ='ELACNN_TrainTest(Adam_CasiaColab).h5'
MODEL_PATH_LSTM ='16_04_21_text_classifier.h5'
MODEL_LSTM_TWITTER = 'TwitterClassificationLSTM.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model_lstm = load_model(MODEL_PATH_LSTM, compile=False)
model_twitter = load_model(MODEL_LSTM_TWITTER, compile=False)

def convert_to_ela_image(path, quality):
    temp_filename = 'temp_file_name.jpg'
    ela_filename = 'temp_ela.png'
    
    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality = quality)
    temp_image = Image.open(temp_filename)
    
    ela_image = ImageChops.difference(image, temp_image)
    
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    return ela_image




def model_predict(img_path, model):
    image_size = (128, 128)


    image1 = np.array(convert_to_ela_image(img_path, 90).resize(image_size)).flatten() / 255.0


    image1 = image1.reshape(-1, 128, 128, 3)

    class_names = ['fake', 'real']

    y_pred = model.predict(image1)
    y_pred_class = np.argmax(y_pred, axis = 1)[0]
    if class_names[y_pred_class]=='real':
        preds="Real"
    elif class_names[y_pred_class]=='fake':
        preds="Fake"

        
    return preds

@app.route('/Classification', methods=['GET', 'POST'])
def classify():
    return render_template('detection.html')


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    print("HI")
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        print(file_path)
        result=preds
        return result
    return None

combined  =pd.read_csv(r"C:\Users\DELL\Work\Fake news\Fake news with LSTM\new_datset\Combined_dataset.csv")
features = combined['text']
labels = combined['fake']
X_train,X_test,y_train,y_test = train_test_split(features, labels, random_state = 42)
max_words = 2000
max_len = 400
token = Tokenizer(num_words=max_words, lower=True, split=' ')
token.fit_on_texts(X_train.values)

tweet_data  =pd.read_csv(r"C:\Users\DELL\Work\Fake news\Fake news with LSTM\new_datset\Twitter.csv")
tweet_features = tweet_data['features']
tweet_labels = tweet_data['labels']
X_train_tweet,X_test_tweet,y_train_tweet,y_test_tweet = train_test_split(tweet_features, tweet_labels, random_state = 42)
token_tweet = Tokenizer(num_words=max_words, lower=True, split=' ')
token_tweet.fit_on_texts(X_train_tweet.values)

@app.route('/predict1', methods=['GET', 'POST'])
def twitter():
    print("HI")
    if request.method == 'POST':
        first_name = request.form.get("abc")
        tweetCount = int(request.form.get("noOfTweets"))
        cursor = tweepy.Cursor(api.user_timeline, id=first_name, tweet_mode="extended").items(tweetCount)
        #print(first_name)
        #print("Hello")
        # Make prediction
        class_names = ['fake', 'real']
        max_words=2000
        max_len=400
        l = []
        for i in cursor:
            j=i.full_text
            l.append(j)
        relist = []
        print(l)
        #token = Tokenizer(num_words=max_words, lower=True, split=' ')
        for i in range(len(l)):
            empty = []
            empty.append(l[i])
            seq = token_tweet.texts_to_sequences(empty)
            print(seq)
            padded = pad_sequences(seq, maxlen=max_len)
            pred = model_twitter.predict(padded)
            list1=pred.tolist()
            efg=list1[0][0]
            x=round(efg)
            k=class_names[x]
            relist.append(k)
            # if pred > 0.5:
            #     ans_1 = 1
            # else:
            #     ans_1 = 0
            # k = class_names[ans_1]
            # relist.append(k)
    return render_template("detection.html", first_name=l, result1=relist)

@app.route('/predict2', methods=['GET', 'POST'])
def classifyText():
    if request.method == 'POST':
        input_text = request.form.get("classifyText")
        class_names = ['real', 'fake']
        max_words=2000
        max_len=400

        l1=[]
        l1.append(input_text)
        print(l1)
        #token = Tokenizer(num_words=max_words, lower=True, split=' ')
        #token.fit_on_texts(l1)
        seq = token.texts_to_sequences(l1)
        print(seq)
        print("\n")
        padded = pad_sequences(seq, maxlen=max_len)
        print(padded)
        pred = model_lstm.predict(padded)
        print(pred)
        #list1=pred.tolist()
        #efg=list1[0][0]
        #x=round(efg)
        #k=class_names[x]
        if pred > 0.5:
            ans_1 = 1
        else:
            ans_1 = 0
        print(ans_1)
        k = class_names[ans_1]
    return render_template("detection.html", classifiedText=input_text, textResult=k)

if __name__ == '__main__':
    app.run(port=5001,debug=True)
