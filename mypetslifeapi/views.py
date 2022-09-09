from django.shortcuts import render
from django.core.exceptions import BadRequest
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage

import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

# imports for dummy data
from datetime import date, datetime
from time import strftime
import random

def initialize():
    global model, le
    le = LabelEncoder()
    le.fit([
        'air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music'
    ])
    model = keras.models.load_model('model/weights.hdf5')


initialize()


def extract_feature(file_name):
    try:
        audio_data, sample_rate = librosa.load(
            file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T, axis=0)
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None, None
    return np.array([mfccsscaled])


def predict(file_name):
    prediction_feature = extract_feature(file_name)
    predicted_vector = np.argmax(model.predict(prediction_feature), axis=-1)
    predicted_class = le.inverse_transform(predicted_vector)
    return predicted_class[0]


def home(request):
    if request.method == 'GET':
        return render(request, 'home.html')

    elif request.method == "POST":
        request_file = request.FILES['file'] if 'file' in request.FILES else None
        if request_file:
            if request_file.content_type != 'audio/wav' and request_file.content_type != 'audio/wave':
                raise BadRequest(request_file.content_type + ' content type is not accepted. Please try again with a .wav file.')
            fs = FileSystemStorage()
            file = fs.save(request_file.name, request_file)
            fileurl = fs.url(file)
            prediction = predict('.' + fileurl)
            # os.remove('.' + fileurl)

            # dummy data starts here
            
            now = datetime.now()
            hour = int(now.strftime("%H"))

            
            if (hour < 12):
              suggestion = 'Give Him some Milk in Morning'
            elif (hour < 17):
              suggestion = 'GGive Him some Milk in Afternoon'
            elif (hour < 20):
              suggestion = 'Give Him some Milk in Evening'
            else:
              suggestion = 'Give Him some Milk in Night'

            emotions = {
                'happy': random.randint(0, 50),
                'angry': random.randint(0, 50),
            }
            emotions['sad'] = 50 - emotions['happy']
            emotions['fearful'] = 50 - emotions['angry']

            maxEmotion = max(emotions, key = lambda x: emotions[x])

            if (prediction == 'dog_bark'):
                return JsonResponse({
                    'message': 'Dog bark detected ✅',
                    'result': 'Your dog is ' + maxEmotion,
                    'suggestion': suggestion,
                    'emotions': emotions,
                })
            else:
                return JsonResponse({
                    'message': 'Sorry, we didn\'t detected a dog bark'
                })
        else:
            return JsonResponse({
                'message': 'Please upload a wav file to analyze'
            })
