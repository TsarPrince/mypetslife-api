from django.shortcuts import render
from django.core.exceptions import BadRequest
from django.http import JsonResponse
from django.core.files.storage import FileSystemStorage

import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras


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
            os.remove('.' + fileurl)
        return JsonResponse({
            'result': prediction
        })
