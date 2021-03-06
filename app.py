import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect, make_response
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow.keras as keras

# MNE
from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf, RawArray

# Some utilites
import numpy as np
from util import base64_to_pil, mi_converter, errp_converter

# Declare a flask app
app = Flask(__name__)
path = os.path.dirname(__file__)

# Model saved with Keras model.save()
SEED_MI = f'{path}/models/seed_mi.h5'
SEED_ERRP = f'{path}/models/seed_errp.h5'

# Load data
mi_x = np.load(f'{path}/data/mi/x_test.npy')
mi_y = np.load(f'{path}/data/mi/y_test.npy')
errp_x = np.load(f'{path}/data/errp/x_test.npy')
errp_y = np.load(f'{path}/data/errp/y_test.npy')

# Load seeds
model_mi = keras.models.load_model(SEED_MI)
model_mi.make_predict_function()
model_errp = keras.models.load_model(SEED_ERRP)
model_errp.make_predict_function()

# Build callbacks and optmizers
callbacks = [
    keras.callbacks.ModelCheckpoint(
        f'{path}/models/seed_mi_adapted.h5', save_best_only=False, monitor='val_loss', mode='min'
    ),
]
opt = keras.optimizers.Adam(learning_rate=0.00001)
model_mi.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])

def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

def random_event_choice(indexes, dataset_x, dataset_y):
    random_index = np.random.choice(indexes)
    event_input = dataset_x[random_index]
    return event_input, random_index

def model_predict(event_input, model):
    array_pred = model.predict(np.array([event_input,]))[0]
    pred = array_pred.argmax()
    confidence = array_pred[pred]
    return pred, confidence

def adaptative_process(input_val, pred_lab, model):    
    print('Adaptative process started...')
    input_val = np.array([input_val,])
    pred_lab = np.array([pred_lab,])
    try:
        model.fit(input_val, pred_lab, epochs=5, verbose=1, callbacks = callbacks)
        model = keras.models.load_model(f'{path}/models/seed_mi_adapted.h5')
        print('Adaptative proccess is done.')
        return True, model
    except:
        print('Adaptative proccess didn\'t work.')
        return False

@app.route('/', methods=['GET'])
def index():
    # Main page
    if request.method == "OPTIONS": # CORS preflight
        return _build_cors_preflight_response()
    return _corsify_actual_response(jsonify({
        'HyBCI Web Application':{
            'methods': {
                '/predict': {
                    'intent': ['RH', 'LH', 'BF', 'TG']
                }
            } 
        }
    }))

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == "OPTIONS": # CORS preflight
        return _build_cors_preflight_response()
    if request.method == 'POST':
        global model_mi

        # Get the move intent from post request
        user_input = request.json['intent'].upper()
        if user_input not in ['BF', 'LH', 'RH', 'TG']:
            result = {
                'Error': 'Invalid input.'
            }
            return jsonify(result=result)

        print(f'User input: {user_input}')
        mi_indexes = np.where(mi_y == mi_converter(user_input))[0]
        mi_input, mi_index = random_event_choice(mi_indexes, mi_x, mi_y)
        
        mi_pred, mi_confidence = model_predict(mi_input, model_mi)
        print(f'MI predicted: {mi_converter(mi_pred, True)}')
        print(f'MI confidence: {mi_confidence}')

        if(mi_y[mi_index] == mi_pred):
            # If predict right
            print('Correct.')
            errp_indexes = np.where(errp_y == 2)[0]
        else:
            # If predict wrong
            print('Incorrect.')
            errp_indexes = np.where((errp_y != 2))[0]
        errp_input, errp_index = random_event_choice(errp_indexes, errp_x, errp_y)
        
        errp_pred, errp_confidence = model_predict(errp_input, model_errp)
        print(f'ErrP predicted: {errp_converter(errp_pred)}')
        print(f'ErrP confidence: {errp_confidence}')

        if (errp_pred == 2 and errp_confidence > 0.85):
            model_will_fit, model_mi = adaptative_process(mi_input, mi_pred, model_mi)
        else:
            model_will_fit = False

        # Build request response
        result = {
            'user_input': user_input,
            'mi_model': {
                'mi_predicted': mi_converter(mi_pred, True),
                'mi_confidence': '{:.4f}'.format(mi_confidence)
            },
            'errp_model': {
                'errp_predicted': errp_converter(errp_pred),
                'errp_confidence': '{:.4f}'.format(errp_confidence)
            },
            'model_fit': model_will_fit
        }
        
        # Serialize the result
        return _corsify_actual_response(jsonify(result=result))
    return _corsify_actual_response(jsonify({
        "Error": "Invalid request method."
    }))

if __name__ == '__main__':
    port = os.environ.get("PORT", 5000)
    app.run(debug = False, host = '0.0.0.0', port=port)