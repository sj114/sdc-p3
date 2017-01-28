import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
import math
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None
n_brake_frames = 0
prev_throttle = 0
prev_angle = 0

import cv2
def shift_image(image, prev_angle):
    # Translation
    if prev_angle >= 0 and prev_angle < 0.2:
        tr_x = -2.5
    elif prev_angle < -0.05:
        tr_x = 2.5
    else:
        tr_x = 0
    tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(160, 50))

    return image_tr

@sio.on('telemetry')
def telemetry(sid, data):
    global prev_throttle
    global prev_angle
    global n_brake_frames
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    #image = image.resize((160, 80))
    #image = image.crop((0, 20, 160, 70))
    image = np.asarray(image)
    #image_array = shift_image(image_array, prev_angle)

    shape = image.shape
    #image = image[int(shape[0]/3):shape[0], 0:shape[1]]
    image = image[math.floor(shape[0]/4):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image, (200, 66), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    image_array = image

    transformed_image_array = image_array[None, :, :, :]
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))

    # The driving model currently just outputs a constant throttle. Feel free to edit this.

    # Throttling model
    if steering_angle > 0.2 or steering_angle < -0.2:
        throttle = 0.3
    elif steering_angle > 0.08 or steering_angle < -0.08:
        throttle = -0.05
        if prev_throttle <= 0:
            if n_brake_frames == 10:
                throttle = 0.3
                n_brake_frames = 0
            else:
                n_brake_frames += 1 #Track number of successive braking frames
    else:
        throttle = 0.15
    throttle = 0.15
    prev_throttle = throttle
    prev_angle = steering_angle

    print(steering_angle, throttle)
    send_control(steering_angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        #model = model_from_json(json.load(jfile))
        model = model_from_json(jfile.read())

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
