'''
Title: Udacity SDC P3 Behavioral Cloning
Author: Soujanya Kedilaya

This module implements the keras training module.
'''

# Load the modules
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Lambda
from keras.layers import Activation, Dropout, Flatten, Dense, ELU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
import pickle
import math
import numpy as np
import h5py
import argparse
import csv
from PIL import Image, ImageOps
import cv2
import os
import sys

# dimensions of our images.
orig_img_width, orig_img_height = 320, 160
img_width, img_height = 160, 50 

# global variable to save sample training images
global_show_img = True 

''' Function to apply any image augmentations or transformations
The following are done here:
    - image resizing
    - cropping unwanted sections of the image
    - random flipping
'''
def transform_image(drive_data_entry, steer_angle):
    global global_show_img

    i_img = np.random.randint(1)
    offset_angle = 0.2

    if (i_img == 1): #left
        img_name = drive_data_entry[1]
        steer_angle += offset_angle
    elif (i_img == 2): #right
        img_name = drive_data_entry[2]
        steer_angle -= offset_angle
    else: #center
        img_name = drive_data_entry[0]

    if os.path.exists(img_name):

        image = cv2.imread(img_name)
        if global_show_img == True:
            cv2.imwrite("test_orig.png", image)
        
        # Resize and crop
        dim = (int(orig_img_width/2), int(orig_img_height/2))
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        if global_show_img == True:
            print ("dim: ", dim)
            cv2.imwrite("resized.png", image)

        image = image[20:img_height+20, 0:img_width]
        if global_show_img == True:
            cv2.imwrite("cropped.png", image)

        # Random flip
        ind_flip = np.random.randint(2)
        if ind_flip:
            image = cv2.flip(image, 1)
            if global_show_img == True:
                cv2.imwrite("flipped.png", image)
                global_show_img = False
            steer_angle = -steer_angle
        image = np.array(image, dtype=np.float32)
    else:
        print (img_name)
        sys.exit("ERROR: Image doesn't exist!")
    
    return image, steer_angle

''' Function to generate batch data
The following are done here:
    - Randomly selects an entry from driving log
    - Applies any applicable image transformations
    - Returns a batch of data
'''
def get_batch_train_data(X_train, Y_train, batch_size):
    
    n_train = len(Y_train)
    print ("Number of training samples: ", n_train)

    batch_x = np.zeros((batch_size, img_height, img_width, 3))
    batch_y = np.zeros(batch_size)
    while 1:
        for i_batch in range(batch_size):
            i_log = np.random.randint(n_train)
            image_path = X_train[i_log] 
                
            image, steer_angle = transform_image(image_path, float(Y_train[i_log]))
            
            batch_x[i_batch] = image
            batch_y[i_batch] = steer_angle 
        yield batch_x, batch_y

''' Function that loads images for the validation data from the image paths '''
def load_val_data(X_val, Y_val):

    n_val = len(X_val)
    X_val_image = np.zeros((n_val, img_height, img_width, 3))
    Y_steer = np.zeros(n_val)

    for i in range(n_val):
        img_name = X_val[i][0]

        if os.path.exists(img_name):
            image = cv2.imread(img_name)
            if i == 0:
                print("Val image path: ", img_name, "n_val: ", n_val)
                cv2.imwrite("val_orig.png", image)
            
            # Resize and crop
            dim = (int(orig_img_width/2), int(orig_img_height/2))
            image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
            if i == 0:
                print ("dim: ", dim)
                cv2.imwrite("val_resized.png", image)

            image = image[20:img_height+20, 0:img_width]
            if i == 0:
                cv2.imwrite("val_cropped.png", image)

            image = np.array(image, dtype=np.float32)
            X_val_image[i] = image
            Y_steer[i] = float(Y_val[i])
        else:
            print (img_name)
            sys.exit("ERROR: Image doesn't exist!")

    return X_val_image, Y_steer


''' Function to create the deep learning model '''
def create_model():

    # base number of convolutional filters to use
    nb_filters = 16 

    # convolution kernel size
    kernel_size = (5, 5)
    kernel_size_s = (3, 3)
    
    # size of pooling area for max pooling
    pool_size = (2, 2)

    model = Sequential()

    # input normalization to fit data between -1 to 1
    model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(img_height, img_width, 3)))

    # convolution and non-linear layers with dropout and max pooling
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='same', subsample=(2,2)))
    model.add(ELU())
    model.add(Convolution2D(nb_filters*2, kernel_size[0], kernel_size[1],
                            border_mode='same', subsample=(2,2)))
    model.add(ELU())
    
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Convolution2D(nb_filters*3, kernel_size[0], kernel_size[1],
                            border_mode='same', subsample=(2,2)))
    model.add(ELU())
    model.add(Convolution2D(nb_filters*4, kernel_size_s[0], kernel_size_s[1],
                            border_mode='same'))
    model.add(ELU())
    
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.25))

    model.add(Convolution2D(nb_filters*4, kernel_size_s[0], kernel_size_s[1],
                            border_mode='same'))
    model.add(ELU())
    model.add(Convolution2D(nb_filters*4, kernel_size_s[0], kernel_size_s[1],
                            border_mode='same'))
    model.add(ELU())

    model.add(MaxPooling2D(pool_size=pool_size, dim_ordering="th"))
    model.add(Dropout(0.25))

    model.add(Flatten())

    # fully connected layers with dropout and non-linearities
    model.add(Dense(100))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(ELU())
    model.add(Dropout(0.5))
    model.add(Dense(1))
    
    model.summary()
    return model

''' Function to train model '''
def train_model(model, drive_data, train_from_scratch, n_epoch):

    # Hyperparameters
    batch_size = 32 

    # Initial learning rate and fine tuning rates differ
    if train_from_scratch:
        learning_rate = 0.0001
    else:
        learning_rate = 0.00001
        print ("Learning rate: ", learning_rate)

    # Set up optimizer properties
    optimizer = Adam(lr=learning_rate, decay=0)
    model.compile(loss='mse', optimizer=optimizer)

    # Load saved weights if fine tuning
    if not train_from_scratch:
        model.load_weights('model.h5')
        print ("Using baseline model.h5")

    # Split the training data into training and validation
    Y_train = [item.pop(3) for item in drive_data]
    X_train, X_val, Y_train, Y_val = train_test_split(drive_data, Y_train, test_size=0.05, random_state=221216)
    X_val_images, Y_steer = load_val_data(X_val, Y_val)

    # Fits the model on batches with real-time data augmentation:
    for i_pr in range(1, n_epoch): 
        history = model.fit_generator(get_batch_train_data(X_train, Y_train, batch_size),
                                      samples_per_epoch=len(Y_train)*2.0, nb_epoch=1, 
                                      validation_data=(X_val_images, Y_val), 
                                      verbose=1) 

        # Save weights after each epoch
        model_json = model.to_json()
        with open("model_{}.json".format(i_pr), 'w') as json_file:
            json_file.write(model_json)
            model.save_weights("model_{}.h5".format(i_pr))

    # Save final weights and architecture  
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--init", help="Train the model from scratch",
                        action="store_true")
    parser.add_argument("-e", "--epoch", type=int, default=1, help='Number of epochs.')
    args = parser.parse_args()

    if not args.init:
        print ("Using baseline model.h5")

    print ("Number of epochs: ", args.epoch)

    # Open driving data csv file 
    with open('../behavioral-cloning/training_data_segments/driving_log.csv', newline='') as csvfile:
        drive_data_reader = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
        drive_data = list(drive_data_reader)

    model = create_model()
    train_model(model, drive_data, args.init, args.epoch)

if __name__ == '__main__': main()