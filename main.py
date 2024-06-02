import tensorflow as tf
from tensorflow.keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import *
import PIL
from PIL import ImageTk, Image, ImageDraw, ImageGrab
import keyboard
from matplotlib import pyplot as plt
import imutils
from imutils.contours import sort_contours


def load_az(filePath):
    #Load in training data for letters A-Z
    data = []
    labels = []
    
    for row in open(filePath):
        row = row.split(',')
        label = int(row[0])
        #Image = row without first value
        image = np.array([int(x) for x in row[1:]],dtype='uint8')

        #Reshape for it to be 28,28 pixels to match mnist dataset
        image = image.reshape((28,28))
        
        data.append(image)
        labels.append(label)

    data = np.array(data,dtype='float32')
    labels = np.array(labels,dtype='int')
    return data,labels

def load_mnist():
    #Load in mnist dataset
    input_shape = (28,28,1)
    ((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()
    #data = np.vstack([trainData,testData])
    #labels = np.hstack([trainLabels,testLabels])
    
    #Reshape and normalize
    trainData = trainData.reshape(trainData.shape[0],trainData.shape[1],trainData.shape[2],1)
    trainData = trainData/255
    testData = testData.reshape(testData.shape[0],testData.shape[1],testData.shape[2],1)
    testData = testData/255
    trainLabels = tf.one_hot(trainLabels.astype(np.int32), depth=10)
    testLabels = tf.one_hot(testLabels.astype(np.int32),depth = 10)
    
    
    return input_shape,trainData,trainLabels,testData,testLabels

def train():
    batch_size = 64
    num_classes = 10
    epochs = 5
    input_shape,trainData,trainLabel,testData,testLabels = load_mnist()
    path = 'numberModel.h5'
    if not os.path.exists(path): #If previous trained model does not exist, train
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
            ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        history = model.fit(trainData, trainLabel,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.1)

        model.save('numberModel.h5')
    else:
        model = tf.keras.models.load_model(path)
    
    #Convert image to array
    imar = convert()
    #Predict reult
    prediction = model.predict(imar)
    result = np.argmax(prediction)

    print(result)
    
def test():
    #Creates drawing board
    global canvas,root
    root = Tk()
    root.geometry('800x800')

    canvas = Canvas(root, bg='white')
    canvas.pack(anchor = 'nw', fill = 'both',expand = 1)

    canvas.bind("<Button-1>",get_coord)
    canvas.bind("<B1-Motion>",draw)
    while True:
        if keyboard.is_pressed('enter'):
            #Save and print predicted number when enter is pressed
            save()
            train()
        elif keyboard.is_pressed('escape'):
            #Clear board if escape is pressed
            canvas.delete('all')
        root.update()
        #root.mainloop()
    


def get_coord(event):
    global x_old, y_old
    x_old, y_old = event.x, event.y

def draw(event):
    #Draw line when button is held
    global x_old, y_old
    canvas.create_line((x_old,y_old,event.x,event.y),width=10)
    x_old,y_old = event.x,event.y

def save():
    #Crop and save image
    x0 = canvas.winfo_rootx()+50
    y0 = canvas.winfo_rooty()+35
    x1 = x0 + canvas.winfo_width()+185
    y1 = y0 + canvas.winfo_height()+190
    
    print(x0,y0,x1,y1)
    image = ImageGrab.grab(bbox=(x0, y0, x1, y1))
    image.save('draw.png')
    
    #Resize it to 28x28 pixels while keeping some quality
    small = image.resize((28,28),Image.ANTIALIAS)
    small = PIL.ImageOps.invert(small)
    small.save('small.png',quality = 100)

    
def convert():
    #imar = image array
    imar = np.array(cv2.imread('small.png',0)) 
    
    #Normalize
    imar = imar.astype('float32')/255
    
    #Reshape for it to work with CNN
    imar = imar.reshape(1,28,28,1)
    
    return imar


test()