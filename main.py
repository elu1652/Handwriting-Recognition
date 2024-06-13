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
from sklearn.preprocessing import LabelBinarizer


FILEPATH = 'azdata.csv'

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
    return (data,labels)

def load_mnist():
    #Load in mnist dataset
    input_shape = (28,28,1)
    ((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()
    #Combine Training and Test data
    data = np.vstack([trainData,testData])
    labels = np.hstack([trainLabels,testLabels])
    
    #Reshape and normalize
    '''trainData = trainData.reshape(trainData.shape[0],trainData.shape[1],trainData.shape[2],1)
    trainData = trainData/255
    testData = testData.reshape(testData.shape[0],testData.shape[1],testData.shape[2],1)
    testData = testData/255
    trainLabels = tf.one_hot(trainLabels.astype(np.int32), depth=10)
    testLabels = tf.one_hot(testLabels.astype(np.int32),depth = 10)'''
    
    
    #return input_shape,trainData,trainLabels,testData,testLabels
    return (data,labels)

def train():
    batch_size = 64
    #Output size
    num_classes = 36
    epochs = 2
    input_shape = (28,28,1)
    #input_shape,trainData,trainLabel,testData,testLabels = load_mnist()
    


    path = 'numbersOnly.h5'
    if not os.path.exists(path): #If previous trained model does not exist, train
        (trainData,trainLabel) = load_mnist()
        (azData,azLabels) = load_az(FILEPATH)

        #Account for numbers
        azLabels += 10

        #Combine
        data = np.vstack([azData,trainData])
        labels = np.hstack([azLabels,trainLabel])

        #Normalize and adjust
        data = np.array(data,dtype='float32')

        data = np.expand_dims(data,axis=-1)
        data /= 255.0

        #Turn label from number to vector
        L = LabelBinarizer()
        labels = L.fit_transform(labels)
        #Train
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (5,5), padding='same', activation='relu', input_shape=input_shape),
            tf.keras.layers.Conv2D(32,(5,5),padding='same',activation='relu'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.Conv2D(64, (3,3), padding='same', activation='relu'),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Dropout(0.25),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation='softmax')
            ])

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
        history = model.fit(data, labels,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.1)

        model.save('numberModel2.h5')
    else:
        model = tf.keras.models.load_model(path)
    
    #Convert image to array
    imar = convert()
    #Predict reult
    prediction = model.predict(imar)
    result = np.argmax(prediction)
    if result > 9:
        result = chr(result+55)
    return result
    
def test():
    #Creates drawing board
    global canvas,root

    root = Tk()
    root.geometry('800x800')

    canvas = Canvas(root, bg='white')
    canvas.pack(anchor = 'nw', fill = 'both',expand = 1)

    canvas.bind("<Button-1>",get_coord)
    canvas.bind("<B1-Motion>",draw)
    canvas.bind("<Button-3>",get_coord)
    canvas.bind("<B3-Motion>",erase)
    while True:
        if keyboard.is_pressed('enter'):
            #Save and print predicted number when enter is pressed
            save()
            detect()
            #train()
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

def erase(event):
    global x_old,y_old
    canvas.create_line((x_old,y_old,event.x,event.y),fill='white',width = 30)
    x_old,y_old = event.x,event.y

def save():
    #Crop and save image
    x0 = canvas.winfo_rootx()+50
    y0 = canvas.winfo_rooty()+35
    x1 = x0 + canvas.winfo_width()+185
    y1 = y0 + canvas.winfo_height()+190
    
    #print(x0,y0,x1,y1)
    image = ImageGrab.grab(bbox=(x0, y0, x1, y1))
    image.save('draw.png')
    '''
    #Resize it to 28x28 pixels while keeping some quality
    small = image.resize((28,28),Image.ANTIALIAS)
    small = PIL.ImageOps.invert(small)
    small.save('small.png',quality = 100)

    #Add contrast to clear up image
    test = cv2.imread('small.png')
    test = cv2.addWeighted(test,5,np.zeros(test.shape,test.dtype),0,10)
    img = Image.fromarray(test)
    img.save('data.png')
    '''
def convert():
    #imar = image array
    imar = np.array(cv2.imread('data.png',0)) 
    
    #Normalize
    imar = imar.astype('float32')/255
    
    #Reshape for it to work with CNN
    imar = imar.reshape(1,28,28,1)

    return imar


def save_coord(x,y,w,h):
    #Crop and save image
    #x += 50
    #y +=50
    x0 = x
    y0 = y
    x1 = x+w
    y1 = y+h
    
    #print(x0,y0,x1,y1)
    #image = ImageGrab.grab(bbox=(x0, y0, x1, y1))
    image = Image.open('cropped.png')
    #image = image.crop((x,y,w,h))
    #image.save('draw1.png')
    
    #Resize it to 28x28 pixels while keeping some quality
    small = image.resize((28,28),Image.ANTIALIAS)
    small = PIL.ImageOps.invert(small)
    small.save('small.png',quality = 100)

    #Add contrast to clear up image
    test = cv2.imread('small.png')
    test = cv2.addWeighted(test,5,np.zeros(test.shape,test.dtype),0,10)
    img = Image.fromarray(test)
    img.save('data.png')

def detect():
    #Load image
    og = cv2.imread('draw.png')
    img = og.copy()
    #Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    #Blur to remove noise
    blur = cv2.GaussianBlur(gray,(5,5),0)

    #Turn background white and letters and numbers black
    ret, thresh1 = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV) 

    #Make number and letter more visible and larger by dilation
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18)) 
    dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1) 

    #Find contours
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, 
                                                    cv2.CHAIN_APPROX_NONE) 

    coords = []
    row = []
    graph = []
    for i in range(54):
        graph.append([])
    for cnt in contours: 
        x, y, w, h = cv2.boundingRect(cnt) 
        
        #Crop out the letter or number
        cropped = img[y:y+h,x:x+w]
        cv2.imwrite('cropped.png',cropped)
        #Save the cropped image then process it
        save_coord(x,y,w,h)
        #Predict
        num = str(train())
        coords.append([x,y,w,h,num])
        r = y//20
        graph[r].append([x,y,w,h,num])
        row.append(r)
        # Draw a rectangle on copied image and print predicted value
        rect = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2) 
        cv2.putText(img, num, (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.imwrite('drawed.png', img)
    coords.sort(key=lambda x:x[0])
    #print(coords)
    graph = [x for x in graph if len(x) > 0]
    
    
    '''
    for i in range(len(coords)-1):
        gap = coords[i+1][0] - (coords[i][0] + coords[i][2])
        if gap <= 30 and abs(coords[i+1][1]-coords[i][1]) <= 20:
            x = coords[i][0] - 20
            y = min(coords[i][1],coords[i+1][1]) - 20
            w = coords[i][2] + gap + coords[i+1][2] + 20
            h = max(coords[i][3],coords[i+1][3]) + 20
            rect = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2) 
            cv2.putText(img, coords[i][4]+coords[i+1][4], (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
'''
    for r in graph:
        r.sort(key=lambda x:x[0])
        for i in range(len(r)-1):
            gap = r[i+1][0] - (r[i][0]+r[i][2])
            if gap <= 50:
                thresh = 30
                x = r[i][0]-thresh
                y = min(r[i][1],r[i+1][1]) - thresh
                w = r[i][2] + gap + r[i+1][2] + thresh
                h = max(r[i][3],r[i+1][3]) + thresh
                rect = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2) 
                cv2.putText(img, r[i][4]+r[i+1][4], (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

    cv2.imwrite('drawed.png', img)
    print(graph)
test()

