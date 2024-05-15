import numpy as np
from keras.preprocessing.image import img_to_array
import cv2 
import keras 
import tensorflow as tf
import os
import dlib


def ProcessImage(image):
    # Save image as file 
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image , [48, 48] , method="bilinear")
    image = tf.expand_dims(image , 0)
    
    # Show the image cv2.imshow("Image", image)
    
    return image
 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
MODEL_PATH = 'FER_model.h5'
model = keras.models.load_model(MODEL_PATH)
import cv2 as cv

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)
# dlib face detector
detector = dlib.get_frontal_face_detector()

def predict(img):
    gray = cv.cvtColor(img , cv.COLOR_BGR2GRAY)
    # Imshow the frame
    rects =  detector(gray , 0)

    if len(rects) >= 1 :
        for rect in rects :
            (x , y , w , h) = rect_to_bb(rect)
            img = gray[y-10 : y+h+10 , x-10 : x+w+10]
            if not (img.shape[0] == 0 or img.shape[1] == 0):
                roi_gray = gray[y:y+h,x:x+w]
                roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)  ##Face Cropping for prediction
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0) ## reshaping the cropped face image for prediction
                r = model.predict(roi)[0]   #Prediction
                # labels = ['Positive','Negative',  'Neutral']
                labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
                prediction = model.predict(roi)[0]   #Prediction
                label=labels[prediction.argmax()]
                return (label, (x, y, w, h))
    return (None, None)


        