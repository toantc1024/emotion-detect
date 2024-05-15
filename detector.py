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
MODEL_PATH = 'fer_model.h5'
loaded_model = keras.models.load_model(MODEL_PATH)
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
                img = cv.resize(img , (48,48))
                # labels = ['Positive','Negative',  'Neutral']
                r = loaded_model.predict(img.reshape(1,48,48,1))
                labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Surprise', 'Neutral']
                index = int(r.argmax())
                out = labels[index]
                return (out, (x, y, w, h))
    return (None, None)





