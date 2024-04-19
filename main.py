import tensorflow as tf
import numpy as np
import cv2
import dlib
import pickle

# import efficientnet.tfkeras as efn
from tensorflow.keras.models import load_model

import tensorflow as tf
print(tf.__version__)

model = load_model("FacialExpressionModel.h5") # Load the saved weights
tflite_converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = tflite_converter.convert()
open("tf_lite_model.tflite", "wb").write(tflite_model)


# Load LabelEncoder
def load_object(name):
    pickle_obj = open(f"{name}.pck","rb")
    obj = pickle.load(pickle_obj)
    return obj

Le = load_object("LabelEncoder")

def ProcessImage(image):
    # Save image as file 
    image = tf.convert_to_tensor(image)
    image = tf.image.resize(image , [96 , 96] , method="bilinear")
    image = tf.expand_dims(image , 0)
    
    # Show the image cv2.imshow("Image", image)
    
    return image

def RealtimePrediction(image , model, encoder_):
    prediction = model.predict(image)
    prediction = np.argmax(prediction , axis = 1)
    print(prediction)
    return encoder_.inverse_transform(prediction)[0]

def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

VideoCapture = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()

vid = cv2.VideoCapture(0) 
  
while True :

    ret, frame = VideoCapture.read()

    if not ret :
        break

    gray = cv2.cvtColor( frame , cv2.COLOR_BGR2GRAY)

    # Imshow the frame
    cv2.imshow("Frame", frame)
    
    rects = detector(gray , 0)

    if len(rects) >= 1 :
        for rect in rects :
            (x , y , w , h) = rect_to_bb(rect)
            img = gray[y-10 : y+h+10 , x-10 : x+w+10]

            if img.shape[0] == 0 or img.shape[1] == 0 :
                cv2.imshow("Frame", frame)

            else :
                img = cv2.cvtColor(img , cv2.COLOR_GRAY2RGB)
                img = ProcessImage(img)
                out = RealtimePrediction(img , model , Le)
                cv2.rectangle(frame, (x, y), (x+w, y+h),(0, 255, 0), 2)
                z = y - 15 if y - 15 > 15 else y + 15
                cv2.putText(frame, str(out), (x, z), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)

    else :
        cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

VideoCapture.release()
cv2.destroyAllWindows()