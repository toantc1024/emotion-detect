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

VideoCapture = cv.VideoCapture(0)

vid = cv.VideoCapture(0)


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
                labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
                index = int(r.argmax())
                out = labels[index]
                return (out, (x, y, w, h))
    return (None, None)

# load image
img = cv.imread('K.jpg')



# while(True): 
      
#     # Capture the video frame 
#     # by frame 
#     ret, frame = vid.read() 
  
#     gray = cv.cvtColor( frame , cv.COLOR_BGR2GRAY)
#     # Imshow the frame
#     cv.imshow("Frame", frame)
#     rects =  detector(gray , 0)

#     if len(rects) >= 1 :
#         for rect in rects :
#             (x , y , w , h) = rect_to_bb(rect)
#             img = gray[y-10 : y+h+10 , x-10 : x+w+10]
#             if img.shape[0] == 0 or img.shape[1] == 0 :
#                 cv.imshow("Frame", frame)
#             else :
#                 # resize image
#                 img = cv.resize(img , (48,48))
#                 # labels = ['Positive','Negative',  'Neutral']
#                 r = loaded_model.predict(img.reshape(1,48,48,1))
#                 labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
#                 index = int(r.argmax())
#                 out = labels[index]

#                 # out = RealtimePrediction(img , model , Le)
#                 cv.rectangle(frame, (x, y), (x+w, y+h),(0, 255, 0), 2)
#                 z = y - 15 if y - 15 > 15 else y + 15
#                 cv.putText(frame, str(out), (x, z), cv.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
#         cv.imshow("Frame", frame)
#     else:
#         cv.imshow("Frame", frame)

#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break
  
# # After the loop release the cap object 
# vid.release() 
# # Destroy all the windows 
# cv.destroyAllWindows() 


