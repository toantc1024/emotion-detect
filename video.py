from detector import predict
import cv2 as cv    


VideoCapture = cv.VideoCapture(0)

vid = cv.VideoCapture(0)



while(True): 
      
    # Capture the video frame 
    # by frame 
    ret, frame = vid.read() 

    out, (x, y, w, h) = predict(frame)
    if out is None:
        cv.imshow("Frame", frame)
    else:
        cv.rectangle(frame, (x, y), (x+w, y+h),(0, 255, 0), 2)
        z = y - 15 if y - 15 > 15 else y + 15
        cv.putText(frame, str(out), (x, z), cv.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)
        cv.imshow("Frame", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv.destroyAllWindows() 
