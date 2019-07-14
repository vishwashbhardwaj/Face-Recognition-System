import cv2
import numpy as np 
import sqlite3
import os
import Helper
conn = sqlite3.connect('database.db')
if not os.path.exists('./dataset'):
    os.makedirs('./dataset')
c = conn.cursor()
face_cascade = cv2.CascadeClassifier('Haar/haarcascade_frontalcatface.xml')
eye_cascade = cv2.CascadeClassifier('Haar/haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
uname = input("Enter your name: ")
c.execute('INSERT INTO users (name) VALUES (?)', (uname,))
uid = c.lastrowid
sampleNum = 0
Count=0
WHITE = [255, 255, 255]
while Count < 50:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.average(gray) > 110:                                                                      
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)                                         
        for (x, y, w, h) in faces:                                                                  
            FaceImage = gray[y - int(h / 2): y + int(h * 1.5), x - int(x / 2): x + int(w * 1.5)]    
            Img = (Helper.DetectEyes(FaceImage))
            cv2.putText(gray, "FACE DETECTED", (x+w, y-5), cv2.FONT_HERSHEY_DUPLEX, 1, WHITE)
            if Img is not None:
                frame = Img                                                                         
            else:
                frame = gray[y: y+h, x: x+w]
            cv2.imwrite("dataSet/User." + str(uid) + "." + str(Count) + ".jpg", frame)
            cv2.waitKey(300)
            cv2.imshow("CAPTURED PHOTO", frame)                                                     
            Count = Count + 1
    cv2.imshow('Face Recognition System Capture Faces', gray)                                       
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print('FACE CAPTURE FOR THE SUBJECT IS COMPLETE')
cap.release()
conn.commit()
conn.close()
cv2.destroyAllWindows()
