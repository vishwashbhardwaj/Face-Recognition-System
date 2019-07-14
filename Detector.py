import cv2
import Helper
import numpy as np 
import sqlite3
import os
conn = sqlite3.connect('database.db')
c = conn.cursor()
face_cascade = cv2.CascadeClassifier('Haar/haarcascade_frontalcatface.xml')
eye_cascade = cv2.CascadeClassifier('Haar/haarcascade_eye.xml')
fname = "recognizer/trainingData.yml"
if not os.path.isfile(fname):
  print("Please train the data first")
  exit(0)
cap = cv2.VideoCapture(0)
recognizer = cv2.face.LBPHFaceRecognizer_create(2, 2, 7, 7, 15)
recognizer.read(fname)
while True:
  ret, img = cap.read()
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray,1.2, 5)
  for (x,y,w,h) in faces:
          gray_face = gray[y: y+h, x: x+w]
          eyes = eye_cascade.detectMultiScale(gray_face)
          
          for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(img,(x,y),(x+w,y+h),(225,0,0),1)
                ids, conf = recognizer.predict(gray_face)
                print('confidence'+str(round(conf, 2)))
                c.execute("select name from users where id = (?);", (ids,))
                result = c.fetchall()
                name = result[0][0]   # Determine the ID of the photo
                if(conf <2):
                  output=name;
                else:
                  output="Unknown"
                
                cv2.putText(img,str(output),(x-50, h),cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0,0), lineType=cv2.LINE_AA)
                cv2.putText(img,str(round(conf,2)),(x+90, h-10),cv2.FONT_HERSHEY_SIMPLEX,1, (255, 0,0), lineType=cv2.LINE_AA)
                
                
                Helper.DispID(x, y, w, h, name, gray)
                
  cv2.imshow('Face Recognizer',img)
  k = cv2.waitKey(30) & 0xff
  if k == 27:
    break
cap.release()
cv2.destroyAllWindows()
