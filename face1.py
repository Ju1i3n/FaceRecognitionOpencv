# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import sys
import cv2
from cv import CV_CAP_PROP_FRAME_COUNT, CV_CAP_PROP_FPS

if len (sys.argv) > 1:
  fileName=sys.argv[1]
else:
  print("Usage : %s nom de fichier vidéo" %sys.argv[0])
  fileName="IR001.avi"
vidFile = cv2.VideoCapture( fileName )
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

nFrames = int(  vidFile.get( CV_CAP_PROP_FRAME_COUNT ) )
fps = vidFile.get( CV_CAP_PROP_FPS )
waitPerFrameInMillisec = int( 1/fps * 1000/1 )

print ('Num. Frames = ', nFrames)
print ('Frame Rate = ', fps, ' frames per sec')

for f in xrange( nFrames ):
  ret, img = vidFile.read()
  
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
  
  cv2.imshow('img',img)
  if cv2.waitKey( waitPerFrameInMillisec  ) >= 0:
    break

# When playing is done, delete the window
#  NOTE: this step is not strictly necessary, 
#         when the script terminates it will close all windows it owns anyways
cv2.destroyAllWindows()
