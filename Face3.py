import numpy as np     #importe éléments nécessaires au fonctionnement programme
import cv2
import smtplib
from cv2 import *

vidFile = cv2.VideoCapture( 0 ) #utiliser la caméra connecté 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  #utiliser bibliothèque image visage+yeux
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

waitPerFrameInMillisec = int( 1000/25 )

ok=True

while ok:
  ok, img = vidFile.read()
  if ok:
	  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	  visages = face_cascade.detectMultiScale(gray, 1.3, 5)
	  if len(visages) > 0:
		  print "visage détecté, auto-destruction initiée, trop de laideur détectée"
	  if not len(visages) == 0:
		namedWindow("cam-test",CV_WINDOW_AUTOSIZE)
		imshow("cam-test",img)
		waitKey(0)
		destroyWindow("cam-test")
		imwrite("filename.jpg",img) #sauvegarder image
	  for (x,y,w,h) in visages:
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)   #rectangles autour yeux + visages
			roi_gray = gray[y:y+h, x:x+w]
			roi_color = img[y:y+h, x:x+w]
			yeux = eye_cascade.detectMultiScale(roi_gray)
			for (ex,ey,ew,eh) in yeux:
				cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	  
	  cv2.imshow('img',img)
  ok = cv2.waitKey( waitPerFrameInMillisec  ) < 0   # à l'appui d'une touche, fermer les fenêtres

		
cv2.destroyAllWindows()

 '''if len(visages) > 0
	def sendemail('michaelmaregrande1234568gmail.com','yamoxrochi@gmail.com', cc_addr_list,
              	'sécurité nom du logiciel', 'un individu a récemment été repéré',
             	 login, password,
             	 smtpserver='smtp.gmail.com:587'):
   	 header  = 'From: %s
	' % from_addr
    header += 'To: %s
	' % ','.join(to_addr_list)
    header += 'Cc: %s
	' % ','.join(cc_addr_list)
    header += 'Subject: %s

	' % subject
    message = header + message
 
    server = smtplib.SMTP(smtpserver)
    server.starttls()
    server.login(login,password)
    problems = server.sendmail(michaelmaregrande1234568gmail.com, yamoxrochi@gmail.com, un individu a récemment été repéré)
    server.quit()''' 
