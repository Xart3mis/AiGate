import os
import cv2
import numpy as np
import dlib
import joblib
import face_recognition
import pyttsx3
import multitasking
import time
import signal
import vlc

import random
from gtts import gTTS
from time import sleep
from smbus import SMBus
from functools import cache


name = "Unknown" #return back later
reject = False

addr = 8
bus = SMBus(0)

data = [0, 0]

engine = pyttsx3.init()
engine.setProperty('volume',0.25)
engine.setProperty("rate", 120)
talkQueue = True
speechQueue = True

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
height, width,_ = frame.shape

xmargin = 70
ymargin = 50

factor = 1.2

frameSqCenter = [int(width/2),int(height/2)]
center = [frameSqCenter[0]+100,frameSqCenter[1]-50]
faceCenter = center

#clf = joblib.load('model.pkl')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cv2.namedWindow("face", cv2.WINDOW_NORMAL)
cv2.setWindowProperty('face', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

@cache
def getName(index):
	frame_enc = face_recognition.face_encodings(small_rgb_frame)[index]
	name = clf.predict([frame_enc])
	print(name[0])
	
@cache
def largestFace(_face):
	X1 = int(_face.left()*factor)
	Y1 = int(_face.top()*factor)
	X2 = int(_face.right()*factor)
	Y2 = int(_face.bottom()*factor)
		
	faceArea = (X2-X1)*(Y2-Y1)
	
	return faceArea
		
def drawFaces():
	if top * right * bottom * left != 0:
		cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 1)
		cv2.rectangle(frame, (int(left), int(top-30)), (int(right), int(top)), (0, 255, 0), -1)
		cv2.putText(frame, name, (int(left), int(top)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def getFaceLandmarks(_face):
	landmarks = predictor(image=gray, box=_face)
	for n in range(0, 68):
		x = landmarks.part(n).x
		y = landmarks.part(n).y
		x*=factor
		y*=factor
		cv2.circle(img=frame, center=(int(x), int(y)), radius=1, color=(100, 255, 6), thickness=-3)
	return landmarks

def getEyeCenter(arg):
	landmarks = arg
	x = landmarks.part(27).x
	y = landmarks.part(27).y
	x*=factor
	y*=factor
	eyeCenter = [int(x),int(y)]
	cv2.circle(img=frame, center=(int(x), int(y)), radius=4, color=(0, 0, 255), thickness=-6)
	return eyeCenter

@multitasking.task
def talk(message,timer):

	global speechQueue
	
	while not(speechQueue):
		pass
		
	speechQueue = False
	
	sleep(timer)	

	name = "output.mp3"
	
	tts= gTTS(text=message, lang='en')
	tts.save(name)

	media = vlc.MediaPlayer(name)
	duration = abs(media.get_length())
	media.play()
	sleep(duration)
	os.remove(name)
	
	speechQueue = True

def MoveHead():
	if(faceCenter[0] < (center[0] - xmargin)):
		data[0] = 1
		print("left")
		
	elif(faceCenter[0] > (center[0] + xmargin)):
		data[0] = 2
		print("right")
		
	else:
		data[0] = 0
		print("stop_w")

	if(faceCenter[1] < (center[1] - ymargin)):
		data[1] = 1
		print("up")
		
	elif(faceCenter[1] > (center[1] + ymargin)):
		data[1] = 2
		print("down")
		
	else:
		data[1] = 0
		print("stop_h")    

while cap.isOpened():
	ret, frame = cap.read()
	frame = cv2.flip(frame,1)
	small_frame = cv2.resize(frame, (0, 0), fx=(1/factor), fy=(1/factor))
	small_rgb_frame = small_frame[:, :, ::-1]
	frame = cv2.circle(frame, (center[0], center[1]), 2, (0, 0, 255), -4)

	top = 0
	right = 0
	bottom = 0
	left = 0

	gray = cv2.cvtColor(src=small_frame, code=cv2.COLOR_BGR2GRAY)
	faces = detector(gray)
    
	i=0
    
	#Detects all faces
	for face in faces:
		left = int(face.left()*factor)
		top = int(face.top()*factor)
		right = int(face.right()*factor)
		bottom = int(face.bottom()*factor)
		
		#print(f"X1 [{left}]  Y1 [{top}]  X2 [{right}]  Y2 [{bottom}]")
		
		#Face identification
		
		i+=1
		"""
		try:
			getName(i)
		except IndexError:
			print("no faces")
		"""
		drawFaces()
		
        #Detect facial landmarks    
		#getFaceLandmarks(face)
	
	try:
		ClosestFace = sorted(faces, key=largestFace, reverse=True)[0]
		landmarks = getFaceLandmarks(ClosestFace)
		faceCenter = getEyeCenter(landmarks)
		MoveHead()
		
		if data[0] == 0 and data [1] == 0:
			if talkQueue:
				#bus.write_i2c_block_data(addr, 0, [0, 0])

				talk("hello, please stand still",0)
				talk("scanning complete",1)
				
				if reject:			
					talk("sorry, you are not allowed to enter ",1.5)
					name = "Unkown"
					drawFaces()
					
				else:
					talk("please enter",1.5)
					name = "Seif zayed"
					drawFaces()
					reject = True
					
				talkQueue = False			
				
		else:
			name = "Unknown"
			talkQueue = True
			
	except IndexError:
		print("no faces")
		data[0] = 0
		data[1] = 0
		
	#bus.write_i2c_block_data(addr, 0, data)
		

	cv2.imshow('face', frame)

	if cv2.waitKey(1) & 0XFF == ord("q"):
		cv2.destroyAllWindows()
		cap.release()
