print("[INFO] loading libraries...")
from imutils.video import VideoStream
from functools import cache
from smbus import SMBus
from time import sleep
from gtts import gTTS
import face_recognition
import multitasking
import imutils
import pickle
import dlib
import time
import vlc
import cv2
import os

#initializing model paths
encodingsPath = "/home/bighero/AiGateBak/Models/Encodings.pickle"
landmarksPath = "/home/bighero/AiGateBak/Models/shape_predictor_68_face_landmarks.dat"

#initializing variables
addr = 8

factor = 1.05 #resize frame

xmargin = 70 #pixel margin of error
ymargin = 50 #pixel margin of error

previousTime = 0

#bools for frame skipping
frameSkip = True
process = True

#queue for speach
speechQueue = True 
talkQueue = True

#detections
names = []
dets = []
boxes = []

identificationFrameMultiplier = 69

#loading face encodings
print("[INFO] loading encodings...")
data = pickle.loads(open(encodingsPath, "rb").read())
detector = dlib.get_frontal_face_detector()

#load face landmark shape predictor model
print("[INFO] loading shape predictor...")
predictor = dlib.shape_predictor(landmarksPath)

#initialize I2C bus
print("[INFO] initializing I2C bus...")
bus = SMBus(0)
i2cData = [0, 0]

#start camera video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

#get frame center
frame = vs.read()
height, width,_ = frame.shape
frameSqCenter = [int(width/2),int(height/2)]
center = [frameSqCenter[0]+100,frameSqCenter[1]-50]

cv2.namedWindow("face", cv2.WINDOW_NORMAL)
cv2.setWindowProperty('face', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

#detect faces
def detectFaces():
	global dets
	global boxes
	
	dets = detector(gray)
	boxes = []
	
	###detect faces###
	for det in dets:
		face = (det.top(), det.right(), det.bottom(), det.left())
		boxes.append(face)
		
	#print(boxes)

#identify faces
def getNames():
	global process
	global names
	global name
	
	names = []
	
	###face identification###
	if process: #frame skipping
		encodings = face_recognition.face_encodings(rgb, boxes)

		###find matches### 
		for encoding in encodings:
		    matches = face_recognition.compare_faces(data["encodings"], encoding) #compare located faces to saved encodings
		    name = "Unknown" #name defaults to "Unknown"
		    
			###check matches###
		    if True in matches:
		        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
		        counts = {}

		        for i in matchedIdxs:
		            name = data["names"][i]
		            counts[name] = counts.get(name, 0) + 1
		        name = max(counts, key=counts.get)

	names.append(name)	    

	if frameSkip: #bool to disable/enable frame skipping
		process = not process #skip every other frame to save performance

#draw boxes on detected faces
def drawFaces():
	for ((top, right, bottom, left), name) in zip(boxes, names):
		if top* right * bottom* left != 0:
			top = int(top * r)
			right = int(right * r)
			bottom = int(bottom * r)
			left = int(left * r)
			
			cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 0), 2)

			y = top - 15 if top - 15 > 15 else top + 15
			#cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

#gets facial features of faces
def getFaceLandmarks(_face):
	landmarks = predictor(image=gray, box=_face)
	for n in range(0, 68): #loop over 68 face landmark points and draw a point
		x = landmarks.part(n).x
		y = landmarks.part(n).y
		x *= r
		y *= r
		cv2.circle(img=frame, center=(int(x), int(y)), radius=1, color=(100, 255, 6), thickness=-3)
	return landmarks

#returns the largest face
@cache
def largestFace(_face):
	X1 = int(_face.left()*r)
	Y1 = int(_face.top()*r)
	X2 = int(_face.right()*r)
	Y2 = int(_face.bottom()*r)
		
	faceArea = (X2-X1)*(Y2-Y1)
	
	return faceArea

#get eye center from landmarks
def getEyeCenter(arg):
	landmarks = arg
	###landmark 27 is between the eyes.
	x = landmarks.part(27).x
	y = landmarks.part(27).y
	x*=r
	y*=r
	eyeCenter = [int(x),int(y)]
	cv2.circle(img=frame, center=(int(x), int(y)), radius=4, color=(0, 0, 255), thickness=-6)
	return eyeCenter

#send no data to microcontroller using I2C
def resetI2C():
	try:
		bus.write_i2c_block_data(addr, 0, [0, 0]) 
	except OSError:
		print("could not communicate with i2c bus")
	
#send data to microcontroller using I2C to move head
def MoveHead(_eyeCenter):
	if(_eyeCenter[0] < (center[0] - xmargin)):
		i2cData[0] = 1
		#print("left")
		
	elif(_eyeCenter[0] > (center[0] + xmargin)):
		i2cData[0] = 2
		#print("right")
		
	else:
		i2cData[0] = 0
		#print("stop_w")

	if(_eyeCenter[1] < (center[1] - ymargin)):
		i2cData[1] = 1
		#print("up")
		
	elif(_eyeCenter[1] > (center[1] + ymargin)):
		i2cData[1] = 2
		#print("down")
		
	else:
		i2cData[1] = 0
		#print("stop_h")
		
	print(i2cData)
	try:
		bus.write_i2c_block_data(addr, 0, i2cData) #send data over i2c
	except OSError:
		print("could not communicate with i2c bus")

@multitasking.task
def talk(message,timer):

	global speechQueue
	
	while not(speechQueue):
		pass
		
	speechQueue = False
	
	name = "output.mp3"
	
	tts= gTTS(text=message, lang='en')
	tts.save(name)

	media = vlc.MediaPlayer(name)
	duration = abs(media.get_length()-1)
	media.play()
	sleep(duration)
	sleep(timer)	
	os.remove(name)
	
	speechQueue = True

#Read frames and process
while True:
	#getting camera frames
	frame = vs.read()
	
	frame = cv2.flip(frame,1)
	
	#Draw offset frame center for reference
	cv2.circle(frame, (center[0], center[1]), 2, (0, 0, 255), -4)
	
	#grab frame delta
	currentTime = time.time()
	fps = 1/(currentTime-previousTime)
	fps=int(fps)
	previousTime = currentTime
	cv2.putText(frame, f"fps: {fps}", (30, 40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 2)

	#resize frame
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb = cv2.resize(frame, (0, 0), fx=(1/factor), fy=(1/factor), interpolation=cv2.INTER_NEAREST)
	
	#increase contrast
	lab= cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
	l, a, b = cv2.split(lab)
	clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(16,16))
	cl = clahe.apply(l)
	limg = cv2.merge((cl,a,b))
	rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
	
	gray = cv2.cvtColor(src=rgb, code=cv2.COLOR_BGR2GRAY) #converts RGB frame to grayscale
	r = frame.shape[1] / float(rgb.shape[1]) 

	#face recognition and identification
	try:
		detectFaces()
		drawFaces()
		ClosestFace = sorted(dets, key=largestFace, reverse=True)[0] #sort by face area
		landmarks = getFaceLandmarks(ClosestFace) #process landmarks for only closest face to save performance
		EyeCenter = getEyeCenter(landmarks)
		MoveHead(EyeCenter)
		
		if i2cData[0] == 0 and i2cData[1] == 0:
			getNames()

			if speechQueue:
			
				if talkQueue:
						
					talk("Please stand still.", 0.1)
					
					if 'Unknown' in names:
						talk("Do not enter!", 0.4)
					else:
						talk("Hello" + names[0] + ", please enter.", 3)
					talkQueue = False
			else:
				talkQueue = True
				names = ['Unkown']
	
	except IndexError:
		print("no faces")
		resetI2C()
	
	#show frames
	cv2.imshow("face", frame)
	key = cv2.waitKey(1) & 0xFF

	#quit code when 'q' is pressed
	if key == ord("q"):
		break

cv2.destroyAllWindows() #destroy all windows
vs.stop() #stop reading from camera
exit() # exit program
