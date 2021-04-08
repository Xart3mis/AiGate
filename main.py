print("[INFO] loading libraries...")
from imutils.video import VideoStream
from functools import cache
from smbus import SMBus
import face_recognition
import multitasking
import imutils
import pickle
import dlib
import time
import cv2

#initializing model paths
encodingsPath = "/home/bighero/AiGate/Models/Encodings.pickle"
landmarksPath = "/home/bighero/AiGate/Models/shape_predictor_68_face_landmarks.dat"

#initializing variables
addr = 8

factor = 1.2 #resize frame

xmargin = 70 #pixel margin of error
ymargin = 50 #pixel margin of error

previousTime = 0

#bools for frame skipping
frameSkip = True
process = True

dets = []

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

#Draw boxes on detected faces
def drawFaces(_left, _top, _right, _bottom, _name):
	cv2.rectangle(frame, (_left, _top), (_right, _bottom), (0, 255, 0), 2)

	y = _top - 15 if _top - 15 > 15 else _top + 15
	cv2.putText(frame, _name, (_left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

#Identify and locate faces
def getNames():

	global dets
	global process
	
	names = []

	dets = detector(gray)
	boxes = []
	
	###detect faces###
	for det in dets:
		face = (det.top(), det.right(), det.bottom(), det.left())
		boxes.append(face)
	#print(boxes)
	
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
		
	for ((top, right, bottom, left), name) in zip(boxes, names):
		top = int(top * r)
		right = int(right * r)
		bottom = int(bottom * r)
		left = int(left * r)

		drawFaces(left, top, right, bottom, name)

#function that gets facial features of people in a frame
def getFaceLandmarks(_face):
	landmarks = predictor(image=gray, box=_face)
	for n in range(0, 68): #loop over 68 face landmark points and draw a point
		x = landmarks.part(n).x
		y = landmarks.part(n).y
		x *= r
		y *= r
		cv2.circle(img=frame, center=(int(x), int(y)), radius=1, color=(100, 255, 6), thickness=-3)
	return landmarks

#function that returns the largest face in a frame
@cache
def largestFace(_face):
	X1 = int(_face.left()*r)
	Y1 = int(_face.top()*r)
	X2 = int(_face.right()*r)
	Y2 = int(_face.bottom()*r)
		
	faceArea = (X2-X1)*(Y2-Y1)
	
	return faceArea

#function tha gets eye center from landmarks
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
	bus.write_i2c_block_data(addr, 0, i2cData) #Send data over i2c

#Read frames and process
while True:
	#getting camera frames
	frame = vs.read()
	frame = cv2.flip(frame,1)
	
	#Draw offset frame center for reference
	cv2.circle(frame, (center[0], center[1]), 2, (0, 0, 255), -4)
	
	#grab frame delta
	currentTime = time.time()
	fps = 1//(currentTime-previousTime)
	previousTime = currentTime
	cv2.putText(frame, f"fps: {fps}", (30, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 50), 2)

	#resize frame to process less pixels
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb = cv2.resize(frame, (0, 0), fx=(1/factor), fy=(1/factor))
	gray = cv2.cvtColor(src=rgb, code=cv2.COLOR_BGR2GRAY)
	r = frame.shape[1] / float(rgb.shape[1])

	#face recognition and identification
	try:
		getNames()
		ClosestFace = sorted(dets, key=largestFace, reverse=True)[0] #sort by face area
		landmarks = getFaceLandmarks(ClosestFace) #process landmarks for only closest face to save performance
		EyeCenter = getEyeCenter(landmarks)
		#MoveHead(EyeCenter)
	
	except IndexError:
		print("no faces")
		i2cData = [0, 0]
		
	#show frames
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	#quit code when 'q' is pressed
	if key == ord("q"):
		break

cv2.destroyAllWindows() #destroy all windows
vs.stop() #stop reading from camera
