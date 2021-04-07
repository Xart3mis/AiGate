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

addr = 8
bus = SMBus(0)
i2cData = [0, 0]

encodingsPath = "/home/bighero/AiGate/Models/goodboi.pickle"

print("[INFO] loading encodings...")
data = pickle.loads(open(encodingsPath, "rb").read())
detector = dlib.get_frontal_face_detector()

previousTime = 0

frameSkip = True
process = True

print("[INFO] loading shape predictor...")
predictor = dlib.shape_predictor(
    "/home/bighero/AiGate/Models/shape_predictor_68_face_landmarks.dat")

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

factor = 1.2

frame = vs.read()
height, width,_ = frame.shape
frameSqCenter = [int(width/2),int(height/2)]
center = [frameSqCenter[0]+100,frameSqCenter[1]-50]
xmargin = 70
ymargin = 50
time.sleep(2.0)

def getFaceLandmarks(_face):
	landmarks = predictor(image=gray, box=_face)
	for n in range(0, 68):
		x = landmarks.part(n).x
		y = landmarks.part(n).y
		x *= r
		y *= r
		cv2.circle(img=frame, center=(int(x), int(y)),
				   radius=1, color=(100, 255, 6), thickness=-3)
	return landmarks

@cache
def largestFace(_face):
	X1 = int(_face.left()*r)
	Y1 = int(_face.top()*r)
	X2 = int(_face.right()*r)
	Y2 = int(_face.bottom()*r)
		
	faceArea = (X2-X1)*(Y2-Y1)
	
	return faceArea

def getEyeCenter(arg):
	landmarks = arg
	x = landmarks.part(27).x
	y = landmarks.part(27).y
	x*=r
	y*=r
	eyeCenter = [int(x),int(y)]
	cv2.circle(img=frame, center=(int(x), int(y)), radius=4, color=(0, 0, 255), thickness=-6)
	return eyeCenter

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
	bus.write_i2c_block_data(addr, 0, i2cData)

while True:
	frame = vs.read()
	frame = cv2.flip(frame,1)
	
	cv2.circle(frame, (center[0], center[1]), 2, (0, 0, 255), -4)
	
	currentTime = time.time()
	fps = 1//(currentTime-previousTime)
	previousTime = currentTime

	cv2.putText(frame, f"fps: {fps}", (30, 40),
		        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 50), 2)

	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	rgb = cv2.resize(frame, (0, 0), fx=(1/factor), fy=(1/factor))
	gray = cv2.cvtColor(src=rgb, code=cv2.COLOR_BGR2GRAY)
	r = frame.shape[1] / float(rgb.shape[1])

	dets = detector(gray)
	boxes = []
	for det in dets:
		face = (det.top(), det.right(), det.bottom(), det.left())
		boxes.append(face)
	#print(boxes)

	if process:
		encodings = face_recognition.face_encodings(rgb, boxes)
		names = []

		for encoding in encodings:
		    matches = face_recognition.compare_faces(data["encodings"],
		                                             encoding)
		    name = "Unknown"

		    if True in matches:
		        matchedIdxs = [i for (i, b) in enumerate(matches) if b]
		        counts = {}

		        for i in matchedIdxs:
		            name = data["names"][i]
		            counts[name] = counts.get(name, 0) + 1
		        name = max(counts, key=counts.get)

		    names.append(name)

	if frameSkip:
		process = not process
	
	try:
		ClosestFace = sorted(dets, key=largestFace, reverse=True)[0]
		landmarks = getFaceLandmarks(ClosestFace)
		EyeCenter = getEyeCenter(landmarks)
		MoveHead(EyeCenter)
	except IndexError:
		print("no faces")
		i2cData = [0, 0]
		
	for ((top, right, bottom, left), name) in zip(boxes, names):
		top = int(top * r)
		right = int(right * r)
		bottom = int(bottom * r)
		left = int(left * r)

		cv2.rectangle(frame, (left, top), (right, bottom),
		              (0, 255, 0), 2)
		y = top - 15 if top - 15 > 15 else top + 15

		cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
		            0.75, (0, 255, 0), 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()
