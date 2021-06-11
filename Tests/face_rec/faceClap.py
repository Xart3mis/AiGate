#----------------------------------------------------< Initializations >--------------------------------------------------

#Modules
print("[INFO] Loading Modules")
from imutils.video import VideoStream
from sklearn import svm
import face_recognition
import numpy as np
import pickle
import time
import cv2
import dlib
import os

#Variables
name = "Unknown"
model = "KNN" # SVM or KNN only
distanceThreshold = 55 # in %
previousTime = 0

#Directories
predictor_path =  "/home/bighero/AiGate/Models/shape_predictor_68_face_landmarks.dat"
model_path =     f"/home/bighero/AiGate/Tests/face_rec/model/{model}_Classifier.pickle"
encodings_path =  "/home/bighero/AiGate/Tests/face_rec/model/Encodings.pickle"
output_image =    "/home/bighero/AiGate/Tests/face_rec/output.jpg"

#Models
print("[INFO] Loading Models")

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)

with open(encodings_path, 'rb') as f:
	encodings = pickle.load(f)
with open(model_path, 'rb') as f:
	clf = pickle.load(f)

#VideoStream
print("[INFO] Initializing Video")
vs = VideoStream(src=0).start()

cv2.namedWindow("face", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("face", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

#----------------------------------------------------< Functions >--------------------------------------------------

def faceTracking():
	face_locations = face_recognition.face_locations(frame)
	
	for (top, right, bottom, left), face_encoding in zip(face_locations, encodings):
		cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

def alignFace():
	dets = detector(frame, 1)

	num_faces = len(dets)

	if num_faces == 0:
		print("Sorry, there were no faces found")
		
		return False
		
	else:
		faces = dlib.full_object_detections()
		
		for detection in dets:
			faces.append(sp(frame, detection))

		images = dlib.get_face_chips(frame, faces, size=320)

		image = dlib.get_face_chip(frame, faces[0])
		
		cv2.imwrite(output_image, image)
		
		return True
		
def identify():
	global name
	
	img = face_recognition.load_image_file(output_image)
	
	try:
		img_enc = face_recognition.face_encodings(img)[0]
		
		known = False
		distances = []

		if model == "SVM":

			face_distances = face_recognition.face_distance(encodings, img_enc)

			for i, face_distance in enumerate(face_distances):
				face_distance = int((1 - face_distance) * 100)
				distances.append(face_distance)
				if face_distance > distanceThreshold:
					known = True
				
				#print(f"The face distance is {face_distance}% from known image #{i}")

			if known:
				name = clf.predict([img_enc])[0]

		elif model == "KNN":
			
			closest_distances = clf.kneighbors([img_enc], n_neighbors=1)
			known = closest_distances[0][0][0] <= distanceThreshold / 100
			
			if known:
				name = clf.predict([img_enc])[0]
			
		print(f"predicted: {name}")
		
	except IndexError as error:
		print("couldn't encode face")
	
#----------------------------------------------------< Main Code >--------------------------------------------------

while True:
	
	frame = vs.read()
	
	frame = cv2.flip(frame,1)
	
	#frame = cv2.resize(frame, (0, 0), fx = 0.5, fy = 0.5, interpolation=cv2.INTER_NEAREST)
	
	savedImage = frame.copy()
	
	gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
	
	currentTime = time.time()
	fps = 1/(currentTime-previousTime)
	fps=int(fps)
	previousTime = currentTime
	cv2.putText(frame, f"fps: {fps}", (30, 40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 2)	
	cv2.putText(frame, f"predicted: {name}", (30, 75), cv2.FONT_HERSHEY_COMPLEX, 0.8, (50, 75, 50), 2)
	key = cv2.waitKey(1) & 0xFF

	faceTracking()

	if key == ord("s"):
		name = "Unknown"
		
		if alignFace():
			identify()

	if key == ord("q"):
		break
		
	cv2.imshow("face", frame)
	
cv2.destroyAllWindows()
vs.stop()
exit()
