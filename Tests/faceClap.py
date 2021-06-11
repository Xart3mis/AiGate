from imutils.video import VideoStream
from sklearn import svm
import face_recognition
import numpy as np
import time
import cv2
import dlib
import os

#train_directory = "/home/bighero/AiGate/Tests/face_rec/train_dir"
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("/home/bighero/AiGate/Models/shape_predictor_68_face_landmarks.dat")

previousTime = 0

encodings = []
names = []

def trainModel():
	# Training directory
	train_dir = os.listdir('/train_dir/')

	# Loop through each person in the training directory
	for person in train_dir:
		pix = os.listdir("/train_dir/" + person)

		# Loop through each training image for the current person
		for person_img in pix:
		    # Get the face encodings for the face in each image file
		    face = face_recognition.load_image_file("/train_dir/" + person + "/" + person_img)
		    face_bounding_boxes = face_recognition.face_locations(face)

		    #If training image contains exactly one face
		    if len(face_bounding_boxes) == 1:
		        face_enc = face_recognition.face_encodings(face)[0]
		        # Add face encoding for current image with corresponding label (name) to the training data
		        encodings.append(face_enc)
		        names.append(person)
		    else:
		        print(person + "/" + person_img + " was skipped and can't be used for training")

def alignFace():
	dets = detector(frame, 1)

	num_faces = len(dets)

	if num_faces == 0:
		print("Sorry, there were no faces found")

	faces = dlib.full_object_detections()
	
	for detection in dets:
		faces.append(sp(frame, detection))

	images = dlib.get_face_chips(frame, faces, size=320)

	image = dlib.get_face_chip(frame, faces[0])
	
	cv2.imwrite("/home/bighero/AiGate/output2.jpg", image)

#trainModel()

vs = VideoStream(src=0).start()
time.sleep(2.0)

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

	key = cv2.waitKey(1) & 0xFF

	if key == ord("s"):
		try:
			alignFace()
		except:
			pass

	if key == ord("q"):
		break
		
	cv2.imshow("face", frame)
	
cv2.destroyAllWindows()
vs.stop()
exit()
