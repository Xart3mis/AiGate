import os
import cv2
import dlib
from imutils.video import VideoStream

trainDir = "/home/bighero/AiGate/Tests/face_rec/train_dir/"
predictor_path =  "/home/bighero/AiGate/Models/shape_predictor_68_face_landmarks.dat"

personName = input("enter person name: ").lower()

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)

print("[INFO] Initializing Video")
vs = VideoStream(src=0).start()

def alignFace(filename):
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
		
		cv2.imwrite(trainDir + personName + "/" + filename +".jpg", image)
		
		return True

try:
	os.mkdir(trainDir + personName)
	
	print("pls look at the camera, press S to take pic")
	
	i = 1
	while i <= 3:
		frame = vs.read()
		frame = cv2.flip(frame,1)
	
		cv2.imshow("face", frame)
	
		if cv2.waitKey(1) == ord("s"):
			alignFace(str(i))
			i += 1		
			
		if cv2.waitKey(1) == ord("q"):
			break

except FileExistsError:
	pass

cv2.destroyAllWindows()
vs.stop()
exit()
