import cv2
import dlib
# Load the detector
detector = dlib.get_frontal_face_detector()
# Load the predictor
predictor = dlib.shape_predictor("AiGate/Models/shape_predictor_68_face_landmarks.dat")
# read the image
img = cv2.imread("test image.png")
# Convert image into grayscale
gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
# Use detector to find landmarks
faces = detector(gray)
for face in faces:
	landmarks = predictor(image=gray, box=face)
	for n in range(0, 68):
		x = landmarks.part(n).x
		y = landmarks.part(n).y
		cv2.circle(img=frame, center=(int(x), int(y)), radius=1, color=(100, 255, 6), thickness=-3)
# show the image
cv2.imshow("Face", img)
# Wait for a key press to exit
cv2.waitKey(0)
# Close all windows
cv2.destroyAllWindows()
