import cv2
import numpy as np
import dlib
import joblib
import face_recognition
from functools import cache

cap = cv2.VideoCapture(0)

factor = 1

clf = joblib.load('model.pkl')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

@cache
def getName():
    for i in range(len(faces)):
        frame_enc = face_recognition.face_encodings(small_frame)[i]
        name = clf.predict([frame_enc])
        print(name[0])

def drawFaces():
    if top * right * bottom * left != 0:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)
        cv2.rectangle(frame, (left, top-30), (right, top), (0, 255, 0), -1)
        cv2.putText(frame, name, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

def getFaceLandmarks():
    landmarks = predictor(image=gray, box=face)
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        x*=factor
        y*=factor
        cv2.circle(img=frame, center=(x, y), radius=1, color=(0, 255, 0), thickness=-3)

while True:
    ret, frame = cap.read()
    small_frame = cv2.resize(frame, (0, 0), fx=(1/factor), fy=(1/factor))

    top = 0
    right = 0
    bottom = 0
    left = 0

    name = "Unknown"

    gray = cv2.cvtColor(src=small_frame, code=cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    #Detects all faces
    for face in faces:
        left = face.left()*factor
        top = face.top()*factor
        right = face.right()*factor
        bottom = face.bottom()*factor

        #Face identification
        try:
            getName()
        except IndexError:
            print("no faces")

        drawFaces()

        #Detect facial landmarks    
        getFaceLandmarks()

    cv2.namedWindow("face", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        'face', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.imshow('face', frame)

    if cv2.waitKey(1) & 0XFF == ord("q"):
        cv2.destroyAllWindows()
        cap.release()