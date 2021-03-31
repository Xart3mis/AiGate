import cv2
import numpy as np
import dlib
import joblib
import face_recognition
from functools import cache

clf = joblib.load('model.pkl')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

@cache
def getName():
    for i in range(len(faces)):
        frame_enc = face_recognition.face_encodings(frame)[i]
        name = clf.predict([frame_enc])
        print(name[0])

@cache
def getFaceLandmarks():
    landmarks = predictor(image=gray, box=face)
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
    return x, y


while True:
    ret, frame = cap.read()

    top = 0
    right = 0
    bottom = 0
    left = 0

    name = "Unknown"

    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    #Detects all faces
    for face in faces:
        left = face.left()
        top = face.top() 
        right = face.right()
        bottom = face.bottom()

        #Face identification
        try:
            getName()
        except IndexError:
            print("no faces")

        if top * right * bottom * left != 0:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)

            cv2.rectangle(frame, (left, top-30), (right, top), (0, 255, 0), -1)
            cv2.putText(frame, name, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        #Detect facial landmarks    
        x, y = getFaceLandmarks()
        cv2.circle(img=frame, center=(x, y), radius=1, color=(0, 255, 0), thickness=-1)
    cv2.namedWindow("face", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        'face', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.imshow('face', frame)

    if cv2.waitKey(1) & 0XFF == ord("q"):
        cv2.destroyAllWindows()
        cap.release()