import os
import cv2
import time
import joblib
from FpsUp import *
import face_recognition
from functools import cache

cap = VideoCaptureThreading(0)
cap.start()

clf = joblib.load('model.pkl')

previousTime = 0


@cache
def getName():
    for i in range(len(face_locations)):
        frame_enc = face_recognition.face_encodings(small_frame)[i]
        name = clf.predict([frame_enc])
        print(name)


while True:
    currentTime = time.time()
    fps = 1//(currentTime-previousTime)
    previousTime = currentTime

    ret, frame = cap.read()

    cv2.putText(frame, f"fps: {fps}", (30, 40),
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 50), 2)

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    face_locations = face_recognition.face_locations(small_frame)

    top = 0
    right = 0
    bottom = 0
    left = 0

    name = "unknown"

    try:
        getName()
        top, right, bottom, left = face_locations[0]
    except IndexError:
        print("no faces")

    if top * right * bottom * left != 0:
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)

        cv2.rectangle(frame, (left, top-30), (right, top), (0, 255, 0), -1)
        cv2.putText(frame, name, (left, top),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.namedWindow("face", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        'face', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.imshow('face', frame)

    if cv2.waitKey(1) & 0XFF == ord("q"):
        cap.stop()
        cv2.destroyAllWindows()
        break
