import os
import cv2
import joblib
import face_recognition
from functools import cache

cap = cv2.VideoCapture(0)

clf = joblib.load('model.pkl')


@cache
def getName():
    for i in range(len(face_locations)):
        frame_enc = face_recognition.face_encodings(small_frame)[i]
        name = clf.predict([frame_enc])
        print(name)


while True:
    ret, frame = cap.read()
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

    cv2.imshow('face', frame)

    if cv2.waitKey(1) & 0XFF == ord("q"):
        cv2.destroyAllWindows()
        cap.release()
