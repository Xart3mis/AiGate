import os
import cv2
import joblib
import face_recognition

cap = cv2.VideoCapture(0)

clf = joblib.load('model.pkl')


while True:
    ret, frame = cap.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.2, fy=0.2)
    face_locations = face_recognition.face_locations(small_frame)
    top = 0
    right = 0
    bottom = 0
    name = "unknown"
    left = 0
    try:
        for i in range(len(face_locations)):
            frame_enc = face_recognition.face_encodings(small_frame)[i]
            name = clf.predict([frame_enc])
            print(name)

        top, right, bottom, left = face_locations[0]
    except IndexError:
        print("no faces")

    top *= 5
    right *= 5
    bottom *= 5
    left *= 5
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)

    cv2.rectangle(frame, (left, top-30), (right, top), (0, 255, 0), -1)
    cv2.putText(frame, name[0], (left, top),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.namedWindow("face", cv2.WINDOW_NORMAL)

    cv2.imshow('face', frame)

    if cv2.waitKey(1) & 0XFF == ord("q"):
        cv2.destroyAllWindows()
        cap.release()
