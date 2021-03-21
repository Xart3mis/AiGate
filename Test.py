import os
import cv2
import joblib
import face_recognition

clf = joblib.load('model.pkl')
test_image = face_recognition.load_image_file('dataset/test/YoussefTest.jpg')

face_recognition.face_encodings(test_image)

face_locations = face_recognition.face_locations(test_image)

print(f"Number of faces detected: {len(face_locations)}")

for i in range(len(face_locations)):
    test_image_enc = face_recognition.face_encodings(test_image)[i]
    name = clf.predict([test_image_enc])
    print(name)

top, right, bottom, left = face_locations[0]

test_image = cv2.imread('dataset/test/YoussefTest.jpg')

cv2.rectangle(test_image, (left, top), (right, bottom), (0,255,0), 1)

cv2.rectangle(test_image, (left, top-30), (right, top), (0,255,0), -1)
cv2.putText(test_image, name[0], (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
cv2.namedWindow("face", cv2.WINDOW_NORMAL)
cv2.resizeWindow("face", 1280, 720)
cv2.imshow('face', test_image)

cv2.waitKey()
cv2.destroyAllWindows()

