import os
import cv2
import joblib
import face_recognition
from sklearn.svm import SVC

TRAIN_DIR = 'dataset/train/'
persons = os.listdir(os.getcwd() + TRAIN_DIR)

#cv2.imshow('face', cv2.imread('dataset/train/face_1/adhast.1.jpg'))

cv2.waitKey()
cv2.destroyAllWindows()

encodings = []
names = []


for person in persons:

    pix = os.listdir(TRAIN_DIR + person)

    for person_img in pix:

        face = face_recognition.load_image_file(
            TRAIN_DIR + person + "/" + person_img)
        face_bounding_boxes = face_recognition.face_locations(face)

        if len(face_bounding_boxes) == 1:
            face_enc = face_recognition.face_encodings(face)[0]
            encodings.append(face_enc)
            names.append(person)
        else:
            print(person + "/" + person_img +
                  " was skipped and can't be used for training")

clf = SVC()

clf.fit(encodings, names)

joblib.dump(clf, 'model.pkl')
