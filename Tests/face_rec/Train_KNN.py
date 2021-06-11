import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

trainDir =     "/home/bighero/AiGate/Tests/face_rec/train_dir/"
modelSavePath =     "/home/bighero/AiGate/Tests/face_rec/model/KNN_Classifier.pickle"
encodingsSavePath = "/home/bighero/AiGate/Tests/face_rec/model/Encodings.pickle"

encodings = []
classes = []


for classDir in os.listdir(trainDir):
    if not os.path.isdir(os.path.join(trainDir, classDir)):
        continue

    for imgPath in image_files_in_folder(os.path.join(trainDir, classDir)):
        image = face_recognition.load_image_file(imgPath)
        faceBoundingBoxes = face_recognition.face_locations(image)

        if len(faceBoundingBoxes) != 1:
            print("Image {} not suitable for training: {}".format(imgPath, "Didn't find a face" if len(faceBoundingBoxes) < 1 else "Found more than one face"))
        else:
            encodings.append(face_recognition.face_encodings(image, known_face_locations=faceBoundingBoxes)[0])
            classes.append(classDir)

nNeighbours = int(round(math.sqrt(len(encodings))))
print("Chose n_neighbors automatically:", nNeighbours)


knnClf = neighbors.KNeighborsClassifier(n_neighbors=nNeighbours, algorithm="auto", weights='distance', n_jobs=-1)
knnClf.fit(encodings, classes)


with open(modelSavePath, 'wb') as f:
    pickle.dump(knnClf, f)
    
with open(encodingsSavePath, 'wb') as f:
    pickle.dump(encodings, f)
