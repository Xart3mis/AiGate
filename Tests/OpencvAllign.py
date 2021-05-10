from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

prototxt = "/home/bighero/AiGate/Models/deploy.prototxt.txt"
model = "/home/bighero/AiGate/Models/res10_300x300_ssd_iter_140000.caffemodel"
landmarksModelPath = "/home/bighero/AiGate/Models/shape_predictor_68_face_landmarks.dat"
eyeCascadePath = "/home/bighero/AiGate/Models/haarcascade_eye.xml"
confidenceThresh = 0.735

print("[INFO] loading models...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

print("[INFO] loading shape predictor...")
predictor = dlib.shape_predictor(landmarksModelPath)

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)


def getEyes(imgGray, face):
    landmarks = predictor(image=imgGray, box=face)

    upper1x = int((landmarks.part(38).x+landmarks.part(39).x)/2)
    upper2x = int((landmarks.part(42).x+landmarks.part(41).x)/2)

    upper1y = int((landmarks.part(38).y+landmarks.part(39).y)/2)
    upper2y = int((landmarks.part(42).y+landmarks.part(41).y)/2)

    eyes = {"left": (int((upper1x+upper2x)/2), int((upper1y+upper2y)/2))}

    return eyes


while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    img = frame.copy()

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence < confidenceThresh:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        x, y, w, h = startX, startY, (endX - startX), (endY - startY)

        img = frame[int(y):int(y+h), int(x):int(x+w)]
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        drect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
        eyes = getEyes(frame_gray, drect)

        img = cv2.circle(frame, eyes["left"], 3, (255, 255, 50), -1)
        print(eyes["left"])

    cv2.imshow("Face", img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
