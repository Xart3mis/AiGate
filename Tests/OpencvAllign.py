from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

prototxt = "D:\\Documents\\Python\\AiGate\\Models\\deploy.prototxt.txt"
model = "D:\\Documents\\Python\\AiGate\\Models\\res10_300x300_ssd_iter_140000.caffemodel"
landmarksModelPath = "D:\\Documents\\Python\\AiGate\\Models\\shape_predictor_68_face_landmarks_GTX.dat"
confidenceThresh = 0.55

print("[INFO] loading models...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

print("[INFO] loading shape predictor...")
predictor = dlib.shape_predictor(landmarksModelPath)

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)


def getEyes(imgGray, face):
    landmarks = predictor(image=imgGray, box=face)

    x1Left = int((landmarks.part(38).x + landmarks.part(39).x)/2)
    y1Left = int((landmarks.part(38).y + landmarks.part(39).y)/2)
    x2Left = int((landmarks.part(42).x + landmarks.part(41).x)/2)
    y2Left = int((landmarks.part(42).y + landmarks.part(41).y)/2)
    x3Left = int(landmarks.part(37).x)
    y3Left = int(landmarks.part(37).y)
    x4Left = int(landmarks.part(40).x)
    y4Left = int(landmarks.part(40).y)

    x1Right = int((landmarks.part(44).x + landmarks.part(45).x)/2)
    y1Right = int((landmarks.part(44).y + landmarks.part(45).y)/2)
    x2Right = int((landmarks.part(48).x + landmarks.part(47).x)/2)
    y2Right = int((landmarks.part(48).y + landmarks.part(47).y)/2)
    x3Right = int(landmarks.part(43).x)
    y3Right = int(landmarks.part(43).y)
    x4Right = int(landmarks.part(46).x)
    y4Right = int(landmarks.part(46).y)

    leftD = (x1Left-x2Left)*(y3Left-y4Left)-(y1Left-y2Left)*(x3Left-x4Left)
    rightD = (x1Right-x2Right)*(y3Right-y4Right) - \
        (y1Right-y2Right)*(x3Right-x4Right)

    PxLeft = int(((x1Left*y2Left-y1Left*x2Left) *
                  (x3Left-x4Left)-(x1Left-x2Left)*(x3Left*y4Left-y3Left*x4Left))/leftD)

    PyLeft = int(((x1Left*y2Left-y1Left*x2Left) *
                  (y3Left-y4Left)-(y1Left-y2Left)*(x3Left*y4Left-y3Left*x4Left))/leftD)

    PxRight = int(((x1Right*y2Right-y1Right*x2Right) *
                  (x3Right-x4Right)-(x1Right-x2Right)*(x3Right*y4Right-y3Right*x4Right))/rightD)

    PyRight = int(((x1Right*y2Right-y1Right*x2Right) *
                  (y3Right-y4Right)-(y1Right-y2Right)*(x3Right*y4Right-y3Right*x4Right))/rightD)

    eyes = {"left": (PxLeft, PyLeft), "right": (PxRight, PyRight)}

    for i in range(68):
        cv2.circle(frame, (landmarks.part(i).x, landmarks.part(i).y),
                   1, (255, 0, 200), -1)

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

        cv2.circle(frame, eyes["left"], 3, (255, 255, 50), -1)
        cv2.circle(frame, eyes["right"], 3, (255, 255, 50), -1)
        print(eyes["left"], eyes["right"])

    cv2.imshow("Face", img)
    cv2.imshow("Facwe", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
