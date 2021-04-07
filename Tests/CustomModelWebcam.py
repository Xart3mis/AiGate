from imutils.video import VideoStream
import face_recognition
import multitasking
import imutils
import pickle
import dlib
import time
import cv2

encodingsPath = "D:\Documents\GitHub\AiGate\Models\goodboi.pickle"

print("[INFO] loading encodings...")
data = pickle.loads(open(encodingsPath, "rb").read())

boxes = []

detector = dlib.get_frontal_face_detector()

print("[INFO] loading shape predictor...")
predictor = dlib.shape_predictor(
    "D:\Documents\GitHub\AiGate\Models\shape_predictor_68_face_landmarks.dat")

previousTime = 0

frameSkip = True
process = True

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

@multitasking.task
def getFaceLandmarks(_face):
    landmarks = predictor(image=gray, box=_face)
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
    x *= r
    y *= r
    cv2.circle(img=frame, center=(int(x), int(y)),
               radius=1, color=(100, 255, 6), thickness=-3)
    return landmarks

def getEyeCenter(arg):
	landmarks = arg
	x = landmarks.part(27).x
	y = landmarks.part(27).y
	x*=r
	y*=r
	eyeCenter = [int(x),int(y)]
	cv2.circle(img=frame, center=(int(x), int(y)), radius=4, color=(0, 0, 255), thickness=-6)
	return eyeCenter

def drawFaces():
    cv2.rectangle(frame, (left, top), (right, bottom),
                  (0, 255, 0), 2)
    y = top - 15 if top - 15 > 15 else top + 15

    cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 0), 2)


@multitasking.task
def getFaces():
    dets = detector(gray)
    boxes = []

    for det in dets:
        getFaceLandmarks(det)
        awooga = (det.top(), det.right(), det.bottom(), det.left())
        boxes.append(awooga)
    print(boxes)


while True:
    frame = vs.read()

    currentTime = time.time()
    fps = 1//(currentTime-previousTime)
    previousTime = currentTime

    cv2.putText(frame, f"fps: {fps}", (30, 40),
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 50), 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(frame, width=526)
    r = frame.shape[1] / float(rgb.shape[1])

    gray = cv2.cvtColor(src=rgb, code=cv2.COLOR_BGR2GRAY)

    getFaces()

    if process:
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []

        for encoding in encodings:

            matches = face_recognition.compare_faces(data["encodings"],
                                                     encoding)
            name = "Unknown"

            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)

            names.append(name)
        if frameSkip:
            process = not process

    for ((top, right, bottom, left), name) in zip(boxes, names):
        top = int(top * r)
        right = int(right * r)
        bottom = int(bottom * r)
        left = int(left * r)

        drawFaces()

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
