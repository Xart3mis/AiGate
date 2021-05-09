from deepface import DeepFace
from imutils.video import VideoStream
import cv2
import time

imag1 = "D:\\Documents\\dataset\\train\seif zayed\\20210401_002105.jpg"
imag2 = "D:\\Documents\\dataset\\train\\seif zayed\\20210401_002113.jpg"

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

previousTime = 0

while True:
    frame = vs.read()

    currentTime = time.time()
    fps = 1//(currentTime-previousTime)
    previousTime = currentTime

    cv2.putText(frame, f"fps: {fps}", (30, 40),
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 50), 2)

    tic = time.time()
    try:
        detected_face = DeepFace.detectFace(
            frame, detector_backend="ssd")
    except ValueError:
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_RGB2BGR)
        print("nu face")

    toc = time.time()

    print(toc-tic)

    detected_face = cv2.cvtColor(detected_face, cv2.COLOR_RGB2BGR)
    cv2.imshow('siodoai', detected_face)

    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
exit(0)
