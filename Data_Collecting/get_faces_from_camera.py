import numpy as np
import cv2
import os
from datetime import datetime

def collectImagesFromCamera(path):
    # initialize video stream
    cap = cv2.VideoCapture(0)

    # Setup some useful var
    faces = 0
    frames = 0
    max_faces = 50

    if not (os.path.exists(path)):
        os.makedirs(path)

    while faces < max_faces:
        ret, frame = cap.read()
        frames += 1

        dtString = str(datetime.now().microsecond)

        cv2.imwrite(os.path.join(path, "{}.jpg".format(dtString)), faces)
        print("[INFO] {} Image Captured".format(faces + 1))
        faces += 1
        cv2.imshow("Face detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
