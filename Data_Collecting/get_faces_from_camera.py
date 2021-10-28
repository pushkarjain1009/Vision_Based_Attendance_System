'''import numpy as np
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
'''

import cv2
import os
from datetime import datetime

# Load functions


def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    face_classifier = cv2.CascadeClassifier(
        'haarcascade_frontalface_default.xml')
    faces = face_classifier.detectMultiScale(img, 1.3, 5)

    if len(faces) == 0:
        return None

    # Crop all faces found
    for (x, y, w, h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face


def collectImagesFromCamera(path):
    # initialize video stream
    cap = cv2.VideoCapture(0)

    # Load HAAR face classifier


    # Setup some useful var
    count = 0
    frames = 0
    max_faces = 50

    if not (os.path.exists(path)):
        os.makedirs(path)

    # Collect ____ samples of your face from webcam input
    while True:

        ret, frame = cap.read()
        if face_extractor(frame) is not None:
            count += 1
            face = cv2.resize(face_extractor(frame), (200, 200))

            # Save file in specified directory with unique name
            # Change between Validation and testion and also change name acc to face

            cv2.imwrite(os.path.join(path, "{}.jpg".format(count)), face)

            # Put count on images and display live count
            cv2.putText(face, str(count), (50, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', face)

        else:
            print("Face not found")
            pass

    # Change count acc. to no. of samples you need
        if cv2.waitKey(1) == 13 or count == 1:  # 13 is the Enter Key
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Collecting Samples Complete")