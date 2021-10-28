import cv2
import os
import numpy as np
from datetime import datetime
from mtcnn.mtcnn import MTCNN

def collectImagesFromCamera(path):
    # initialize video stream
    cap = cv2.VideoCapture(0)
    detector = MTCNN()

    # Setup some useful var
    faces = 0
    frames = 0
    max_faces = 50
    landmarks = 0
    if not (os.path.exists(path)):
        os.makedirs(path)

    while faces < max_faces:
        ret, frame = cap.read()
        frames += 1

        dtString = str(datetime.now().microsecond)
        bboxes = detector.detect_faces(frame)
        if len(bboxes) != 0:
            # Get only the biggest face
            max_area = 0
            for bboxe in bboxes:
                bbox = bboxe["box"]
                bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                keypoints = bboxe["keypoints"]

                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if area > max_area:
                    max_bbox = bbox
                    landmarks = keypoints
                    max_area = area
            max_bbox = max_bbox[0:4]
            # get each of 3 frames
            if frames % 3 == 0:
                # convert to face_preprocess.preprocess input
                landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][
                    landmarks["mouth_left"][0], landmarks["mouth_right"][0],
                    landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][
                        landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
                nimg = landmarks.reshape((2, 5)).T
                cv2.imwrite(path, "{}.jpg".format(dtString)), nimg)
                cv2.rectangle(frame, (max_bbox[0], max_bbox[1]), (max_bbox[2], max_bbox[3]), (255, 0, 0), 2)

                print("[INFO] {} Image Captured".format(faces + 1))
                faces += 1

        cv2.imshow("Face detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
