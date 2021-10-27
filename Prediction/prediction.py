from mtcnn import MTCNN
from keras.models import load_model
import numpy as np
import pickle
import cv2
import dlib

def findCosineDistance(vector1, vector2):
    vec1 = vector1.flatten()
    vec2 = vector2.flatten()

    a = np.dot(vec1.T, vec2)
    b = np.dot(vec1.T, vec1)
    c = np.dot(vec2.T, vec2)
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def CosineSimilarity(test_vec, source_vecs):

    cos_dist = 0
    for source_vec in source_vecs:
        cos_dist += findCosineDistance(test_vec, source_vec)
    return cos_dist / len(source_vecs)


def detectFace():
    # Initialize some useful arguments
    cosine_threshold = 0.8
    proba_threshold = 0.85
    comparing_num = 5
    trackers = []
    texts = []
    frames = 0
    detector = MTCNN()
    embeddings = "../face_embedding_models/embeddings.pickle"
    le = "../face_embedding_models/le.pickle"

    data = pickle.loads(open(embeddings, "rb").read())
    le = pickle.loads(open(le, "rb").read())

    embeddings = np.array(data['embeddings'])
    labels = le.fit_transform(data['names'])

    # Start streaming and recording
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    save_width = 800
    save_height = int(800 / frame_width * frame_height)

    while True:
        ret, frame = cap.read()
        frames += 1
        frame = cv2.resize(frame, (save_width, save_height))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if frames % 3 == 0:
            tracker = []
            texts = []

            bboxes = detector.detect_faces(frame)

            if len(bboxes) != 0:

                for bboxe in bboxes:
                    bbox = bboxe['box']
                    bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                    landmarks = bboxe['keypoints']
                    landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0],
                                          landmarks["mouth_left"][0], landmarks["mouth_right"][0],
                                          landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1],
                                          landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
                    landmarks = landmarks.reshape((2, 5)).T
                    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
                    nimg = np.transpose(nimg, (2, 0, 1))

                    text = "Unknown"
                    model = load_model("../face_embedding_models/model.h2")
                    # Predict class
                    preds = model.predict(nimg)
                    preds = preds.flatten()
                    # Get the highest accuracy embedded vector
                    j = np.argmax(preds)
                    proba = preds[j]
                    # Compare this vector to source class vectors to verify it is actual belong to this class
                    match_class_idx = (labels == j)
                    match_class_idx = np.where(match_class_idx)[0]
                    selected_idx = np.random.choice(match_class_idx, comparing_num)
                    compare_embeddings = embeddings[selected_idx]
                    # Calculate cosine similarity
                    cos_similarity = CosineSimilarity(nimg, compare_embeddings)
                    if cos_similarity < cosine_threshold and proba > proba_threshold:
                        name = le.classes_[j]
                        text = "{}".format(name)
                        print("Recognized: {} <{:.2f}>".format(name, proba * 100))
                    # Start tracking
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(bbox[0], bbox[1], bbox[2], bbox[3])
                    tracker.start_track(rgb, rect)
                    trackers.append(tracker)
                    texts.append(text)

                    y = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[1] + 10
                    cv2.putText(frame, text, (bbox[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 1)
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (179, 0, 149), 4)
        else:
            for tracker, text in zip(trackers, texts):
                pos = tracker.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                cv2.rectangle(frame, (startX, startY), (endX, endY), (179, 0, 149), 4)
                cv2.putText(frame, text, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 1)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
