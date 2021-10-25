from PIL import Image
import numpy as np
from imutils import paths
from mtcnn.mtcnn import MTCNN
import pickle
import cv2
import os


# extraction of faces

def extract_face(path, embeddings, required_size=(160, 160)):

    print("[INFO] quantifying faces...")
    imagePaths = list(paths.list_images(path))

    # Initialize our lists of extracted facial embeddings and corresponding people names
    knownEmbeddings = []
    knownNames = []

    # Initialize the total number of faces processed
    total = 0

    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]

        # load the image
        image = cv2.imread(imagePath)

        nimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        nimg = np.transpose(nimg, (2, 0, 1))

        detector = MTCNN()
        results = detector.detect_faces(nimg)

        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = nimg[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_embedding = np.asarray(image)

        knownNames.append(name)
        knownEmbeddings.append(face_embedding)
        total += 1

    print(total, " faces embedded")

    # save to output
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    f = open(embeddings, "wb")
    f.write(pickle.dumps(data))
    f.close()

    return