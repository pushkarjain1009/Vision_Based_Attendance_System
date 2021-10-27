from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
# from keras.models import load_model
#import matplotlib.pyplot as plt
# from softmax import SoftMax
import numpy as np
import pickle

from Training.model import Model

import logging


def get_logger():
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s \t%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def trainKerasModelForFaceRecognition(embeddings):

    data = pickle.loads(open(embeddings, "rb").read())

    # Encode the labels
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])
    num_classes = len(np.unique(labels))
    labels = labels.reshape(-1, 1)
    labels = labels.reshape(-1, 1)
    one_hot_encoder = OneHotEncoder(categorical_features=[0])
    labels = one_hot_encoder.fit_transform(labels).toarray()
    logger = get_logger()

    embeddings = np.array(data["embeddings"])

    # Initialize Softmax training model arguments
    BATCH_SIZE = 8
    EPOCHS = 5
    input_shape = embeddings.shape[1]

    # Build sofmax classifier
    model = Model(input_shape=(input_shape,), num_classes=num_classes)

    # Create KFold
    cv = KFold(n_splits = 5, random_state = 42, shuffle=True)
    history = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}

    # Train
    for train_idx, valid_idx in cv.split(embeddings):
        X_train, X_val, y_train, y_val = embeddings[train_idx], embeddings[valid_idx], labels[train_idx], labels[valid_idx]
        his = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(X_val, y_val))
        print(his.history['acc'])

        history['acc'] += his.history['acc']
        history['val_acc'] += his.history['val_acc']
        history['loss'] += his.history['loss']
        history['val_loss'] += his.history['val_loss']

        logger.info(his.history['acc'])

    # write the face recognition model to output
    model.save('../face_embedding_models/model')
    f = open("../face_embedding_models/le", "wb")
    f.write(pickle.dumps(le))
    f.close()
