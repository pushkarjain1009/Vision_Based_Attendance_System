from Data_Collecting.get_faces_from_camera import TrainingDataCollector
from Face_Embeddings.face_embedding import extract_face
from Training.training import trainKerasModelForFaceRecognition
#from Prediction.prediction import detectFace
import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

name = input("Enter Your Name: ")

collect_image = TrainingDataCollector()
collect_image.collectImagesFromCamera('Dataset/Train/'+name)

out_path = extract_face('Dataset/Train/'+name)

trainKerasModelForFaceRecognition(out_path)

#detectFace()