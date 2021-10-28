from Data_Collecting.get_faces_from_camera import collectImagesFromCamera
from Face_Embeddings.face_embedding import extract_face
from Training.training import trainKerasModelForFaceRecognition
#from Prediction.prediction import detectFace

name = input("Enter Your Name: ")

collectImagesFromCamera('Dataset/Train/'+name)
out_path = extract_face('Dataset/Train/'+name)
trainKerasModelForFaceRecognition(out_path)
#detectFace()