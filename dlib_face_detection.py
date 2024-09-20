import dlib
import numpy as np
import os
import cv2
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("Other_dependencies/DLIB_landmark_det/shape_predictor_68_face_landmarks.dat")

Global_images_path='data/VideoDataset/extracted/CelebVHQ'
Json_save_path='data/VideoDataset/extracted/CelebVHQ_faces'


image_folder_list = os.listdir(Global_images_path)

for image_folder in image_folder_list:
    images=os.listdir(Global_images_path+'/'+image_folder)
    for image in images:
        image_path=Global_images_path+'/'+image_folder+'/'+image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:
            x1 = face.left()


