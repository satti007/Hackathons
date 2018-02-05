"""
@module : Detect 
@desc   : detect faces in the input image
"""
import sys

from PIL import Image
import dlib
import cv2
import numpy as np
from Align import *

class Detect() :

    def __init__(self) :
        self.detector = dlib.get_frontal_face_detector()
        self.predictor_model = "shape_predictor_68_face_landmarks.dat"
        self.face_pose_predictor = dlib.shape_predictor(self.predictor_model)
        self.face_aligner = Align(self.predictor_model)
        pass

    def run(self, X, is_file=True) :
        if is_file :
            print ("Processing : {}".format(X))
            img = cv2.imread(X)

            dets, scores, idx = self.detector.run(img, 1, -1)
            print scores
            # Max Score Face
            face_id = np.argmax(scores)

            d = dets[face_id]

            pose_landmarks = self.face_pose_predictor(img, d)

            aligned_face = self.face_aligner.align(227, img, d, landmarkIndices=Align.OUTER_EYES_AND_NOSE)

            cv2.imwrite("aligned_face.jpg", aligned_face)
            win = dlib.image_window()
            win.clear_overlay()
            win.set_image(img)
            win.add_overlay(d)
            dlib.hit_enter_to_continue()
