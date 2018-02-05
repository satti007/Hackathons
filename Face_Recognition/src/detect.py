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
from preprocess import *

detector = dlib.get_frontal_face_detector()
predictor_model = "models/shape_predictor_68_face_landmarks.dat"
face_pose_predictor = dlib.shape_predictor(predictor_model)
face_aligner = Align(predictor_model)

def rotate_image(image, angle):
    if angle == 0: return image
    height, width = image.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((width/2, height/2), angle, 0.9)
    result = cv2.warpAffine(image, rot_mat, (width, height), flags=cv2.INTER_LINEAR)
    return result

def rotate_point(pos, img, angle):
    if angle == 0: return pos
    x = pos[0] - img.shape[1]*0.4
    y = pos[1] - img.shape[0]*0.4
    newx = x*cos(radians(angle)) + y*sin(radians(angle)) + img.shape[1]*0.4
    newy = -x*sin(radians(angle)) + y*cos(radians(angle)) + img.shape[0]*0.4
    return int(newx), int(newy)

def face_detect(file) :

    # file = '../data/lynk/100.jpg'
    # img = cv2.imread(file,cv2.IMREAD_COLOR)
    img = transform_img(cv2.imread(file,cv2.IMREAD_COLOR))

    maxscore, maxangle = (-1,-1)
    eps = 0.05 
    detected_rect = None

    for angle in range(-90, 90, 15):
        rimg = rotate_image(img, angle)
        dets, scores, idx = detector.run(cv2.cvtColor(rimg, cv2.COLOR_BGR2GRAY), 1, -1)
        if len(scores) : 
            face_id = np.argmax(scores)
            score = scores[face_id]
            d = dets[face_id]

            # print 'score %g' % score

            if(score > maxscore + eps):
                maxangle = angle
                maxscore = score
                detected_rect = d
                # print 'angle %g' %angle

    rimg = rotate_image(img, maxangle)
    aligned_face = face_aligner.align(227, rimg, detected_rect, landmarkIndices=Align.OUTER_EYES_AND_NOSE)

    # print 'Max Score %g' % maxscore
    # cv2.imwrite("aligned_face.jpg", aligned_face)

    return aligned_face, maxscore

    # win = dlib.image_window()
    # win.clear_overlay()
    # win.set_image(rimg)
    # win.add_overlay(detected_rect)
    # dlib.hit_enter_to_continue()
