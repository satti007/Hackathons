import os
import cv2
import glob
import random
import numpy as np

#Size of images
IMAGE_MIN = 512

def transform_img(img):
    
    #Histogram Equalization
    # img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    # img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    # img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    
    #Image Resizing
    img_height, img_width, channels = img.shape
    if(img_height<img_width):
        ratio = IMAGE_MIN*1.0/img_height
        img_height = IMAGE_MIN
        img_width = int(img_width*ratio)
    else:
        ratio = IMAGE_MIN*1.0/img_width
        img_width = IMAGE_MIN
        img_height = int(img_height*ratio)

    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)
    
    return img


# os.system('rm -rf  HE_preprocess_data')
# os.system('rm -rf  AHE_preprocess_data')
# os.system('mkdir HE_preprocess_data')
# os.system('mkdir AHE_preprocess_data')

# train_data = [img for img in glob.glob("train_data/*jpg")]

# #Shuffle train_data
# random.shuffle(train_data)
# print 'Creating train_lmdb'

# for in_idx, img_path in enumerate(train_data):
#     img = cv2.imread(img_path, cv2.IMREAD_COLOR)
#     img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
#     index = int(img_path.split('/')[1].split('.')[0])
#     cv2.imwrite( "HE_preprocess_data/"+str(index)+".jpg", img)
#     # img = AHE(img)
#     # cv2.imwrite( "AHE_preprocess_data/"+str(index)+".jpg", img)
#     print "Done ",in_idx



# import numpy as np
# import cv2
# img = cv2.imread('train_data/1.jpg',0)
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# cl1 = clahe.apply(img)
# cv2.imwrite('clahe_2.jpg',cl1)