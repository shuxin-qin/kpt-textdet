import tensorflow as tf
import numpy as np
import cv2
from ThinPlateSpline import ThinPlateSpline as stn

def tps_cv2(source, target, img):
    """
    使用cv2自带的tps处理
    """
    tps = cv2.createThinPlateSplineShapeTransformer()

    source_cv2 = source.reshape(1, -1, 2)
    target_cv2 = target.reshape(1, -1, 2)

    matches = list()
    for i in range(0, len(source_cv2[0])):
        matches.append(cv2.DMatch(i,i,0))

    tps.estimateTransformation(target_cv2, source_cv2, matches)
    new_img_cv2 = tps.warpImage(img)

    return new_img_cv2


img = cv2.imread("test.jpg")
print(img.shape)

size = list(img.shape)
print(size)


s = np.array([[[51,55], [90,32], [120,30], [154,36], [182,60], [72,72], [100,54], [120,50], [138,54], [162,72]],
               [[69,123], [90,145], [116,152], [142,147], [166,125], [49,136], [80,164], [116,174], [155,166], [183,142]]])


t = np.array([[[0,0], [50,0], [100,0], [150,0], [200,0], [0,50], [50,50], [100,50], [150,50], [200,50]]])


for i in range(s.shape[0]):
  
    source = s[i]
    target = t
    out = tps_cv2(source, target, img)
    print(out.shape)
    cv2.imwrite('out_'+str(i)+'.jpg', out[:50, :200, :])