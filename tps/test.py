import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
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

shape = [1]+size
print(shape)

# 69,123  49,136  90 145   80 164  116 152   116 174    142 147  155 166  166 125  183 142
# 51,55   72 72   90 32  100 54  120 30  120 50   154 36  138 54  182 60   162 72
#s1 = np.array([[55,51], [72,72], [32,90], [54,100], [30,120], [50,120], [36,154], [54,138], [60,182], [72, 162]], dtype=np.float32)
s1 = np.array([[[51,55], [90,32], [120,30], [154,36], [182,60], [72,72], [100,54], [120,50], [138,54], [162,72]],
               [[69,123], [90,145], [116,152], [142,147], [166,125], [49,136], [80,164], [116,174], [155,166], [183,142]]], dtype=np.float32)
#s1 = np.array([[51,55], [72,72], [182,60], [162,72]], dtype=np.float32)
s_ = s1 / [size[1],size[0]]*2-1

print(s_)

t1 = np.array([[[0,0], [50,0], [100,0], [150,0], [200,0], [0,50], [50,50], [100,50], [150,50], [200,50]],
               [[0,0], [50,0], [100,0], [150,0], [200,0], [0,50], [50,50], [100,50], [150,50], [200,50]]], dtype=np.float32)
#t1 = np.array([[0,0], [0,50], [200,0], [200, 50]], dtype=np.float32)
t_ = t1/[200, 50]*2-1

#t_ = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
print(t_)

s = tf.constant(s_.reshape([2, 10, 2]), dtype=tf.float32)
t = tf.constant(t_.reshape([2, 10, 2]), dtype=tf.float32)

img = img.reshape(shape)
img = np.concatenate((img, img))
t_img = tf.constant(img, dtype=tf.float32)

new_size = [50, 200, 3]
t_img = stn(t_img, t, s, new_size[:2])   # 函数中的 source 和dst是反的， 另外输入和输出的x y指图像的宽、高，与矩阵相反

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  out = sess.run(t_img)
  cv2.imwrite("out0.jpg", np.uint8(out[0].reshape(new_size)))
  cv2.imwrite("out1.jpg", np.uint8(out[1].reshape(new_size)))

