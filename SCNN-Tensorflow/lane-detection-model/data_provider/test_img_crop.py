import tensorflow as tf
from matplotlib import pyplot as plt
import cv2
import numpy as np
# from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard


img_raw = tf.read_file('images/04320.jpg')
img_decoded = tf.image.decode_jpeg(img_raw, channels=3)
img_decoded_ = tf.expand_dims(img_decoded, 0)
# img_crop_resize = tf.image.crop_and_resize(img_decoded_, [[0,0.2,1,0.75]], crop_size = [400, 600] , box_ind = [0])
img_resized = tf.image.resize_images(img_decoded, [288, 800], method=tf.image.ResizeMethod.BICUBIC)
# TODO: Check problems with dimensions - compare with lanenet_data_processor.py
with tf.Session() as sess:
    imgr, imgd = sess.run([img_raw, img_decoded])
    # imgd_, imgcr= sess.run([img_decoded_, img_crop_resize])
    img_fin = sess.run([img_resized])
    # img_ = tf.squeeze(img_fin,0)
    # img_ = img[0][:][:][:]
    print(np.shape(img_fin))
    # print(np.shape(img_fin))
    # print(np.shape(img_))

    # plt.imshow(img_/255)
    # plt.show()
    # plt.imshow(imgd)
    # plt.show()