import uuid
import cv2
import tensorflow as tf
import os
import numpy as np

def data_aug(img):

    # add some random characteristic change to pic to augment the dataset
    data = []
    for i in range(9):
        img = tf.image.stateless_random_brightness(img, max_delta=0.02, seed=(1,2))
        img = tf.image.stateless_random_contrast(img, lower=0.6, upper=1, seed=(1,3))
        # img = tf.image.stateless_random_crop(img, size=(20,20,3), seed=(1,2))
        img = tf.image.stateless_random_flip_left_right(img, seed=(np.random.randint(100),np.random.randint(100)))
        img = tf.image.stateless_random_jpeg_quality(img, min_jpeg_quality=90, max_jpeg_quality=100, seed=(np.random.randint(100),np.random.randint(100)))
        img = tf.image.stateless_random_saturation(img, lower=0.9,upper=1, seed=(np.random.randint(100),np.random.randint(100)))
            
        data.append(img)
    
    return data

def aug_ANC_and_POS_dataset():
    # Setup paths
    POS_PATH = os.path.join('data', 'positive')
    ANC_PATH = os.path.join('data', 'anchor')

    # ANC Dataset augmentation
    for file_name in os.listdir(os.path.join(ANC_PATH)):
        img_path = os.path.join(ANC_PATH, file_name)
        img = cv2.imread(img_path)
        augmented_images = data_aug(img) 
        
        for image in augmented_images:
            cv2.imwrite(os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1())), image.numpy())
    
    # POS Dataset augmentation
    for file_name in os.listdir(os.path.join(POS_PATH)):
        img_path = os.path.join(POS_PATH, file_name)
        img = cv2.imread(img_path)
        augmented_images = data_aug(img) 
        
        for image in augmented_images:
            cv2.imwrite(os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1())), image.numpy())

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu,True)
    aug_ANC_and_POS_dataset()
