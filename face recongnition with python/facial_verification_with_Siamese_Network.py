import cv2
import os
import random
import numpy as np
import uuid

from matplotlib import pyplot as plt
from tensorflow.keras.models import  Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

# Setup paths
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

def gpu_growth():
    # Avoid OOM errors by setting GPU Memory Consumption Growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu,True)

class File():

    def create_folder_structure(self):
        
        # Make the directories
        os.makedirs(POS_PATH)
        os.makedirs(NEG_PATH)
        os.makedirs(ANC_PATH)

        # Move Images to the following repository data/negative    
        for directory in os.listdir('lfw'):
            for file in os.listdir(os.path.join('lfw', directory)):
                EX_PATH = os.path.join('lfw', directory, file)
                NEW_PATH = os.path.join(NEG_PATH, file)
                os.replace(EX_PATH, NEW_PATH)

    def Collect_Positive_and_Anchor_Classes(self):
        cap = cv2.VideoCapture(0)
        while cap.isOpened(): 
            ret, frame = cap.read()
        
            # Cut down frame to 250x250px
            frame = frame[120:120+250, 200:200+250, :]
            
            key_wait = cv2.waitKey(1)

            # Collect anchors 
            if  key_wait & 0XFF == ord('a'):
                # Create the unique file path 
                imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
                # Write out anchor image
                cv2.imwrite(imgname, frame)
            
            # Collect positives
            if key_wait & 0XFF == ord('p'):
                # Create the unique file path
                imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
                # Write out positive image
                cv2.imwrite(imgname, frame)

            # Show image back to screen
            cv2.imshow('WebCam', frame)

            # Breaking gracefully
            if key_wait & 0XFF == ord('q'):
                break    
        
        # Release the webcam
        cap.release()
        # Close the image show frame
        cv2.destroyAllWindows()

class Data_augmentation():

    def data_aug(self, img):

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

    def aug_ANC_and_POS_dataset(self):

        # ANC Dataset augmentation
        for file_name in os.listdir(os.path.join(ANC_PATH)):
            img_path = os.path.join(ANC_PATH, file_name)
            img = cv2.imread(img_path)
            augmented_images = self.data_aug(img) 
            
            for image in augmented_images:
                cv2.imwrite(os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1())), image.numpy())
        
        # POS Dataset augmentation
        for file_name in os.listdir(os.path.join(POS_PATH)):
            img_path = os.path.join(POS_PATH, file_name)
            img = cv2.imread(img_path)
            augmented_images = self.data_aug(img) 
            
            for image in augmented_images:
                cv2.imwrite(os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1())), image.numpy())

# get image directories
anchor = tf.data.Dataset.list_files(ANC_PATH+'/*.jpg').take(3000)
positive = tf.data.Dataset.list_files(POS_PATH+'/*.jpg').take(3000)
negative = tf.data.Dataset.list_files(NEG_PATH+'/*.jpg').take(3000)

class Dataset():

    def preprocess(self, file_path):

        # Read in image from file path
        byte_img = tf.io.read_file(file_path)
        
        # Load in the image
        img = tf.io.decode_jpeg(byte_img)

        # preprocessing resize image to 105x105x3
        img = tf.image.resize(img,(105,105))

        # scale image to between 0 and 1
        img = img / 255.0
        
        return img

    def create_labelled_dataset():
        # (anchor, positive) => 1
        # (anchor, negative) => 0

        val_positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
        val_negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
        data = val_positives.concatenate(val_negatives)

        return data

    def preprocess_twin(self, img_input, img_validation, label):
        return (self.preprocess(img_input), self.preprocess(img_validation), label)

    def preprocess_labelled_dataset(self, data):

        # build dataloader pipeline
        data = data.map(self.preprocess_twin)
        data = data.cache()
        data = data.shuffle(buffer_size=1024)

        return data

    def Create_Train_and_Test_Data_batches(self, data): # data is  preprocess labelled data
        
        # Training partition
        train_data = data.take(round(len(data)*.7))
        train_data = train_data.batch(16)
        train_data = train_data.prefetch(8)

        # Testing partition
        test_data = data.skip(round(len(data)*.7))
        test_data = test_data.take(round(len(data)*.3))
        test_data = test_data.batch(16)
        test_data = test_data.prefetch(8)

        return train_data, test_data

class Siamese_model():
    
    # Embedding Layer
    def make_embedding(self):
        inp = Input(shape=(105,105,3), name='input_image')
        
        # First Block
        c1 = Conv2D(64, (10,10), activation='relu')(inp)
        m1 = MaxPooling2D(64, (2,2), padding='same')(c1)

        # Second Block
        c2 = Conv2D(128, (7,7), activation='relu')(m1)
        m2 = MaxPooling2D(64, (2,2), padding='same')(c2)

        # Third Block
        c3 = Conv2D(128, (4,4), activation='relu')(m2)
        m3 = MaxPooling2D(64, (2,2), padding='same')(c3)

        # Final Embedding Block
        c4 = Conv2D(256, (4,4), activation='relu')(m3)
        f1 = Flatten()(c4)
        d1 = Dense(4096, activation='sigmoid')(f1)
        

        return Model(inputs=[inp], outputs=[d1], name='embedding')

