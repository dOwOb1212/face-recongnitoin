import os
import numpy as np

from matplotlib import pyplot as plt
from keras import  Model
from keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from keras.metrics import Precision, Recall


import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
	tf.config.experimental.set_memory_growth(physical_devices[0],True)	
	logical_devices = tf.config.list_logical_devices("GPU")

def preprocess(file_path):

    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    
    # Load in the image
    img = tf.io.decode_jpeg(byte_img)

    # preprocessing resize image to 105x105x3
    img = tf.image.resize(img,(105,105))

    # scale image to between 0 and 1
    img = img / 255.0
    
    return img


def preprocess_twin(img_input, img_validation, label):
    return (preprocess(img_input), preprocess(img_validation), label)


# Embedding Layer
def make_embedding():
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


class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


def make_siamese_model(): 
    
    embedding = make_embedding()

    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(105,105,3))
    
    # Validation image in the network 
    validation_image = Input(name='validation_img', shape=(105,105,3))
    
    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(validation_image))
    
    # Classification layer 
    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

siamese_model = make_siamese_model()

# Loss and Optimizer
binary_cross_loss = tf.keras.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4)


@tf.function
def train_step(batch):

    # Record all of our operations 
    with tf.GradientTape() as tape:     
        # Get anchor and positive/negative image
        X = batch[:2]
        # Get label
        y = batch[2]
        
        # Forward pass
        y_pred = siamese_model(X, training=True)
        # Calculate loss
        loss = binary_cross_loss(y, y_pred)
        
    # Calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)
    
    # Calculate updated weights and apply to siamese model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))
    
    # Return loss
    return loss


def train(data, EPOCHS, checkpoint_prefix):
    # Loop through epochs
    for epoch in range(1, EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))
        
        # creating a metric object
        r = Recall()
        p = Precision()

        # Loop through each batch
        for idx, batch in enumerate(data):
            
            # Run train step here
            loss = train_step(batch).numpy()
            yhat = siamese_model.predict(batch[:2],verbose=0)
            r.update_state(batch[2], yhat)
            p.update_state(batch[2], yhat)
            progbar.update(idx+1)
        
        print(loss, r.result().numpy(), p.result().numpy())

        # Save checkpoints
        if epoch % 10 == 0: 
            checkpoint.save(file_prefix=checkpoint_prefix)


if __name__ == '__main__':


    # Setup paths
    POS_PATH = os.path.join('data', 'positive')
    NEG_PATH = os.path.join('data', 'negative')
    ANC_PATH = os.path.join('data', 'anchor')

    # get image directories
    anchor = tf.data.Dataset.list_files(ANC_PATH+'/*.jpg').take(3000)
    positive = tf.data.Dataset.list_files(POS_PATH+'/*.jpg').take(3000)
    negative = tf.data.Dataset.list_files(NEG_PATH+'/*.jpg').take(3000)

    # (anchor, positive) => 1
    # (anchor, negative) => 0

    val_positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
    val_negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
    data = val_positives.concatenate(val_negatives)

    # build dataloader pipeline
    data = data.map(preprocess_twin)
    data = data.cache()
    data = data.shuffle(buffer_size=10000)

    # Training partition
    train_data = data.take(round(len(data)*.7))
    train_data = train_data.batch(16)
    train_data = train_data.prefetch(8)

    # Testing partition
    test_data = data.skip(round(len(data)*.7))
    test_data = test_data.take(round(len(data)*.3))
    test_data = test_data.batch(16)
    test_data = test_data.prefetch(8)

    # establish checkpoints
    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(opt=opt, siamese_model=siamese_model)

    # train model 
    EPOCHS = 50
    train(train_data, EPOCHS, checkpoint_prefix=checkpoint_prefix)

    # save model
    siamese_model.save('siamesemodel.h5')
