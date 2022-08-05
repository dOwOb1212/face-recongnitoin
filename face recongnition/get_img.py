import cv2
import os
import uuid 
import tensorflow as tf

def Collect_Positive_and_Anchor_Classes(self):
    # Setup paths
    POS_PATH = os.path.join('data', 'positive')
    ANC_PATH = os.path.join('data', 'anchor')
    
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

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu,True)
    Collect_Positive_and_Anchor_Classes()
