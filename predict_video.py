'''
PyPower Projects
Mask Detection Using Machine Learning
'''

#USAGE : python predict_video.py

from tensorflow.keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

#face_classifier = cv2.CascadeClassifier('./face.xml')
classifier =load_model('./mask_imagenet.h5')

class_labels = ['Mask ON','NO Mask']
cap = cv2.VideoCapture('test_no_mask.mp4')
outputFile = "no_output_fast_one.mp4"

fourcc = cv2.VideoWriter_fourcc('X','V','I','D')

vid_writer = cv2.VideoWriter(outputFile, fourcc, 30.0, (1280,720))

start_point = (15, 15)
end_point = (300, 80) 
thickness = -1

while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    labels = []
    
    gray = cv2.resize(frame,(224,224))
    roi = gray.astype('float')/255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi,axis=0)

    # make a prediction on the ROI, then lookup the class

    preds = classifier.predict(roi)[0]
    #print("\nprediction = ",preds)
    label=class_labels[preds.argmax()]
    #print("\nprediction max = ",preds.argmax())
    #print("\nlabel = ",label)
    
    if(label=='NO Mask'):
        image = cv2.rectangle(frame, start_point, end_point, (0,0,255), thickness)
        cv2.putText(image,label,(30,60),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),3)
    if(label=='Mask ON'):
        image = cv2.rectangle(frame, start_point, end_point, (0,255,0), thickness)
        cv2.putText(image,label,(30,60),cv2.FONT_HERSHEY_SIMPLEX,1.6,(0,0,0),3)
    cv2.imshow('Mask Detector',frame)
    img = cv2.resize(frame, (1280,720))
    vid_writer.write(img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
vid_writer.release()
cap.release()
cv2.destroyAllWindows()


























