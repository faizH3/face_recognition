import os
from PIL import Image
import numpy as np
import cv2
import pickle

dir_name = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(dir_name, "images")

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(img_dir):
    for file in files:
        if file.endswith('png') or file.endswith('jpg'):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            print(label, path)
            if label in label_ids:
                pass
            else:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            print(label_ids)
            # y_labels.append(label) # some number
            # x_train.append(path) # verify this image, turn into numpy ARRAY, GRAY
            pil_img = Image.open(path).convert('L')
            img_array = np.array(pil_img, "uint8")
            print(img_array)
            faces = face_cascade.detectMultiScale(img_array, scaleFactor=1.5, minNeighbors=5) 

            for (x, y, w, h) in faces:
                roi = img_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

print(y_labels)
print(x_train)

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")