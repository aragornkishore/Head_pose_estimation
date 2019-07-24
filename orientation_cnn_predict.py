import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

classes = ['down', 'down,left', 'down,right', 'left', 'right', 'straight','up', 'up,left', 'up,right']
model = Sequential()

#add model layers
model.add(Conv2D(64, kernel_size=7, strides=(2,2), activation='relu', input_shape=(160,90,3)))
model.add(Conv2D(32, kernel_size=5, strides=(2,2), activation='relu'))
model.add(Conv2D(16, kernel_size=3, strides=(2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(9, activation='softmax'))

model.load_weights('orientation_cnn_weights\\weights-improvement-19-0.3600.hdf5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

df = np.array(pd.read_csv('dataset.csv'))

encoder = LabelBinarizer()
labels = encoder.fit_transform(df[:,1])
print(encoder.classes_)

images = np.zeros([2000, 160, 90, 3])

i = 0

for image in df[:,0]:
    frame = cv2.imread(('unlabelled_images\\' + image + '.png'),1)
    frame_resized = cv2.resize(frame, (90,160))
    images[i,:,:,:] = frame_resized
    i += 1

images = images / 255
X_train = images[:1200,:,:,:]
Y_train = labels[:1200,:]

X_test = images[1200:,:,:,:]
Y_test = labels[1200:,:]

print(model.evaluate(X_test, Y_test))