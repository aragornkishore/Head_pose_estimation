import cv2
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from keras.models import Model
from keras.layers import Dense, Flatten, Input
from vgg16 import VGG16

classes = ['down', 'down,left', 'down,right', 'left', 'right', 'straight','up', 'up,left', 'up,right']

image_input = Input(shape=(160, 90, 3))

model = VGG16(input_tensor=image_input, include_top=False,weights='imagenet')
last_layer = model.get_layer('block5_pool').output
x= Flatten(name='flatten')(last_layer)
x = Dense(128, activation='relu', name='fc1')(x)
x = Dense(128, activation='relu', name='fc2')(x)
out = Dense(9, activation='softmax', name='output')(x)
custom_vgg_model2 = Model(image_input, out)
custom_vgg_model2.load_weights('transfer_learning_weights\\weights-improvement-23-0.5521.hdf5')

custom_vgg_model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

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

print(custom_vgg_model2.evaluate(X_test, Y_test, batch_size=100))