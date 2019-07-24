import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import cv2
from keras.layers import Dense, Flatten
from vgg16 import VGG16
from keras.layers import Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint

df = np.array(pd.read_csv('dataset.csv'))

encoder = LabelBinarizer()
labels = encoder.fit_transform(df[:,1])
print(encoder.classes_)

images = np.zeros([2000, 160, 90, 3])

i = 0

for im in df[:,0]:
    frame = cv2.imread(('unlabelled_images\\' + im + '.png'),1)
    frame_resized = cv2.resize(frame, (90,160))
    images[i,:,:,:] = frame_resized
    i += 1

images = images / 255
X_train = images[:1200,:,:,:]
Y_train = labels[:1200,:]

X_test = images[1200:,:,:,:]
Y_test = labels[1200:,:]



image_input = Input(shape=(160, 90, 3))

model = VGG16(input_tensor=image_input, include_top=False,weights='imagenet')

model.summary()

last_layer = model.get_layer('block5_pool').output
x= Flatten(name='flatten')(last_layer)
x = Dense(128, activation='relu', name='fc1')(x)
x = Dense(128, activation='relu', name='fc2')(x)
out = Dense(9, activation='softmax', name='output')(x)
custom_vgg_model2 = Model(image_input, out)
custom_vgg_model2.summary()

# freeze all the layers except the dense layers
for layer in custom_vgg_model2.layers[:-3]:
	layer.trainable = False

custom_vgg_model2.summary()

custom_vgg_model2.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

filepath="transfer_learning_weights\\weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

#custom_vgg_model2.fit(X_train, Y_train, batch_size=100, epochs=12, verbose=1, validation_data=(X_test, Y_test))
custom_vgg_model2.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=25, batch_size=200, callbacks=callbacks_list)