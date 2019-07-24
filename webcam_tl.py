import cv2
import numpy as np
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

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Unable to connect to camera.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    frame_disp = frame.copy()
    if ret:
        frame = cv2.resize(frame, (90,160))
        output_label = custom_vgg_model2.predict(np.expand_dims(frame, axis=0))
        orientation = classes[np.argmax(output_label)]
        cv2.putText(frame_disp, orientation, (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), thickness=2)
        cv2.imshow("demo", frame_disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
