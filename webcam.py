import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

classes = ['down', 'down,left', 'down,right', 'left', 'right', 'straight','up', 'up,left', 'up,right']
model = Sequential()

#add model layers
model.add(Conv2D(64, kernel_size=7, strides=(2,2), activation='relu', input_shape=(160,90,3)))
model.add(Conv2D(32, kernel_size=5, strides=(2,2), activation='relu'))
model.add(Conv2D(16, kernel_size=3, strides=(2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(9, activation='softmax'))

model.load_weights('orientation_cnn_weights\\weights-improvement-19-0.3600.hdf5')

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
        output_label = model.predict(np.expand_dims(frame, axis=0))
        orientation = classes[np.argmax(output_label)]
        cv2.putText(frame_disp, orientation, (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), thickness=2)
        cv2.imshow("demo", frame_disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
