import os 
import numpy as np
import cv2
from head_pose_img_script import detect_blobs,sort_image_points, head_pose
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.callbacks import ModelCheckpoint

model_points = np.array([   (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corner
                            (0.0, 0.0, 0.0),             # Nose tip
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0),      # Right mouth corner
                            (0.0, -330.0, -65.0)        # Chin                      
                        ])
 
 
# Camera internals
camera_matrix = np.array([[720,   0, 360],
                          [  0, 720, 640], 
                          [  0,   0,   1]], dtype = "double")

images = np.zeros([1600, 320, 180, 3])
euler_angles = np.zeros([1600, 3])

for filename in sorted(os.listdir('images')):
    i = int(filename.split('_')[1].split('.')[0])
    frame = cv2.imread(('images\\' + filename),1)
    frame_resized = cv2.resize(frame, (180,320))
    
    img_points = detect_blobs(frame)
    sorted_img_points = sort_image_points(img_points)
    euler_angle = head_pose(model_points, sorted_img_points, camera_matrix)
    

    images[i-1,:,:,:] = frame_resized
    euler_angles[i-1,:] = euler_angle.ravel()





#create model
model = Sequential()

#add model layers
model.add(Conv2D(64, kernel_size=7, activation='relu', input_shape=(320,180,3)))
model.add(Conv2D(32, kernel_size=7, activation='relu'))
model.add(Flatten())
model.add(Dense(3, activation='softmax'))



#compile model using accuracy as a measure of model performance
model.compile(loss='mean_squared_error', optimizer='sgd')
    


filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

#Model is trained using the fit method
model.fit(images, euler_angles, epochs=10, batch_size=32, callbacks=callbacks_list)






