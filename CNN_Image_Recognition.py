#### Data pre-processing ####
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


keras = tf.keras
# Pre-process the training set ONLY to avoid overfitting
# Overfitting happens when the network gets very close to 100% recognizing the training images
# but will have very low accuracy on any other image outside of the training set
# This process applies simple geometrical transformations (zooms / flips / rotations)

# Image augumentation
train_datagen = ImageDataGenerator(
    rescale= 1./255, # applies feature scaling to each pixel
    shear_range=0.2,
    zoom_range=0.2, 
    horizontal_flip=True
)
training_set = train_datagen.flow_from_directory(
    'training_set',
    target_size=(64, 64), # resize images fro faster results
    batch_size=32,
    class_mode='binary'
)

# Preprocessing the test set
test_datagen = ImageDataGenerator(rescale=1./255) # only feature scale the pixels
test_set = test_datagen.flow_from_directory(
    'test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

#### Building the CNN ####
cnn = keras.models.Sequential()
# Convolution
cnn.add(keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
# 32 filters
# each filter is a 3 by 3 pattern
# activation is rectifier func
# shape is the image size, and the dimension of the layers - in this case we're using R G B = 3 layers

# Pooling (max pooling)
cnn.add(keras.layers.MaxPool2D(pool_size= 2, strides= 2))

# Second Convolutional layer with max pooling
cnn.add(keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(keras.layers.MaxPool2D(pool_size= 2, strides= 2))

# Flattening
cnn.add(keras.layers.Flatten())

# Full Connection
cnn.add(keras.layers.Dense(units=128, activation='relu'))

# Output Layer
cnn.add(keras.layers.Dense(units=1, activation='sigmoid'))

#### Train the CNN ####
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Training and evaluation happens at the same time for visual programs
cnn.fit(x=training_set, validation_data=test_set, batch_size=32, epochs=25)


#### Prediction ####
test_image = tf.keras.utils.load_img('single_prediction/twisted_covid.jpg', target_size=(64, 64))
test_image = tf.keras.utils.img_to_array(test_image) # Convert the image to a 2D array
test_image = np.expand_dims(test_image, axis=0) # Because the network was train on batches, we need to add an extra dimension to the image
result = cnn.predict(test_image/255.0)

training_set.class_indices # Get the category indices
if(result[0][0] > 0.5): 
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)