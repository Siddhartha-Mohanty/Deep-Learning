#!/usr/bin/env python
# coding: utf-8

# # Type-1

# In[ ]:


from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, AveragePooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.layers import UpSampling2D

# Convert labels to one-hot encoding
num_classes = len(classes)
Y_train_one_hot = to_categorical(Y_train, num_classes=num_classes)
Y_val_one_hot = to_categorical(Y_val, num_classes=num_classes)

# Input layer
input_layer = Input(shape=(img_size, img_size, 3))

from keras.layers import Activation

def vgg16_block(x):
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    return x


from keras.layers import UpSampling2D

from keras.layers import Conv2DTranspose

# Function to create the Inception-Reduction block
def inception_reduction_block(x):
    # Define the inception block
    inception = Conv2D(64, (1, 1), activation='relu', padding='same')(x)
    inception = Conv2D(128, (3, 3), activation='relu', padding='same')(inception)
    
    # Define the reduction block
    reduction = Conv2D(128, (1, 1), activation='relu', padding='same')(x)
    
    # Transpose Convolutional layer instead of UpSampling2D
    reduction = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(reduction)
    
    # Convolutional layer instead of UpSampling2D
    x1 = Conv2D(128, (1, 1), activation='relu', padding='same')(x)
    
    # Transpose Convolutional layer instead of UpSampling2D
    x1 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x1)
    
    # Ensure both x1 and reduction have the same spatial dimensions
    x1 = Conv2D(128, (1, 1), activation='relu', padding='same')(x1)
    
    # Concatenate along the channel axis
    concatenated = concatenate([x1, reduction], axis=-1)
    
    return concatenated


# Function to create the Inception-Reduction block for Block 3
def inception_reduction_block_3(x):
    # Inception block
    x1 = Conv2D(64, (7, 1), activation='relu', padding='same')(x)
    x1 = Conv2D(64, (1, 7), activation='relu', padding='same')(x1)
    
    x2 = Conv2D(64, (7, 1), activation='relu', padding='same')(x)
    x2 = Conv2D(64, (1, 7), activation='relu', padding='same')(x2)
    
    x3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    x3 = Conv2D(64, (1, 1), activation='relu', padding='same')(x3)
    
    inception_output = concatenate([x1, x2, x3], axis=-1)
    
    # Reduction block
    reduction = Conv2D(128, (3, 3), activation='relu', padding='same')(inception_output)
    reduction = MaxPooling2D((2, 2), strides=(2, 2))(reduction)
    
    return reduction

# VGG16 block
block1_output = vgg16_block(input_layer)

# Inception-Reduction block for Block 2
block2_output = inception_reduction_block(block1_output)

# Inception-Reduction block for Block 3
block3_output = inception_reduction_block_3(block2_output)

# Classification block (Block 4)
x = AveragePooling2D((7, 7))(block3_output)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output_layer = Dense(num_classes, activation='sigmoid')(x)

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)
model.summary()


# # Type-2

# In[ ]:


from keras.models import Model
from keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, Reshape
from keras.applications import VGG16, InceptionV3
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Assuming img_size, classes, and X_train, Y_train, X_val, Y_val are defined

# Define batch size
batch_size = 32

# Data augmentation for the training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# No augmentation for the validation set
val_datagen = ImageDataGenerator(rescale=1./255)

# Base VGG16 model
base_model_vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Apply VGG16 to training and validation sets
X_train_vgg16 = base_model_vgg16.predict(X_train)
X_val_vgg16 = base_model_vgg16.predict(X_val)

# Create data generators
train_generator = train_datagen.flow(X_train_vgg16, Y_train, batch_size=batch_size)
val_generator = val_datagen.flow(X_val_vgg16, Y_val, batch_size=batch_size)

# Load the pre-trained VGG16 model
input_shape_vgg16 = (img_size, img_size, 3)  # Adjusted input shape
input_layer_vgg16 = Input(shape=input_shape_vgg16)

# VGG16 model
base_model_vgg16 = VGG16(weights='imagenet', include_top=False, input_tensor=input_layer_vgg16)

# Get the output tensor of VGG16
output_vgg16 = base_model_vgg16.output

# Global Average Pooling
output_vgg16 = GlobalAveragePooling2D()(output_vgg16)

# Dense layer with ReLU activation
output_vgg16 = Dense(512, activation='relu')(output_vgg16)

# Dropout for regularization
output_vgg16 = Dropout(0.2)(output_vgg16)

# Create a model using VGG16 as the base model
model_vgg16 = Model(inputs=input_layer_vgg16, outputs=output_vgg16)

# Freeze the layers in the pre-trained VGG16 model
for layer in base_model_vgg16.layers:
    layer.trainable = False

# InceptionV3 model with VGG16 output as input
input_layer_inceptionv3 = Input(shape=(int(output_vgg16.shape[1]),))

# Reshape the input to (1, 1, 512)
input_layer_inceptionv3_reshaped = Reshape((1, 1, 512))(input_layer_inceptionv3)


# InceptionV3 model
base_model_inceptionv3 = InceptionV3(weights='imagenet', include_top=False, input_tensor=input_layer_inceptionv3_reshaped)

# Get the output tensor of InceptionV3
output_inceptionv3 = base_model_inceptionv3.output

# Global Average Pooling
output_inceptionv3 = GlobalAveragePooling2D()(output_inceptionv3)

# Dense layer with ReLU activation
output_inceptionv3 = Dense(512, activation='relu')(output_inceptionv3)

# Dropout for regularization
output_inceptionv3 = Dropout(0.2)(output_inceptionv3)

# Final output layer for binary classification
predictions = Dense(len(classes), activation='sigmoid')(output_inceptionv3)

# Create the final model with VGG16 output as input and InceptionV3 output as the final output
model = Model(inputs=model_vgg16.input, outputs=predictions)

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model using data generators
epochs = 8  # You can adjust the number of epochs
history = model.fit(train_generator, epochs=epochs, verbose=1, validation_data=val_generator)


# # Type-3

# In[ ]:


from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, AveragePooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from keras.layers import UpSampling2D

# Convert labels to one-hot encoding
num_classes = len(classes)
Y_train_one_hot = to_categorical(Y_train, num_classes=num_classes)
Y_val_one_hot = to_categorical(Y_val, num_classes=num_classes)

# Input layer
input_layer = Input(shape=(img_size, img_size, 3))

from keras.layers import Activation

def vgg16_block(x):
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    return x


# from keras.layers import UpSampling2D

# from keras.layers import Conv2DTranspose

# Function to create the Inception-Reduction block
def inception_reduction_block(x):
    # Define the inception block
    inception = Conv2D(64, (1, 1), activation='relu', padding='same')(x)
    inception = Conv2D(128, (3, 3), activation='relu', padding='same')(inception)

    # Define the reduction block
    reduction = Conv2D(128, (1, 1), activation='relu', padding='same')(x)
    
    # Use AveragePooling2D instead of Conv2DTranspose
    reduction = AveragePooling2D((2, 2), strides=(2, 2), padding='same')(reduction)

    # Convolutional layer instead of UpSampling2D
    x1 = Conv2D(128, (1, 1), activation='relu', padding='same')(x)
    
    # Use AveragePooling2D instead of Conv2DTranspose
    x1 = AveragePooling2D((2, 2), strides=(2, 2), padding='same')(x1)

    # Ensure both x1 and reduction have the same spatial dimensions
    x1 = Conv2D(128, (1, 1), activation='relu', padding='same')(x1)
    
    # Concatenate along the channel axis
    concatenated = concatenate([x1, reduction], axis=-1)
    
    return concatenated


# Function to create the Inception-Reduction block for Block 3
def inception_reduction_block_3(x):
    # Inception block
    x1 = Conv2D(64, (7, 1), activation='relu', padding='same')(x)
    x1 = Conv2D(64, (1, 7), activation='relu', padding='same')(x1)
    
    x2 = Conv2D(64, (7, 1), activation='relu', padding='same')(x)
    x2 = Conv2D(64, (1, 7), activation='relu', padding='same')(x2)
    
    x3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
    x3 = Conv2D(64, (1, 1), activation='relu', padding='same')(x3)
    
    inception_output = concatenate([x1, x2, x3], axis=-1)
    
    # Reduction block
    reduction = Conv2D(128, (3, 3), activation='relu', padding='same')(inception_output)
    reduction = MaxPooling2D((2, 2), strides=(2, 2))(reduction)
    
    return reduction

# VGG16 block
block1_output = vgg16_block(input_layer)

# Inception-Reduction block for Block 2
block2_output = inception_reduction_block(block1_output)

# Inception-Reduction block for Block 3
block3_output = inception_reduction_block_3(block2_output)

# Classification block (Block 4)
# Classification block (Block 4)
x = AveragePooling2D((2, 2))(block3_output)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output_layer = Dense(num_classes, activation='sigmoid')(x)

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)
model.summary()

