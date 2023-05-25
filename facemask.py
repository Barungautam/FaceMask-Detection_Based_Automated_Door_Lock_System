#importing Libraries
from calendar import EPOCH
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from keras.models import Sequential,load_model
import matplotlib.pyplot as plt
import numpy as np
import os

EPOCHS = 6


# Developing model to classify BETWEEN MASK AND NO Mask
# relu = rectified linear unit
#CONV2D =  convolution 2d
#sequential model
model=Sequential()
#layers
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(MaxPooling2D() )
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D() )
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D() ) # Maxpooling2D = Max Pooling simply says to the Convolutional Neural Network that we will carry forward only that information
model.add(Flatten()) # flatten = Flatten layers are used when we got a multidimensional output and  to make it linear to pass it onto a Dense layer
model.add(Dense(100,activation='relu')) #dense = the neurons of the layer are connected to every neuron of its preceding layer
model.add(Dense(1,activation='sigmoid')) #sigmoid = activation Function

def savegraph(H):
        global EPOCHS
        # plot the training loss and accuracy
        N = EPOCHS
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
        plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig("plot.png")

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy']) #
#Training Model
from keras.preprocessing.image import ImageDataGenerator #generating image data
#if condition to check either the model is exist or not
if not os.path.exists("mymodel.h5"):
        print('training model')
        #training Data generator 
        train_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)
        #testing data
        test_datagen = ImageDataGenerator(rescale=1./255)
        # Training Dataset
        training_set = train_datagen.flow_from_directory(
                'Train',
                target_size=(150,150),
                batch_size=16 ,
                class_mode='binary')

        test_set = test_datagen.flow_from_directory(
                'Test',
                target_size=(150,150),
                batch_size=16,
                class_mode='binary')
        # Model Training
        model_saved=model.fit(
                training_set,
                epochs=EPOCHS,
                validation_data=test_set,

                )
        #Saving Model
        model.save('mymodel.h5',model_saved)
        savegraph(model_saved)
#loading model
mymodel=load_model('mymodel.h5')

