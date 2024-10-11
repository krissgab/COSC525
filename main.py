import pandas as pd
import argparse
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import fashion_mnist
from matplotlib import pyplot as plt
import time
import glob
from tensorflow.keras.layers import UpSampling2D,Reshape,Conv2DTranspose,Lambda, Dropout, Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from PIL import Image
from tensorflow.keras.utils import to_categorical
from numpy import asarray
from numpy import save,load
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
from tensorflow.keras.callbacks import ModelCheckpoint
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.models import load_model


# load data, shuffle it, and then take 20% of it as testing data.
def load_data():
	n_classes = 43
	x = []
	y = []
	width = 40
	height = 40
	for i in range(n_classes):
		path = "./data/Train/{}/*".format(i)
		files = glob.glob(path)
		for file in files:
			img = Image.open(file)
			img = img.resize((height,width))
			x.append(np.array(img))
			y.append(i)

	# shuffle the data
	x = np.array(x)
	y = np.array(y)
	index = np.arange(len(y))
	np.random.seed(0)
	np.random.shuffle(index)
	x = x[index]
	y = y[index]
	x_train,y_train = (x[int(len(x) * 0.2):],y[int(len(x) * 0.2):])
	x_test,y_test = (x[0:int(len(x) * 0.2)],y[0:int(len(x) * 0.2)])
	
	x_train = x_train / 255
	x_test = x_test / 255
	
	y_train = to_categorical(y_train, n_classes)
	y_test = to_categorical(y_test, n_classes)
	return (x_train,x_test,y_train,y_test)

# generate images whose brightess range from 0.85 to 1.15 compared to original images
def brightness_augmentation(x_train, y_train):
	datagen = ImageDataGenerator(brightness_range = [0.85,1.15])
	datagen.fit(x_train)
	for X_batch, y_batch in datagen.flow(x_train, y_train, batch_size= int(x_train.shape[0])):
		return np.concatenate((x_train, X_batch / 255), axis=0),  np.concatenate((y_train, y_batch), axis = 0)

# roate the images in range(0,15)
def rotate_augmentation(x_train,y_train):
	datagen = ImageDataGenerator(rotation_range=15)
	datagen.fit(x_train)
	for X_batch, y_batch in datagen.flow(x_train, y_train, batch_size = x_train.shape[0]):
		return np.concatenate((x_train, X_batch), axis=0),  np.concatenate((y_train, y_batch), axis = 0)


# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples, n_classes=43):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	z_input = x_input.reshape(n_samples, latent_dim)
	# generate labels
	labels = randint(0, n_classes, n_samples)
	return [z_input, labels]

# gan generator generate images
def gan_augmentation(x_train, y_train, n_samples = 700):
	# load model
	model = load_model('traffic_generator.h5')
	# generate images

	n_classes = 43
	x = None
	y = None
	is_first = True
	

	for i in range(n_samples):
		# generate images
		latent_points, labels = generate_latent_points(128, n_classes)
		# specify labels
		labels = np.random.randint(0,42, size = n_classes)
		X  = model.predict([latent_points, labels])
		# scale from [-1,1] to [0,1]
		X = (X + 1) / 2.0
		
		if(i == 0):
			x = X
			y = labels
		else:
			x = np.concatenate((x, X), axis=0)
			y = np.concatenate((y,labels), axis = 0)

	return np.concatenate((x_train, x), axis=0),  np.concatenate((y_train, to_categorical(y, n_classes)), axis = 0)

#model1
def model1(argumentation = ""):

	(x_train,x_test,y_train,y_test) = load_data()
	if(argumentation == "brightness"): (x_train,y_train) = brightness_augmentation(x_train, y_train)
	if(argumentation == "rotate"): (x_train,y_train) = rotate_augmentation(x_train, y_train)
	if(argumentation == "brightness_rotate"): (x_train,y_train) = brightness_rotate_augmentation(x_train, y_train)
	if(argumentation == "gan"): (x_train,y_train) = gan_augmentation(x_train, y_train,500)
	
	#conver to grey scale
	x_train = np.dot(x_train[...,:3], [0.299, 0.587, 0.114])
	x_test = np.dot(x_test[...,:3], [0.299, 0.587, 0.114])
	print(x_train.shape)
	x_train = x_train.reshape(x_train.shape[0],np.prod(x_train.shape[1:]))
	x_test = x_test.reshape(x_test.shape[0],np.prod(x_test.shape[1:]))
	print(x_train.shape)

	epochs = 50
	if(argumentation != ""):
		batch_size = 64
		
	else:
		batch_size = 32

	model = Sequential()
	model.add(Dense(128,input_shape = (x_train.shape[1],), activation = 'tanh'))
	model.add(Dropout(0.25))
	model.add(Dense(64,activation = 'tanh'))
	model.add(Dense(43,activation = 'softmax'))
	model.summary()	

	checkpoint = ModelCheckpoint('model1' + argumentation + '.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), callbacks = [checkpoint])
	save_data(history, filename = 'model1' + argumentation + '.npy')


#model2
def model2(argumentation = ""):
	(x_train,x_test,y_train,y_test) = load_data()

	if(argumentation == "brightness"): (x_train,y_train) = brightness_augmentation(x_train, y_train)
	if(argumentation == "rotate"): (x_train,y_train) = rotate_augmentation(x_train, y_train)
	if(argumentation == "brightness_rotate"): (x_train,y_train) = brightness_rotate_augmentation(x_train, y_train)
	if(argumentation == "gan"): (x_train,y_train) = gan_augmentation(x_train, y_train,500)
	epochs = 50
	if(argumentation != ""):
		batch_size = 64
		
	else:
		batch_size = 32
	


	model = Sequential()
	model.add(Conv2D(filters = 8, kernel_size = (3,3), activation = 'relu', input_shape = x_train.shape[1:]))

	model.add(MaxPooling2D((2,2)))
	model.add(Conv2D(filters = 4, kernel_size = (3,3), activation = 'relu'))
	model.add(MaxPooling2D((2,2)))

	model.add(Flatten())
	model.add(Dense(43, activation = 'softmax'))
	model.summary()

	checkpoint = ModelCheckpoint('model2' + argumentation + '.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), callbacks = [checkpoint])
	save_data(history, filename = 'model2' + argumentation + '.npy')
	

# model3
def model3(argumentation = ""):
	(x_train,x_test,y_train,y_test) = load_data()
	if(argumentation == "brightness"): (x_train,y_train) = brightness_augmentation(x_train, y_train)
	if(argumentation == "rotate"): (x_train,y_train) = rotate_augmentation(x_train, y_train)
	if(argumentation == "brightness_rotate"): (x_train,y_train) = brightness_rotate_augmentation(x_train, y_train)
	if(argumentation == "gan"): (x_train,y_train) = gan_augmentation(x_train, y_train,500)
	epochs = 50
	if(argumentation != ""):
		batch_size = 64
		
	else:
		batch_size = 32


	model = Sequential()
	model.add(Conv2D(filters = 16, kernel_size = (3,3), activation = 'relu', input_shape = x_train.shape[1:]))
	model.add(MaxPooling2D(pool_size = (2,2)))
	model.add(Conv2D(filters=8, kernel_size=(3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dense(43, activation = 'softmax'))
	model.summary()

	checkpoint = ModelCheckpoint('model3' + argumentation + '.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), callbacks = [checkpoint])
	save_data(history, filename = 'model3' + argumentation + '.npy')
	

# save the loss and accuracy information
def save_data(history,filename):
	
	data = []

	data.append(history.history['accuracy'])
	data.append(history.history['val_accuracy'])
	data.append(history.history['loss'])
	data.append(history.history['val_loss'])
	
	plt.figure(0)
	plt.plot(data[0], label='training accuracy')
	plt.plot(data[1], label='testing accuracy')
	plt.title('Accuracy')
	plt.xlabel('epochs')
	plt.ylabel('accuracy')
	plt.legend()
	
	plt.figure(1)
	plt.plot(data[2], label='training loss')
	plt.plot(data[3], label='testing loss')
	plt.title('Loss')
	plt.xlabel('epochs')
	plt.ylabel('loss')
	plt.legend()
	plt.show()

	data = asarray(data)
	save(filename, data)
	

# plot a image form npy files.
def plot(filename):
	data = load(filename)
	plt.figure(0)
	plt.plot(data[0], label='training accuracy')
	plt.plot(data[1], label='testing accuracy')
	plt.title('Accuracy')
	plt.xlabel('epochs')
	plt.ylabel('accuracy')
	plt.legend()
	
	plt.figure(1)
	plt.plot(data[2], label='training loss')
	plt.plot(data[3], label='testing loss')
	plt.title('Loss')
	plt.xlabel('epochs')
	plt.ylabel('loss')
	plt.legend()
	plt.show()



if(sys.argv[1] == "model1"): model1(sys.argv[2])
if(sys.argv[1] == "model2"): model2(sys.argv[2])
if(sys.argv[1] == "model3"): model3(sys.argv[2])
