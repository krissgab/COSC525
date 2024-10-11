import sys
import numpy as np
import time
import glob
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from PIL import Image

# load data
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

# predict the output
def prediction():
	(x_train,x_test,y_train,y_test) = load_data()
	
	x_grey_test = np.dot(x_test[...,:3], [0.299, 0.587, 0.114])
	x_grey_test = x_grey_test.reshape(x_grey_test.shape[0],np.prod(x_grey_test.shape[1:]))
	

	y = np.zeros(y_test.shape)
	y_test = np.argmax(y_test, axis = 1)


	models = []
	print(len(sys.argv))
	for i in range(1, len(sys.argv)):
		models.append(load_model(sys.argv[i]))

	start = time.time()
	for model in models:
		if(model.layers[0].input_shape[1] == 1600):
			y += model.predict(x_grey_test)
		else:
			y += model.predict(x_test)

	
	y = np.argmax(y,axis = 1)
	

	acc = 0
	for label,correct in zip(y,y_test):
		
		if(label == correct): acc += 1

	print("accuracy: {}".format(acc/y.shape[0]))

prediction()

