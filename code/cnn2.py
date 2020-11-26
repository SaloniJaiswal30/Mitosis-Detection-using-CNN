import numpy as np
import os
import cv2
import csv as csv
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score, precision_score

def data_preprocess(path,d):
	img=[]
	label=[]
	count=0
	for j in os.listdir(path):
		count = count+1
		if os.path.isfile(path+j):
			try:		
				dst = cv2.imread(path+j,1)
				#print(dst.shape)
			except IOError:
				continue
		img.append(dst)
		label.append(d[count-1])
	data=np.array(img)
	data =data / 255
	label = np.array(label)
	label1 = np_utils.to_categorical(label,2)
	return data,label1,label

def cnn(xtrain,ytrain,yytrain,xtest,ytest,yytest):
	#Declare a sequential model
	model=Sequential()
	#Input layer
	model.add(Convolution2D(96,(11,11),activation='relu',input_shape=xtrain[0].shape,data_format='channels_last'))
	model.add(MaxPooling2D(pool_size=(3,3)))
	model.add(Convolution2D(384,(3,3),activation='relu',data_format='channels_last'))
	model.add(MaxPooling2D(pool_size=(3,3)))
	#Normalisation layer
	layer = LocalResponseNormalisation()
	layer_config = layer.get_config()
	layer_config["input_shape"] = xtrain.shape
	layer = layer.__class__.from_config(layer_config)
	model.add(layer)
	model.add(Convolution2D(384,(3,3),activation='relu',data_format='channels_last',padding="same"))
	model.add(Convolution2D(384,(3,3),activation='relu',data_format='channels_last',padding="same"))
	model.add(Convolution2D(128,(3,3),activation='relu',data_format='channels_last',padding="same"))
	model.add(MaxPooling2D(pool_size=(3,3),padding="same"))
	#Fully connected layers left
	model.add(Flatten())
	model.add(Dense(4096,activation='relu'))
	model.add(Dense(4096,activation='relu'))
	#get the output of this layer
	model.add(Dense(2,activation='softmax'))
	#print(model.layers[0].input,K.learning_phase())
	get_activations=K.function([model.layers[0].input,K.learning_phase()],[model.layers[11].output])
	feature=get_activations([xtrain,0])
	feature1=get_activations([xtest,0])
	#print(feature,len(feature[0]),len(feature[0][0]))
	#2 refers to two class labels
	model.compile(loss='categorical_crossentropy',
				optimizer='adam',
				metrics=['accuracy'])
	model.fit(xtrain,ytrain)
	score=model.evaluate(xtest,ytest)
	#print(score) #Score if we use the softmax classification instead of svm
	#model.summary()
	svc = svm.SVC(kernel='rbf', C=1,gamma='auto')
	xs=np.array(feature[0])
	#print(feature[0])
	ys=np.array(yytrain)
	print(xs.shape,ys.shape)
	print(xtest.shape,yytest.shape)
	svc.fit(xs,ys)
	score=svc.score(np.array(feature1[0]),np.array(yytest))
	print("accuracy:",score)
	predicted=svc.predict(np.array(feature1[0]))
	#print(np.array(yytest),predicted)
	cnfm=confusion_matrix(np.array(yytest),predicted)
	p=cnfm[0][0]/(cnfm[0][0]+cnfm[1][0])
	r=cnfm[0][0]/(cnfm[0][0]+cnfm[0][1])
	print("confusion matrix:\n",cnfm)
	f_s=2*p*r/(p+r)
	print("precision:",cnfm[0][0]/(cnfm[0][0]+cnfm[1][0]))
	print("recall:",cnfm[0][0]/(cnfm[0][0]+cnfm[0][1]))
	print("f_score",f_s)    
	
	
class LocalResponseNormalisation(Layer):
	
	def __init__(self, n=5, alpha=0.0005, beta=0.75, k=2, **kwargs):
		self.n =n
		self.alpha = alpha
		self.beta = beta
		self.k = k
		super(LocalResponseNormalisation, self).__init__(**kwargs)
	
	def build(self,input_shape):
		self.shape = input_shape
		super(LocalResponseNormalisation, self).build(input_shape)

	def call(self,x,mask=None):
		_,r,c,f = self.shape
	#	_,f,r,c = self.shape
		squared = K.square(x)
		#print r,c,f,K.image_data_format
		#print squared
		pooled = K.pool2d(squared, (self.n, self.n), strides=(1, 1),padding="same", pool_mode="avg")
		#print pooled		
		summed = K.sum(pooled, axis=1, keepdims=True)
		averaged = self.alpha * K.repeat_elements(summed, r, axis=1)
		denom = K.pow(self.k + averaged, self.beta)
		#print summed,averaged,x,denom
		return x / denom	

	def compute_output_shape(self, input_shape):
		return input_shape

path='../output1/'
data=csv.reader(open('trainlabels.csv','r'))
d=[]
for j in data:
	d.append(j[0])
train,trainlabelcat,trainlabel=data_preprocess(path,d)
print(train.shape)
path='../output2/'
data=csv.reader(open('testlabels.csv','r'))
del d[:]
for j in data:
	d.append(j[0])
test,testlabelcat,testlabel=data_preprocess(path,d)
print(test.shape)
cnn(train,trainlabelcat,trainlabel,test,testlabelcat,testlabel)
