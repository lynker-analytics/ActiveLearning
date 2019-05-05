import numpy as np
import pandas as pd
import pyodbc
from os import listdir
from os.path import isfile
from config import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as acc
import tensorflow as tf
from time import time, sleep

#focal loss from here: https://github.com/mkocabas/focal-loss-keras
from keras import backend as K
def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
	return focal_loss_fixed

#this is a workaround for loading existing models
def focal_loss_fixed(y_true, y_pred):
	gamma=2.
	alpha=.25
	pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
	pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
	return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

if loss=='focal_loss': loss=focal_loss()
if finetune_loss=='focal_loss': finetune_loss=focal_loss()

conn = pyodbc.connect(
	'Driver={ODBC Driver 17 for SQL Server};'
	'Server='+db_Server+';'
	'Database='+db_Database+';'
	'UID='+db_UID+';PWD='+db_PWD+';'
)
cursor = conn.cursor()


sql = "select filename, class, createdby from boris_classify where class != 'discard' and createdby != 'ml';"
data = pd.read_sql(sql,conn)
dummies = pd.get_dummies(data,columns=['class'],prefix='',prefix_sep='')
data["IsValid"]=[int(x.replace('.jpg',''))%3 == 0 for x in data['filename'].values]


print ( data.columns )
classes=dummies.columns[1:].values.tolist()
print ( 'classes:', classes, type(classes) )

holdoutdata=data.loc[data['createdby'] == 'holdout']
validdata=data.loc[(data['IsValid'] == True) & (data['createdby'] != 'holdout')]
traindata=data.loc[(data['IsValid'] == False) & (data['createdby'] != 'holdout')]
#traindata, validdata = train_test_split(data, test_size=validation_split, random_state=42)
holdoutdata=holdoutdata.reset_index(drop=True)
validdata=validdata.reset_index(drop=True)
traindata=traindata.reset_index(drop=True)

print ( 'traindata records', len(traindata), 'validdata records', len(validdata), 'holdout records', len(holdoutdata) )

o_features=len(classes)

from keras.models import Model, Input, load_model
from keras.applications.inception_v3 import InceptionV3 #(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
from keras.layers import Dense, Conv2D, GlobalAveragePooling2D, SpatialDropout2D, Dropout, BatchNormalization as BN, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint #(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
from keras.callbacks import EarlyStopping #(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
from keras.callbacks import CSVLogger #(filename, separator=',', append=False)
from keras.callbacks import ReduceLROnPlateau #(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

MySaver=ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
MyCSV=CSVLogger(csvfile, separator=',', append=False)
MyStopper=EarlyStopping(monitor='val_loss', min_delta=0, patience=earlystop_patience, verbose=1, mode='auto', baseline=None, restore_best_weights=False)
MyLR=ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=lr_patience, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

callbacks=[MySaver,MyCSV,MyStopper]

def maketrainable(model,trainable=True):
	for layer in model.layers:
		layer.trainable=trainable
		if hasattr(layer,'layers'):
			for layer2 in layer.layers:
				layer2.trainable=trainable

def getmodel(model_file=model_file):
        modelage='new'
        if isfile(model_file):
                modelage='existing'
                model=load_model(model_file,custom_objects={'focal_loss_fixed':focal_loss_fixed})
        else:
                basemodel=InceptionV3(include_top=False, weights='imagenet', input_shape=(height,width,3))
                maketrainable(basemodel,False)
                x=basemodel.output
                x=GlobalAveragePooling2D()(x)
                x=Dropout(0.2)(x)
                output_1=Dense(o_features,activation='softmax')(x)

                model=Model(inputs=basemodel.inputs,outputs=output_1)

        return model, modelage

testdatagen = ImageDataGenerator(
	rescale=1./255
)
datagen = ImageDataGenerator(
	rescale=1./255
	,rotation_range=10
	,zoom_range=0.1
	,shear_range=0.1
	,channel_shift_range=0.1
	,fill_mode='reflect'
	,horizontal_flip=True
)

traingen=datagen.flow_from_dataframe(traindata, traindir, x_col='filename', y_col='class', has_ext=True, target_size=(height, width), color_mode='rgb', classes=classes, class_mode='categorical', batch_size=batch_size, shuffle=True)
validgen=datagen.flow_from_dataframe(validdata, validdir, x_col='filename', y_col='class', has_ext=True, target_size=(height, width), color_mode='rgb', classes=classes, class_mode='categorical', batch_size=batch_size, shuffle=True)
holdoutgen=datagen.flow_from_dataframe(holdoutdata, validdir, x_col='filename', y_col='class', has_ext=True, target_size=(height, width), color_mode='rgb', classes=classes, class_mode='categorical', batch_size=batch_size, shuffle=True)

#------------------------------------------------------------------------
# Active Learning Sample Functions
# All take a set of predictions and a number to retain as inputs
# and return back a list of indexes to the elements of the X array that
# are to be retained
#------------------------------------------------------------------------
def Most_Confident(p,n):
	#-----------------------------------------
	# given predictions p return indexes for the most confident f fraction based on the highest class prediction per sample
	# assumes p has shape (samples,classes)
	#-----------------------------------------
	return np.argsort(np.amax(p,axis=-1))[-n:]

def Least_Confident(p,n):
	#-----------------------------------------
	# given predictions p return indexes for the least confident f fraction based on the highest class prediction per sample
	# assumes p has shape (samples,classes)
	#-----------------------------------------
	return np.argsort(np.amax(p,axis=-1))[:n]

def Most_and_Least_Confident(p,n):
	#-----------------------------------------
	# half from the most confident
	# half from the least confident
	#-----------------------------------------
	n1=int(n/2.)
	n2=n-n1
	return np.concatenate((Least_Confident(p,n1), Most_Confident(p,n2)),axis=0)

def Least_Margin(p,n):
	#-----------------------------------------
	# find the samples that have the least difference between their max probably class and their second choice.
	#-----------------------------------------
	b=np.array([x[np.argsort(x)] for x in p])
	return np.argsort(b[:,-1]-b[:,-2],axis=0)[:n]

def Most_Entropy(p,n):
	return np.argsort(-np.sum(p*np.log(p),axis=1),axis=0)[-n:]

def NULL(p,n):
	return np.array([x for x in range(n)])

