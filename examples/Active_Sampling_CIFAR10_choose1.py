import numpy as np
from time import time

from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score as acc, roc_auc_score as auc
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from copy import deepcopy as cp
import multiprocessing
from scipy.misc import toimage,imread,imresize,imsave

from keras.models import Model,Input,load_model
from keras.layers import Conv2D, SpatialDropout2D, BatchNormalization as BN, GlobalAveragePooling2D, Add, MaxPooling2D, Dense, Dropout, Concatenate

from resnet20 import getmodel

#------------------------------------------------------------
# Network and data parameters
#------------------------------------------------------------
batch_size=32
num_classes=2
num_epochs=1000
num_workers=6 #start with multiprocessing.cpu_count() and fine-tune from there
early_stop_patience=35
lr_patience=15
model_file='models/active_samples_cifar100_choose1_tmp.h5'
csvfile='logs/Active_Sampling_cifar100_choose1_{}_{}_{}.csv'

#------------------------------------------------------------
# Active Learning parameters
#------------------------------------------------------------
seed=100
data_aug=True
cvs=5
Active_rounds=5
Active_sample=2500
Active_fractions=[0.2]
#Active_metrics=['NULL','Most Confident','Least Confident','Least Margin','Most Entropy','Most and Least Confident']
Active_metrics=['NULL','Most Entropy']


def main():
	(X, y), (X_test, y_test) = cifar10.load_data()

	y[y!=1]=0
	y_test[y_test!=1]=0

	y=to_categorical(y,num_classes)
	y_test=to_categorical(y_test,num_classes)

	X=X.astype(np.float32)/255.
	X_test=X_test.astype(np.float32)/255.

	(m_samples, height, width, channels)=X.shape
	(m_samples2, o_features)=y.shape

	for af in range(len(Active_fractions)):
		for am in range(len(Active_metrics)):
			for cv in range(cvs):

				#give our data a good shuffle before splitting into samples
				random_state=100*am+10*af+cv
				X,y=shuffle(X,y,random_state=random_state)

				X_valid,y_valid=shuffle(X_test,y_test)

				idx=0
				X_train=X[idx:idx+seed]
				y_train=y[idx:idx+seed]

				idx+=seed
				model=''
				for i in range(Active_rounds+1):
					print ( 'LOG,',',cv,', cv, ',Active Metric,', Active_metrics[am], ',Active Fraction,', Active_fractions[af], ',Train samples,', len(X_train), ',Active Round,', i, end="" )

					del model
					t1=time()
					model, best_epoch, validation_score=trainmodel(
						X_train
						,y_train
						,X_valid[:1000]
						,y_valid[:1000]
						,Active_metrics[am].replace(' ','_')
						,Active_fractions[af]
					)
					p=model.predict(X_test)
					#test_score=acc(np.argmax(y_test,axis=-1),np.argmax(p,axis=-1))
					test_score=auc(y_test[:,0],p[:,0])
					t2=time()

					print ( ',best epoch,', best_epoch, ',validation score,', validation_score, ',test score,', test_score, ',duration,', t2-t1, flush=True )

					if i < Active_rounds:

						X_sample=X[idx:idx+Active_sample]
						y_sample=y[idx:idx+Active_sample]
						idx+=Active_sample
	
						p=model.predict(X_sample)
	
						if Active_metrics[am] == 'NULL': pidx=NULL(p,f=Active_fractions[af])
						if Active_metrics[am] == 'Most Confident': pidx=Most_Confident(p,f=Active_fractions[af])
						if Active_metrics[am] == 'Least Confident': pidx=Least_Confident(p,f=Active_fractions[af])
						if Active_metrics[am] == 'Most and Least Confident': pidx=Most_and_Least_Confident(p,f=Active_fractions[af])
						if Active_metrics[am] == 'Least Margin': pidx=Least_Margin(p,f=Active_fractions[af])
						if Active_metrics[am] == 'Most Entropy': pidx=Most_Entropy(p,f=Active_fractions[af])
	
						X_active=X_sample[pidx]
						y_active=y_sample[pidx]
	
						X_train=np.concatenate((X_train,X_active),axis=0)
						y_train=np.concatenate((y_train,y_active),axis=0)
						

def Most_Confident(p,f=0.5):
	#-----------------------------------------
	# given predictions p return indexes for the most confident f fraction based on the highest class prediction per sample
	# assumes p has shape (samples,classes)
	#-----------------------------------------
	return np.argsort(np.amax(p,axis=-1))[-int(f*len(p)):]

def Least_Confident(p,f=0.5):
	#-----------------------------------------
	# given predictions p return indexes for the least confident f fraction based on the highest class prediction per sample
	# assumes p has shape (samples,classes)
	#-----------------------------------------
	return np.argsort(np.amax(p,axis=-1))[:int(f*len(p))]

def Most_and_Least_Confident(p,f=0.5):
	#-----------------------------------------
	# half from the most confident
	# half from the least confident
	#-----------------------------------------
	return np.concatenate((Least_Confident(p,f/2.), Most_Confident(p,f/2.)),axis=0)

def Least_Margin(p,f=0.5):
	#-----------------------------------------
	# find the samples that have the least difference between their max probably class and their second choice.
	#-----------------------------------------
	b=np.array([x[np.argsort(x)] for x in p])
	return np.argsort(b[:,-1]-b[:,-2],axis=0)[:int(f*len(p))]

def Most_Entropy(p,f=0.5):
	return np.argsort(-np.sum(p*np.log(p),axis=1),axis=0)[-int(f*len(p)):]
	
def NULL(p,f=0.5):
	return np.array([x for x in range(int(f*len(p)))])
	


def trainmodel(X_train,y_train,X_valid,y_valid,metric,fraction,data_aug=True):

	model=getmodel(num_classes=num_classes)
	
	MyStopper=EarlyStopping(monitor='val_loss', min_delta=0, patience=early_stop_patience, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
	MySaver=ModelCheckpoint(model_file, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
	MyLR=ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=lr_patience, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
	MyCSV=CSVLogger(csvfile.format(len(X_train),metric,fraction), separator=',', append=False)
	callbacks=[MyStopper,MySaver,MyLR,MyCSV]
		
	model.compile(
		loss='categorical_crossentropy'
		,optimizer='Adam'
		,metrics=['accuracy']
	)
		
	#print ( model.summary() )
	
	if data_aug == False:
		hist=model.fit(
			x=X_train
			,y=y_train
			,batch_size=batch_size
			,epochs=num_epochs
			,validation_data=(X_valid,y_valid)
			,callbacks=callbacks
			,verbose=0
		)
	else:
		datagen = ImageDataGenerator(
			rotation_range=10
			,shear_range=0.1
			,zoom_range=0.1
			,fill_mode='reflect'
			,horizontal_flip=True
			,vertical_flip=False
		)

		hist=model.fit_generator(
				datagen.flow(X_train, y_train, batch_size=batch_size)
				,epochs=num_epochs
				,steps_per_epoch=int(len(X_train)/batch_size)
				,validation_data=(X_valid, y_valid)
				,callbacks=callbacks
				,max_queue_size=32
				,use_multiprocessing=False
				,workers=num_workers
				,verbose=0
		)
		
	
	best_epoch=np.argmax(hist.history['val_acc'])
	validation_score=hist.history['val_acc'][best_epoch]
	
	return load_model(model_file), best_epoch, validation_score

if __name__ == "__main__":
	main()	
