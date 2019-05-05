import numpy as np

from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score as acc
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model,Input,load_model
from keras.layers import Conv2D, SpatialDropout2D, BatchNormalization as BN, GlobalAveragePooling2D, Add, MaxPooling2D, Dense, Dropout, Concatenate

from resnet20 import getmodel

batch_size=32
num_classes=10
num_epochs=1000
num_workers=6
early_stop_patience=35
lr_patience=15
model_file='models/sample_sizes_tmp.h5'
csvfile='logs/sample_sizes_{}_aug_{}.csv'

(X, y), (X_test, y_test) = cifar10.load_data()

y=to_categorical(y,num_classes)
y_test=to_categorical(y_test,num_classes)

X=X.astype(np.float32)/255.
X_test=X_test.astype(np.float32)/255.


(m_samples, height, width, channels)=X.shape
(m_samples2, o_features)=y.shape

assert m_samples == m_samples2
assert num_classes == o_features

max_validation_samples=1000
min_training_samples=100
max_training_samples=m_samples-max_validation_samples
growth_factor=1.3

datagen = ImageDataGenerator(
	rotation_range=10,
	shear_range=0.1,
	zoom_range=0.1,
	fill_mode='reflect',
	horizontal_flip=True,
	vertical_flip=False
)

t=min_training_samples
#t=45000
#t=11124
loopcount=0
while t < max_training_samples:
	X_train,X_valid, y_train,y_valid = train_test_split(X,y,test_size=int(m_samples - t), random_state=loopcount)

	X_valid=X_valid[:1000]
	y_valid=y_valid[:1000]

	for data_aug in [False,True]:
	
		model=getmodel()
	
		MyStopper=EarlyStopping(monitor='val_loss', min_delta=0, patience=early_stop_patience, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
		MySaver=ModelCheckpoint(model_file, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
		MyLR=ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=lr_patience, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
		MyCSV=CSVLogger(csvfile.format(t,data_aug), separator=',', append=False)
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
	
		model=load_model(model_file)
	
		p=model.predict(X_test)
	
		print ( 'LOG, train samples,', t, ',test accuracy,', acc(np.argmax(y_test,axis=-1),np.argmax(p,axis=-1)), ',validation accuracy,', validation_score, ',best epoch,', best_epoch, ',data aug,', data_aug, flush=True )

	t=int(t*growth_factor)
	loopcount+=1
