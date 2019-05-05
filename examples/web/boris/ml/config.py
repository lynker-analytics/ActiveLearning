#-------------------------------------
# database config
#-------------------------------------
db_Server='localhost'
db_Database='LA'
db_UID='###'
db_PWD='###'

#-------------------------------------
# Image and class config
#-------------------------------------
height=299
width=height
channels=3
o_features=4	#Note, this will be overridden in common after reading the data in

traindir='../boris/unclassified/'
validdir=traindir
testdir=traindir

#-------------------------------------
# NN params
#-------------------------------------
model_file='models/model.h5'
csvfile='logs/epochs.csv'

batch_size=16
initial_epochs=40

metrics=['accuracy']
loss='focal_loss'
optimizer='adam'
finetune_loss='focal_loss'
finetune_optimizer='adam'

use_multiprocessing=False
workers=6

num_epochs=1000
earlystop_patience=22
lr_patience=10
samples_per_epoch=256			#mini-mini-mini epochs... needed to allow early stopping to be useful with small dataset
validation_samples_per_epoch=256

validation_split=0.3  #Note, we're ignoring this and using a fixed set of validation data based on the filename. This is necessary to allow restarting of training.

#-------------------------------------
# Active Learning params
#-------------------------------------
#query_method='Most Entropy'
#query_method='Most Confident'
query_method='NULL'
#query_method='Most and Least Confident'
suggest_samples=samples_per_epoch
