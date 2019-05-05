from common import *

def main():
	model,modelage=getmodel()

	if modelage == 'new':
		model.compile(
			loss=finetune_loss
			,optimizer=finetune_optimizer
			,metrics=metrics
		)
		print ( model.summary(), flush=True )
	
		model.fit_generator(
			traingen
			,steps_per_epoch=int(samples_per_epoch/batch_size)
			#,validation_data=[X_valid,y_valid]
			,validation_data=validgen
			,validation_steps=int(validation_samples_per_epoch/batch_size)
			,use_multiprocessing=False
			,workers=workers
			,callbacks=callbacks
			,epochs=initial_epochs
			,max_queue_size=128
			,verbose=1
		)
	
		#model,modelage=getmodel()
	
	maketrainable(model,trainable=True)

	model.compile(
		loss=loss
		,optimizer=optimizer
		,metrics=metrics
	)
	print ( model.summary(), flush=True )

	hist=model.fit_generator(
		traingen
		,steps_per_epoch=int(samples_per_epoch/batch_size)
		#,validation_data=[X_valid, y_valid]
		,validation_data=validgen
		,validation_steps=int(validation_samples_per_epoch/batch_size)
		,use_multiprocessing=False
		,workers=workers
		,callbacks=callbacks
		,initial_epoch=initial_epochs
		,epochs=num_epochs-initial_epochs
		,verbose=1
		,max_queue_size=128
	)

	best_epoch=np.argmax(hist.history['val_acc'])
	val_acc=hist.history['val_acc'][best_epoch]

	print ( 'best epoch', best_epoch, 'best val_acc', val_acc )

	cursor.execute("delete from boris_validation_acc where 1=1")
	cursor.execute("insert into boris_validation_acc(valid_acc) values(?)",hist.history['val_acc'][best_epoch])
	cursor.commit()


if __name__ == "__main__":
	main()
