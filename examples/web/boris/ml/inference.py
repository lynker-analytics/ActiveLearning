from common import *
from scipy.misc import imread, imresize

def main():
	model,modelage=getmodel()

	filenames=[]
	classified_filenames=data['filename'].values

	for f in listdir(traindir):
		if isfile(traindir+'/'+f) and 'jpg' in f and f not in classified_filenames:
			filenames.append(f)

	m_samples=len(filenames)
	X=np.zeros((m_samples,height,width,channels),dtype=np.float32)

	m=0
	for f in filenames:
		im=imresize(imread(traindir+'/'+f,mode='RGB'),(height,width))/255.
		X[m]=im
		m+=1

	p=model.predict(X)

	print ( p.shape )

	if query_method == 'NULL': pidx=NULL(p,suggest_samples)
	if query_method == 'Most Confident': pidx=Most_Confident(p,suggest_samples)
	if query_method == 'Least Confident': pidx=Least_Confident(p,suggest_samples)
	if query_method == 'Most and Least Confident': pidx=Most_and_Least_Confident(p,suggest_samples)
	if query_method == 'Least Margin': pidx=Least_Margin(p,suggest_samples)
	if query_method == 'Most Entropy': pidx=Most_Entropy(p,suggest_samples)

	cursor.execute("delete from boris_classify where createdby='ml'")
	cursor.commit()

	for i in pidx:
		filename=filenames[i]
		label=classes[np.argmax(p[i])]
		confidence=p[i,np.argmax(p[i])]

		#print ( filename, label, confidence )

		cursor.execute("insert into boris_classify(filename,class,confidence,createdby) values (?,?,?,'ml')", filename, label, str(confidence))
		cursor.commit()

	X=np.zeros((batch_size*int(validation_samples_per_epoch/batch_size),height,width,channels),dtype=np.float32)
	y=np.zeros((batch_size*int(validation_samples_per_epoch/batch_size),o_features),dtype=np.float32)
	for i in range(int(validation_samples_per_epoch/batch_size)):
		X_valid,y_valid=next(holdoutgen)
		while len(X_valid) != batch_size:
			X_valid,y_valid=next(holdoutgen)
		X[i*batch_size:(i+1)*batch_size],y[i*batch_size:(i+1)*batch_size] = X_valid, y_valid

	p=model.predict(X)
	
	cursor.execute("delete from boris_validation_acc where 1=1")
	cursor.commit()
	cursor.execute("insert into boris_validation_acc(valid_acc) values(?)",acc(np.argmax(y,axis=-1),np.argmax(p,axis=-1)))

	cursor.commit()

	cursor.execute("delete from [LA].[dbo].[boris_classify] where filename in ( select filename from [LA].[dbo].[boris_classify] group by filename having count(*)> 1) and createdby='ml' ")
	cursor.commit()



if __name__ == "__main__":
	main()
