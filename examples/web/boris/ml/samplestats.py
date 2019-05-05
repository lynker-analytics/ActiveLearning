from common import *
from scipy.misc import imread, imresize

def main():
	model=loadmodel()

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


	p=shuffle(p)[:500]

	p=np.array([x[np.argsort(x)] for x in p])

	suggest_samples=50

	pidx1=Most_Confident(p,suggest_samples)
	pidx2=Least_Confident(p,suggest_samples)
	pidx3=Most_and_Least_Confident(p,suggest_samples)
	pidx4=Least_Margin(p,suggest_samples)
	pidx5=Most_Entropy(p,suggest_samples)

	for i in range(len(p)):
		for c in range(len(p[i])):
			print ( p[i,c], ',', end='' )
		if i in pidx1:
			print ( '1,', end='' )
		else:
			print ( '0,', end='' )
		if i in pidx2:
			print ( '1,', end='' )
		else:
			print ( '0,', end='' )
		if i in pidx3:
			print ( '1,', end='' )
		else:
			print ( '0,', end='' )
		if i in pidx4:
			print ( '1,', end='' )
		else:
			print ( '0,', end='' )
		if i in pidx5:
			print ( '1,', end='' )
		else:
			print ( '0,', end='' )
		print ( '' )


if __name__ == "__main__":
	main()
