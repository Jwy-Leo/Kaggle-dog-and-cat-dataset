import os 

train_filenames=os.listdir('./data/train')
def readfile(train_filenames):
	tr_cat=filter(lambda x:x[:3]=='cat',train_filenames)
	tr_dog=filter(lambda x:x[:3]=='dog',train_filenames)
	#print(tr_cat)
	#print(type(tr_dog))
	return tr_cat,tr_dog

