import os
import random 
from preprocess import readfile
from scipy import misc as image_read
import numpy as np
import model 

import tensorflow as tf
class_name_index={'cat':0,'dog':1}
def randindex(max_num,dataindex,rangeindex,batch_half):
	index_num=[];
	for i in range(batch):	
		INDEXTEMP=dataindex+(i-int(batch_half/2))*rangeindex
		if INDEXTEMP<0:
			index_num.append(int(INDEXTEMP+max_num))
		elif INDEXTEMP>max_num-1:
			index_num.append(int(INDEXTEMP-max_num))
		else:
			index_num.append(int(INDEXTEMP))
	return index_num
def data_concat(index,tcat,tdog,dir_name):
	temp=[]
	label=[]
	label2=[]
	mw=4096
	mh=4096
	for i in range(len(index)):
		temp.append(image_read.imread(dir_name+'/'+tcat[index[i]]))
		label.append(0)
		if(temp[2*i].shape[0]<mw):
			mw=temp[2*i].shape[0]
		if(temp[2*i].shape[1]<mh):
			mh=temp[2*i].shape[1]
		temp.append(image_read.imread(dir_name+'/'+tdog[index[i]]))
		label.append(1)
		if(temp[2*i+1].shape[0]<mw):
			mw=temp[2*i+1].shape[0]
		if(temp[2*i+1].shape[1]<mh):
			mh=temp[2*i+1].shape[1]
	SAMPLE=random.sample(xrange(0,len(temp)),len(temp))

	COUNTER=1
	for i in SAMPLE:
		label2.append(label[i])
		imtemp=np.resize(temp[i],(128,128,temp[i].shape[2]))
		imtemp=np.reshape(imtemp,(1,imtemp.shape[0],imtemp.shape[1],imtemp.shape[2]))
		if COUNTER==1:
			COUNTER=COUNTER+1
			temp2=imtemp
		else:
			temp2=np.concatenate((temp2,imtemp),axis=0)	
	return temp2,label2
dir_name='./data/train'
data_dir=os.listdir(dir_name)
tr_cat,tr_dog=readfile(data_dir)
max_epoch=10
batch=64 	#even
#lr=0.0001

                             
net=model.Dog_cat_train_classifier()

cls_score=net.get_output('cls_score')
label=net.get_output('groundtruth')
print(cls_score)
print(label)
cross_entropy = tf.reduce_mean(-label*tf.log(tf.nn.sigmoid(cls_score)))
accr1=tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(cls_score,1),tf.arg_max(label,1)),tf.float32))
accr2=tf.reduce_mean(tf.cast(tf.equal(tf.arg_min(cls_score,1),tf.arg_min(label,1)),tf.float32))
accr=accr1+accr2
print(cross_entropy)
loss=cross_entropy
# optimizer and learning rate
global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(0.001, global_step,
                                50000, 0.1, staircase=True)
train_op = tf.train.MomentumOptimizer(lr, 0.9).minimize(loss, global_step=global_step)
#train_op=tf.train.GradientDescentOptimizer(1e-4).minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	net.load('./data/my_pretrain.npy', sess, tf.train.Saver(), True)

	run_options = None
	run_metadata = None
	for i in range(max_epoch):
		for j in range(len(tr_cat)):
			index=randindex(len(tr_cat),j,i,int(batch/2))
			data,Label=data_concat(index,tr_cat,tr_dog,dir_name)
			
			LABEL=np.zeros([len(Label),2])
			for k in range(len(Label)):
				LABEL[k][Label[k]]=1
#			print(LABEL)
			fed_dic={net.data:data,net.groundtruth:LABEL,net.keep_prob: 0.5}

			LOSS,_,accuracy=sess.run([loss,train_op,accr],
					feed_dict=fed_dic,
					options=run_options,
					run_metadata=run_metadata)
			if(i*len(tr_cat)+j)%10:
				print("LOSS:\t%.8f"%LOSS+"\taccuracy:\t%.4f"%accuracy)

		


		
