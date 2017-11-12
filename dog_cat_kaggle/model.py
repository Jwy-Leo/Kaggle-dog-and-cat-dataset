import tensorflow as tf
from network import Network
#define 
n_classes=2

class Dog_cat_train_classifier(Network):
	def __init__(self,trainable=True):
		self.inputs=[]
		self.data=tf.placeholder(tf.float32,shape=[None,128,128,3])
		self.groundtruth=tf.placeholder(tf.float32,shape=[None,n_classes])
		self.layers=dict({'data':self.data,'groundtruth':self.groundtruth})	
		self.keep_prob = tf.placeholder(tf.float32)
		self.trainable=trainable
		self.setup()
	def setup(self):
		(self.feed('data')
		.conv(3,3,16,1,1,name='conv1_1')
		.max_pool(2,2,2,2,padding='SAME',name='pool1')
		.conv(3, 3, 32, 1, 1, name='conv2_1')
		.max_pool(2, 2, 2, 2, padding='SAME', name='pool2')
		.fc(4096, name='fc4')
		.dropout(0.5, name='drop4')
		.fc(1024, name='fc5')
		.dropout(0.5, name='drop5')
		.fc(n_classes, relu=False, name='cls_score')
		.softmax(name='cls_prob'))
