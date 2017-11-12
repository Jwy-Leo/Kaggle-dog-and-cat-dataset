import numpy as np 
import copy 
def printmodel(model):
	print(type(model))
	print(model.keys())
model=np.load ('VGG_imagenet.npy')

printmodel(model.item())
model_para=model.item()
print(model_para['conv1_1']['weights'].shape)

new_model={	
		'conv1_1':	{'weights':copy.deepcopy(np.resize(model_para['conv1_1']['weights'],(3,3,3,16))),
				 'biases':copy.deepcopy(np.resize(model_para['conv1_1']['biases'],(16)))},

		'conv1_2':	{'weights':copy.deepcopy(np.resize(model_para['conv1_2']['weights'],(3,3,3,16))),
                                 'biases':copy.deepcopy(np.resize(model_para['conv1_2']['biases'],(16)))},


		'conv2_1':      {'weights':copy.deepcopy(np.resize(model_para['conv2_1']['weights'],(3,3,3,32))),
                                 'biases':copy.deepcopy(np.resize(model_para['conv2_1']['biases'],(32)))},

                'conv2_2':      {'weights':copy.deepcopy(np.resize(model_para['conv2_2']['weights'],(3,3,3,32))),
                                 'biases':copy.deepcopy(np.resize(model_para['conv2_2']['biases'],(32)))},


		'conv3_1':      {'weights':copy.deepcopy(np.resize(model_para['conv3_1']['weights'],(3,3,3,64))),
                                 'biases':copy.deepcopy(np.resize(model_para['conv3_1']['biases'],(64)))},

                'conv3_2':      {'weights':copy.deepcopy(np.resize(model_para['conv3_2']['weights'],(3,3,3,64))),
                                 'biases':copy.deepcopy(np.resize(model_para['conv3_2']['biases'],(64)))},

		'conv3_3':      {'weights':copy.deepcopy(np.resize(model_para['conv3_3']['weights'],(3,3,3,64))),
                                 'biases':copy.deepcopy(np.resize(model_para['conv3_3']['biases'],(64)))},
		'fc4':copy.deepcopy(model_para['fc6']),
		'fc5':copy.deepcopy(model_para['fc7'])}
print(new_model['conv1_1']['weights'].shape)
np.save('my_pretrain.npy',new_model)
