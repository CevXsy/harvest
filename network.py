import numpy as np

class Network(object):
	"""
	   初始化神经网络
	   num_layer有多少层
	   size输入每层有多少神经元
	   biases隐藏层和输出层偏差
	   weight神经网络层之间的权重
	"""
	def __init__(self,size):
		self.num_layer = len(size)
		self.size = size
		biases = [np.random.randn(y,1) for y in size[1:]]
		weight = [np.random.randn(y,x) for x,y in zip(size[:-1],size[1:])]

	def feedForward(self,a):
		for w,b in zip(self.weight,self.biases):
			a = Sigmoid(np.dot(w,a)+b)
		return a
	"""
	    随机梯度下降
        training_data 训练集
		epochs  训练次数
		mini_batch_size 训练集中采样个数
		eta 学习率
		test_data 测试集
        
	"""
	def SGD(self,training_data,epochs,mini_batch_size,eta,test_data=None):
		if test_data:
			n_test = len(test_data)
		n = len(train_data)
		for i in xrange(epochs):
			random.shuffle(training_data)
			mini_batchs =[
				training_data[k:k+mini_batch_size] for k in xrange(0,n,mini_batch_size)
			]
			for mini_batch in mini_bachs:
			    self.update_mini_batch(mini_batch,eta)
			if test_data:
			    print("Epoch {0}: {1}/{2}".format(epochs,self.evaluate(test_data),n_test))
			else:
			    print("Epoch {0} complete".format(j))
	"""
		更新权重和偏向的值
	"""
	def update_mini_batch(self,mini_batch,eta):
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weight]
		for x,y in mini_batch:
			delta_nabla_b,delta_nabla_w = self.backprop(x,y)
			nabla_b=[nb+dnb for nb,dnb in zip(nable_b,delta_nabla_b)]
			nabla_w=[nw+dnw for nw,dnw in zip(nable_w,delta_nabla_w)]
		self.weight = [
			w-(eta/len(mini_batch))*nw for w,nw in zip(self.weight,nabla_w)
		]
		self.biases = [
			b-(eta/len(mini_batch))*nb for b,nb in zip(self.biases,nabla_b)
		]

