# -*- coding: utf-8 -*-
"""
Sample code for
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf

Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
"""
import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import re
import warnings
import sys
import time
warnings.filterwarnings("ignore") 
#reload(sys)
#sys.setdefaultencoding( "utf-8" )  

#词向量维数
vector_size = 50
#隐层输入size
hidden_layer_input_size = 100
#输出size 二分类问题即为2
hidden_layer_output_size = 2

#激活函数
#different non-linearities
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)
def Tanh(x):
    y = T.tanh(x)
    return(y)
def Iden(x):
    y = x
    return(y)

'''
训练卷积神经网络模型
	datasets 训练与测试数据
	U 词向量矩阵
	img_w 词向量维度
	filter_hs 滤波器组中每个滤波器的高度
	hidden_units MLP层输入输出的size
	dropout_rate 一个优化参数，每次将该比例的连接置为无效，降低过拟合 其比率为50% 
	n_epochs 迭代的最大次数
	batch_size 批量训练的样本数量
	lr_decay 梯度下降的一个参数，随着训练次数的增加，梯度下降每次更新的幅度将减小
	conv_non_linear 
	activations 使用的激活函数
	sqr_norm_lim 正则化参数
	non_static 是否在训练卷积神经网络的同时更新词向量矩阵   
	
'''
def train_conv_net(datasets,
                   U,
                   img_w=vector_size, 
                   filter_hs=[3,4,5], 
                   hidden_units=[hidden_layer_input_size,hidden_layer_output_size], 
                   dropout_rate=[0.5],
                   shuffle_batch=True, 
                   n_epochs=25, 
                   batch_size=50,  
                   lr_decay = 0.95, 
                   conv_non_linear="relu",
                   activations=[Iden],
                   sqr_norm_lim=9,
                   non_static=True): 
    rng = np.random.RandomState(3435) #随机数生成器
    img_h = len(datasets[0][0])-1  #句子的长度 这里相当于图片的高度
    filter_w = img_w    #词向量的长度 这里相当于图片的宽度
    feature_maps = hidden_units[0] #feature_map的个数
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        #定义卷积的参数：输出feature_map的个数，输入feature_map的个数，filter的高，filter的宽
        #如filter_shapes.append((100, 1, 3, 300)) 
        #假设输入为1*64*300 （一个feature_map其szie=64*300）
        #则使用一个filter_szie=3*300的滤波器对其做卷积,输出的每一个feature_map的size=(64-3+1)*1  共100个这样的feature_map
        filter_shapes.append((feature_maps, 1, filter_h, filter_w)) 
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1)) #定义池化层的参数,卷积窗口越大，池化窗口越小
        #pool_sizes.append((img_h-filter_h+1, 1)) #定义池化层的参数,卷积窗口越大，池化窗口越小,由于输入只有1列，所以其size可以定义为(img_h-filter_h+1, 1)
    #dropout的比例为50%
    #Dropout是指在模型训练时随机让网络某些隐含层节点的权重不工作，不工作的那些节点可以暂时认为不是网络结构的一部分，但是它的权重得保留下来（只是暂时不更新而已），因为下次样本输入时它可能又得工作了
    #dropout是防止模型过拟合的一种trikc
    #batch_size 批量训练的样本数量
    #learn_decay 调整学习速率的参数
    #shuffle_batch 在机器学习中，如果训练数据之间相关性很大，可能会让结果很差（泛化能力得不到训练）。这时通常需要将训练数据打散。
    parameters = [("image shape",img_h,img_w),("filter shape",filter_shapes),("pool_sizes",pool_sizes), ("hidden_units",hidden_units),
                  ("dropout", dropout_rate), ("batch_size",batch_size),("non_static", non_static),
                    ("learn_decay",lr_decay), ("conv_non_linear", conv_non_linear), ("non_static", non_static)
                    ,("sqr_norm_lim",sqr_norm_lim),("shuffle_batch",shuffle_batch)]
    print parameters    
    
    #define model architecture
    index = T.lscalar()
    x = T.matrix('x')   
    y = T.ivector('y')
    Words = theano.shared(value = U, name = "Words") #word vector matrix
    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(img_w)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))], allow_input_downcast=True)
    #flatten将多维数组降为1维后进行reshape
    layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],1,x.shape[1],Words.shape[1]))    

    #定义一个卷积池化层====================================
    conv_layers = [] #存储所有的卷积结构
    layer1_inputs = [] #存储所有的卷积结果
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        #一个filter对应同一卷积层中的一个卷积网络
        #一个输入64*300的feature_map 进过(100, 1, 3, 300)的卷积，输出100个feature_map 其size=(64-3+1)*1
        #进过池化层pool_szie=(64-3+1)*(300-3+1) 后，得到100个1*1的输出
        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,image_shape=(batch_size, 1, img_h, img_w),
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)

    #定义MLP层=============================
    #concatenate进行连接layer1_inputs=[[o1],[o2]..]连接为[o1,o2,...]的一维向量，作为全连接层输入
    layer1_input = T.concatenate(layer1_inputs,1)
    hidden_units[0] = feature_maps*len(filter_hs)
    #带dropout的多层感知机 由隐层(全连接层)+ logRegressionLayer(softmax层)构成并输出分类结果
    #hidden_units 定义MLP层输入输出size
    classifier = MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate)
    print "===1===="
    #define parameters of the model and update functions using adadelta
    params = classifier.params     
    for conv_layer in conv_layers:
        params += conv_layer.params
    if non_static:
        #if word vectors are allowed to change, add them as model parameters
        params += [Words]
    cost = classifier.negative_log_likelihood(y) #模型的损失函数
    dropout_cost = classifier.dropout_negative_log_likelihood(y)           
    grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)#参数的更新方法
    
    #shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate 
    #extra data (at random)
    np.random.seed(3435)
    if datasets[0].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[0].shape[0] % batch_size
        train_set = np.random.permutation(datasets[0])   
        extra_data = train_set[:extra_data_num]
        new_data=np.append(datasets[0],extra_data,axis=0)
    else:
        new_data = datasets[0]
    new_data = np.random.permutation(new_data)
    n_batches = new_data.shape[0]/batch_size
    n_train_batches = int(np.round(n_batches*0.9))
    #divide train set into train/val sets 
    test_set_x = datasets[1][:,:img_h] 
    test_set_y = np.asarray(datasets[1][:,-1],"int32")
    train_set = new_data[:n_train_batches*batch_size,:]
    val_set = new_data[n_train_batches*batch_size:,:]     
    train_set_x, train_set_y = shared_dataset((train_set[:,:img_h],train_set[:,-1]))
    val_set_x, val_set_y = shared_dataset((val_set[:,:img_h],val_set[:,-1]))
    n_val_batches = n_batches - n_train_batches
    val_model = theano.function([index], classifier.errors(y),
         givens={
            x: val_set_x[index * batch_size: (index + 1) * batch_size],
             y: val_set_y[index * batch_size: (index + 1) * batch_size]},
                                allow_input_downcast=True)
    print "===2===="        
    #compile theano functions to get train/val/test errors
    test_model = theano.function([index], classifier.errors(y),
             givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                 y: train_set_y[index * batch_size: (index + 1) * batch_size]},
                                 allow_input_downcast=True)               
    train_model = theano.function([index], cost, updates=grad_updates,
          givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
              y: train_set_y[index*batch_size:(index+1)*batch_size]},
                                  allow_input_downcast = True)     
    test_pred_layers = []
    test_size = test_set_x.shape[0]
    test_layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]))
    print "===3===="
    for conv_layer in conv_layers:
        test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
        test_pred_layers.append(test_layer0_output.flatten(2))
    print "===4===="
    test_layer1_input = T.concatenate(test_pred_layers, 1)
    test_y_pred = classifier.predict(test_layer1_input)
    print "===5===="
    #test_error = T.mean(T.neq(test_y_pred, y))
    test_error = T.neq(test_y_pred, y)
    print "===6===="
    test_model_all = theano.function([x,y], test_error, allow_input_downcast = True) 
    print "===7===="
    #start training over mini-batches
    print '... training'
    epoch = 0
    best_val_perf = 0
    val_perf = 0
    test_perf = 0       
    cost_epoch = 0    

    print Words.get_value()
    while (epoch < n_epochs):
        start_time = time.time()
        epoch = epoch + 1
        if shuffle_batch: 
            for minibatch_index in np.random.permutation(range(n_train_batches)): #随机打散 每次输入的样本的顺序都不一样
                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)
        else:
            for minibatch_index in xrange(n_train_batches):
                cost_epoch = train_model(minibatch_index)  
                set_zero(zero_vec)
        train_losses = [test_model(i) for i in xrange(n_train_batches)]
        train_perf = 1 - np.mean(train_losses)
        val_losses = [val_model(i) for i in xrange(n_val_batches)]
        val_perf = 1- np.mean(val_losses)                        
        print('epoch: %i, training time: %.2f secs, train perf: %.2f %%, val perf: %.2f %%' % (epoch, time.time()-start_time, train_perf * 100., val_perf*100.))
        if val_perf >= best_val_perf:
            best_val_perf = val_perf
            test_loss = T.mean(test_model_all(test_set_x,test_set_y))      
            test_perf = 1- test_loss
    print "end"
    print Words.get_value()
    return test_perf,Words.get_value()

def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')
        
def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    #求偏倒数
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param      
    return updates 

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[str(theano.config.floatX)](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[str(theano.config.floatX)](variable)
    return theano.tensor.cast(variable, theano.config.floatX)
    
def safe_update(dict_to, dict_from):
    """
    re-make update dictionary for safe updating
    """
    for key, val in dict(dict_from).iteritems():
        if key in dict_to:
            raise KeyError(key)
        dict_to[key] = val
    return dict_to
    
def get_idx_from_sent(sent, word_idx_map, max_l=51, k=vector_size, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    #一个训练的一个输入 形式为[0,0,0,0,x11,x12,,,,0,0,0] 向量长度为max_l+2*filter_h-2
    return x 

def make_idx_data_cv(revs, word_idx_map, cv, max_l=51, k=vector_size, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    for rev in revs:
        #rev["text"]原始句子文本 word_idx_map：词向量矩阵索引，最大句子长度，词向量维度，滤波器大小
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)
        sent.append(rev["y"])
        if rev["split"]==cv:            
            test.append(sent)        
        else: 
            #一个训练样本包括一段输入向量  和 一个lable
            train.append(sent)    
    train = np.array(train,dtype="int")
    test = np.array(test,dtype="int")
    return [train, test]     
  
   
if __name__=="__main__":
    print "loading data...",
    x = cPickle.load(open("mr.p","rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4] # 读取出预处理后的数据
    print "data loaded!"
    mode= sys.argv[1]
    word_vectors = sys.argv[2]    
    if mode=="-nonstatic":
        print "model architecture: CNN-non-static"
        non_static=True
    elif mode=="-static":
        print "model architecture: CNN-static"
        non_static=False
    execfile("conv_net_classes.py")    
    if word_vectors=="-rand":
        print "using: random vectors"
        U = W2 #使用随机初始化的词向量
    elif word_vectors=="-word2vec":
        print "using: word2vec vectors"
        U = W
    results = []
    r = range(0,10)    
    for i in r:
        #max_l 最大的句子长度
        datasets = make_idx_data_cv(revs, word_idx_map, i, max_l=100,k=vector_size, filter_h=5)
        perf,Words = train_conv_net(datasets,
                              U,
                              lr_decay=0.95,
                              filter_hs=[3,4,5],
                              conv_non_linear="relu",
                              hidden_units=[hidden_layer_input_size,hidden_layer_output_size], 
                              shuffle_batch=True, 
                              n_epochs=25, 
                              sqr_norm_lim=9,
                              non_static=non_static,
                              batch_size=50,
                              dropout_rate=[0.5])
        print "cv: " + str(i) + ", perf: " + str(perf)
        results.append(perf)
        filename = "CNN_result_"+str(i)
        cPickle.dump([Words, word_idx_map], open(filename, "wb"))
    print str(np.mean(results))
