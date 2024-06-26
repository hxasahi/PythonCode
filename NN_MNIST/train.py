import numpy as np
import math
from pathlib import Path
import struct
import matplotlib.pyplot as plt
import copy
np.set_printoptions(threshold=np.inf)

dataset_path = Path('./dataset')
train_img_path=dataset_path/'train-images-idx3-ubyte'
train_lab_path=dataset_path/'train-labels-idx1-ubyte'
test_img_path=dataset_path/'t10k-images-idx3-ubyte'
test_lab_path=dataset_path/'t10k-labels-idx1-ubyte'

train_num=50000
valid_num=10000
test_num=10000
batch_size = 100

#定义输入输出维度
dimensions=[28*28,10]
#定义参数
distribution=[
    {'b':[0,0]},
    {'b':[0,0],'w':[-math.sqrt(6/(dimensions[0]+dimensions[1])),math.sqrt(6/(dimensions[0]+dimensions[1]))]
     }]

#重定义需要用到的激活函数
def tanh(x):
    return np.tanh(x)
def softmax(x):
    exp=np.exp(x-x.max())
    return exp/exp.sum()

activation=[tanh,softmax]

#初始化参数
def init_parameter_b(layer):
    dist=distribution[layer]['b']
    return np.random.rand(dimensions[layer])*(dist[1]-dist[0])+dist[0]
def init_parameter_w(layer):
    dist=distribution[layer]['w']
    return np.random.rand(dimensions[layer-1],dimensions[layer])*(dist[1]-dist[0])+dist[0]
def init_parameters():
    parameter=[]
    for i in range(len(dimensions)):
        layer_parameter={}
        for j in distribution[i].keys():
            if j == 'b' :
                layer_parameter['b']=init_parameter_b(i)
                continue
            if j == 'w' :
                layer_parameter['w']=init_parameter_w(i)
                continue
        parameter.append(layer_parameter)
    return parameter

parameters=init_parameters()


def predict(img,parameters):
    l0_in = img + parameters[0]['b']
    l0_out = activation[0](l0_in)
    l1_in=np.dot(l0_out,parameters[1]['w'])+parameters[1]['b']
    l1_out=activation[1](l1_in)
    return l1_out

#用于训练的图像
with open(train_img_path,'rb') as f:
    struct.unpack('>4i',f.read(16))
    temp_img=np.fromfile(f,dtype = np.uint8).reshape(-1,28*28)
    train_img=temp_img[:train_num]
    valid_img=temp_img[train_num:]

#用于训练的标签
with open(train_lab_path,'rb') as f:
    struct.unpack('>2i',f.read(8))
    temp_lab=np.fromfile(f,dtype = np.uint8)
    train_lab=temp_lab[:train_num]
    valid_lab=temp_lab[train_num:]

#用于测试的图像
with open(test_img_path,'rb') as f:
    struct.unpack('>4i',f.read(16))
    test_img=np.fromfile(f,dtype = np.uint8).reshape(-1,28*28)

#用于测试的标签
with open(test_lab_path,'rb') as f:
    struct.unpack('>2i',f.read(8))
    test_lab=np.fromfile(f,dtype = np.uint8)


def show_train(index):
    plt.imshow(train_img[index].reshape(28,28),cmap='gray')    
    print('label : {}'.format(train_lab[index]))
    plt.show()
def show_valid(index):
    plt.imshow(valid_img[index].reshape(28,28),cmap='gray')    
    print('label : {}'.format(valid_lab[index]))
    plt.show()
def show_test(index):
    plt.imshow(test_img[index].reshape(28,28),cmap='gray')    
    print('label : {}'.format(test_lab[index]))
    plt.show()

#求导
def d_softmax(data):
    sm=softmax(data)
    return np.diag(sm)-np.outer(sm,sm)

def d_tanh(data):
    return (1/(np.cosh(data))**2)

differential={softmax:d_softmax,tanh:d_tanh}
onehot=np.identity(dimensions[-1])

#损失函数
def sqr_loss(img,lab,parameters):
    y_pred = predict(img,parameters)
    y = onehot[lab]
    diff = y - y_pred
    return np.dot(diff,diff)

def grad_parameters(img,lab,parameters):
    l0_in = img + parameters[0]['b']
    l0_out = activation[0](l0_in)
    l1_in=np.dot(l0_out,parameters[1]['w'])+parameters[1]['b']
    l1_out=activation[1](l1_in)

    diff = onehot[lab]-l1_out
    act1 = np.dot(differential[activation[1]](l1_in),diff)
    
    grad_b1 = -2*act1
    grad_w1 = -2*np.outer(l0_out,act1)
    grad_b0 = -2*differential[activation[0]](l0_in)*np.dot(parameters[1]['w'],act1)
    return {'w1':grad_w1,'b1':grad_b1,'b0':grad_b0}

#计算损失
def valid_loss(parameters):
    loss_accu=0
    for img_i in range(valid_num):
        loss_accu+=sqr_loss(valid_img[img_i],valid_lab[img_i],parameters)
    return loss_accu

#计算准确率
def valid_accuracy(parameters):
    correct=[predict(valid_img[img_i],parameters).argmax()==valid_lab[img_i] for img_i in range(valid_num)]
    print('validation accuracy {} '.format(correct.count(True)/len(correct)))

def train_batch(current_batch,parameters):
    grad_accu=grad_parameters(train_img[current_batch*batch_size+0],train_lab[current_batch*batch_size+0],parameters)
    for img_i in range(1,batch_size):
        grad_temp = grad_parameters(train_img[current_batch*batch_size+img_i],train_lab[current_batch*batch_size+img_i],parameters)
        for key in grad_accu.keys():
            grad_accu[key]+=grad_temp[key]
    for key in grad_accu.keys():
        grad_accu[key]/=batch_size
    return grad_accu

def combine_parameters(parameters,grad,learn_rate):
    parameters_temp = copy.deepcopy(parameters)
    parameters_temp[0]['b']-=learn_rate*grad['b0']
    parameters_temp[1]['b']-=learn_rate*grad['b1']
    parameters_temp[1]['w']-=learn_rate*grad['w1']
    return parameters_temp

def learn_self(learn_rate):
    for i in range(train_num//batch_size):
        if i%100==99:
            print('running batch {}/{}'.format(i+1,train_num//batch_size))
        global parameters
        grad_temp = train_batch(i,parameters)
        parameters = combine_parameters(parameters,grad_temp,learn_rate)
    valid_accuracy(parameters)
    print('learn_rate = {}'.format(learn_rate))

valid_accuracy(parameters)
learn_self(0.5)
print(valid_loss(parameters))
parameters_file = open('./parameters.txt','w')
parameters_file.write('parameters=' + str(parameters))
parameters_file.close()
