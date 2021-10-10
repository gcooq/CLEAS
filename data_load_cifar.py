from __future__ import division
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import  numpy as np
(x_train, y_train), (x_test, y_test)=keras.datasets.cifar100.load_data(label_mode='fine')
print x_train.shape
print y_train.shape
num_classes=len(np.unique(y_train))
x_train=x_train.astype('float32')/255
x_test=x_test.astype('float32')/255
print 32*32*3
x_train=np.reshape(x_train,[x_train.shape[0],x_train.shape[1]*x_train.shape[2]*x_train.shape[3]])
x_test=np.reshape(x_test,[x_test.shape[0],x_test.shape[1]*x_test.shape[2]*x_test.shape[3]])
print x_train.shape,x_test.shape
classes=np.unique(y_train)
#print ('classes',classes)
Task_train_x={}
Task_train_y={}
Task_test_x={}
Task_test_y={}
def get_hot(index):
    x=[0.0]*10
    x[index]=1.0
    return x
Task_vali_x={}
Task_vali_y={}
for index in classes:
    for i in range(len(x_train)):
        if y_train[i]==index:
            Task_train_x.setdefault(index,[]).append(x_train[i])
            Task_train_y.setdefault(index,[]).append(get_hot(index%10))
            #print index%10
    for j in range(len(x_test)):
        if y_test[j]==index:
            Task_test_x.setdefault(index,[]).append(x_test[j])
            Task_test_y.setdefault(index, []).append(get_hot(index%10))
#INDEX=[x for x in range(0,500)]
#print(INDEX)
import random
#random.seed(0)#fixed seed
for label in Task_train_x.keys():
    x=Task_train_x[label]
    y=Task_train_y[label]
    #print('x',len(x))
    index=random.sample(range(0,len(x)),100)
    index=sorted(index)
    for i in index:
        Task_vali_x.setdefault(label,[]).append(x[i])
        Task_vali_y.setdefault(label,[]).append(y[i])
#print(Task_vali_x[0])
#fig=plt.figure(figsize=(20,5))
#for i in range(36):
#    ax=fig.add_subplot(3,12,i+1,xticks=[],yticks=[])
#    ax.imshow(np.squeeze(Task_train_x[4][i]))
#plt.show()

def next_batch(train_data,train_target,batch_szie):
    index=[i for i in range(0,len(train_target))]
    np.random.shuffle(index)
    batch_data=[]
    batch_target=[]
    for i in range(0,batch_szie):
        batch_data.append(train_data[index[i]])
        batch_target.append(train_target[index[i]])
    return batch_data,batch_target



