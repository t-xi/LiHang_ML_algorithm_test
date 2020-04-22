import numpy as np
import pandas as pd
import time
import math
import random

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

def logistics_function_predict1(w,x):
    wx=0.0
    for j in range(len(w)):
        wx +=w[j]*x[j]
    predict1=math.exp(wx)/(1+math.exp(wx))
    return predict1

def train(train_features,train_labels):
    w=[0.0]*(1+len(train_features[0]))
    learning_rate =0.00001
    correct_count = 0
    time = 0
    max_iteration = 5000  # 4*len(lables)
    #gredent_model=1000
    #随机梯度下降迭代
    while(time<max_iteration):
        #随机选取样本
        index = np.random.randint(0,len(train_labels))
        x=list(train_features[index])
        x.append(1.0)
        y=train_labels[index]
        #随机梯度下降,最大似然估计
        predict1 = logistics_function_predict1(w, x)
        predict2 = 1 - predict1
        if predict1 > predict2:
            y_predict=1
        else:
            y_predict=0

        if y_predict==y:
            correct_count+=1
            if correct_count>max_iteration:
                break
            continue
        else:
            for j in range(len(w)):
                w[j]+=learning_rate*(y-logistics_function_predict1(w,x))*x[j]
                time+=1
                #梯度的模
                #gredent_model+=((y-logistics_function_predict1(w,x))*x[j])**2

    return w

def predict(w,test_features):
    labels=[]
    for feature in test_features:
        x=list(feature)
        x.append(1.0)
        predict1=logistics_function_predict1(w,x)
        predict2=1-predict1
        if predict1>predict2:
            labels.append(1)
        else:
            labels.append(0)

    return labels



if __name__=="__main__":

    #读入数据
    time_1=time.time()
    raw_data=pd.read_csv('diabetes.csv.gz', header=0)
    data=raw_data.values
    features=data[:,:-1]
    labels=data[:,[-1]]

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.33, random_state=23323)
    time_2=time.time()
    print("读取数据用时：",time_2-time_1,"s","\n")
    print(train_features.shape, test_features.shape, train_labels.shape, test_labels.shape)

    #训练数据
    w=train(train_features,train_labels)
    time_3=time.time()
    print("训练用时：",time_3-time_2,"s","\n")
    #检验模型预测准确率

    test_predict=predict(w,test_features)
    time_4=time.time()
    print("预测时间：",time_4 - time_3 , 's', '\n')
    scores=accuracy_score(test_labels,test_predict)
    print("训练精度：", scores, '%', '\n')