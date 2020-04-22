import numpy as np
import pandas as pd
import random
import time
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
#训练数据方法
def train(features,lables):
    learning_rate =0.00001
    max_iteration = 5000#4*len(lables)
    correct_count=0
    #权重初始化
    w=[0.0]*(len(features[0])+1)
    #构造感知器
    while(1):
        index=random.randint(0,len(lables)-1)
        x=list(features[index])
        x.append(1.0)
        y=2*lables[index]-1
        wx = sum([w[j] * x[j] for j in range(len(w))])
        #随机梯度下降
        if wx*y>0:
            correct_count += 1
            if correct_count > max_iteration:
                break
            continue
        else:
            for i in range(len(w)):
                w[i]+=learning_rate*(x[i]*y)


    return w

def predict(w,test_features):
    lables = []
    for test_feature in test_features:
        wx = 0.0
        x = list(test_feature)
        x.append(1.0)
        #wx = sum([w[j] * x[j] for j in range(len(w))])
        for i in range(len(x)):
           wx += w[i]*x[i]

        if wx > 0:
            lables.append(1)
        else:
            lables.append(0)

    return lables


if __name__=='__main__':
    #加载数据集
    time_1=time.time()
    raw_data = pd.read_csv('diabetes.csv.gz', header=0)
    data = raw_data.values

    features = data[:, :-1]
    lables = data[:, [-1]]

    ## 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features , test_features , train_lables , test_lables=train_test_split(features,
                    lables , test_size=0.33 , random_state=23323)
    print(train_features.shape,test_features.shape,train_lables.shape,test_lables.shape)
    time_2=time.time()
    print("加载数据用时：",time_2-time_1,'second', '\n')
    #模型训练
    w=train(train_features,train_lables)
    time_3=time.time()
    print("训练用时：",time_3-time_2,'second', '\n')

    #测试集预测
    test_predict = predict(w, test_features)
    time_4 = time.time()
    print("预测时间：",time_4 - time_3 , 'second', '\n')
    #训练精度
    scoer=accuracy_score(test_lables , test_predict)
    print("训练精度：",scoer,'%', '\n')
