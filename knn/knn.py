# import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
std = StandardScaler()
X_train = std.fit_transform(X_train)
X_test = std.transform(X_test)
#print(X_train[0:10])
#print(y_train[0:10])
print(X_test[0:10])
ls_pre=[4,4,4,4,4,4,4,4,4,4]
#print(y_test[0:10]) 真实的类别
pos_pre=0
for i in X_test[0:10]:
    #print(i)    需要预测的值
    ls_dis=[100,101,102,103,104]
    ls_res=[4,4,4,4,4]
    pos=0
    for item in X_train:
        distance=pow((i[0]-item[0])*(i[0]-item[0])+(i[1]-item[1])*(i[1]-item[1])+(i[2]-item[2])*(i[2]-item[2])+(i[3]-item[3])*(i[3]-item[3]),0.5)
        #print(distance) 算的距离
        if(distance<max(ls_dis)):
            get_pos = ls_dis.index(max(ls_dis))
            ls_dis[get_pos]=distance
            ls_res[get_pos]=y_train[pos]
        #print(ls_dis)  内部的收敛过程
        #print(ls_res)
        pos = pos+1
    #计算频次
    most_counterNum = Counter(ls_res).most_common(1)
    #print(most_counterNum[0][0])  最终预测结果
    ls_pre[pos_pre]=most_counterNum[0][0]
    pos_pre=1+pos_pre
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
# 预测
y_predict = knn.predict(X_test[0:10])#预测test的前10个
print("实际结果", y_test[0:10])
print("调包预测结果", y_predict)
print("手算预测结果", ls_pre)