import english
from sklearn.naive_bayes import BernoulliNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import time
import chinese
import os
def BernoulliNBClassify(x_train, x_test, y_train, y_test):
    print('邮件数量为:',len(y_train)+len(y_test))
    start = time.time()
    clf = BernoulliNB()  # 伯努利贝叶斯
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("伯努利贝叶斯结果：")
    print('准确率：', metrics.accuracy_score(y_test, y_pred))
    end = time.time()
    print('Running time: %s Seconds' % (end - start))

def LinearSVCClassify(x_train, x_test, y_train, y_test):
    print('邮件数量为:', len(y_train) + len(y_test))
    start = time.time()
    clf = svm.LinearSVC()  # 线性分类支持向量机
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("线性分类支持向量机结果：")
    print('准确率：', metrics.accuracy_score(y_test, y_pred))
    end = time.time()
    print('Running time: %s Seconds' % (end - start))

def KNeighborsClassify(x_train, x_test, y_train, y_test):
    print('邮件数量为:', len(y_train) + len(y_test))
    start = time.time()
    clf = KNeighborsClassifier(n_neighbors=4)  # k近邻
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("k近邻结果：")
    print('准确率：', metrics.accuracy_score(y_test, y_pred))
    end = time.time()
    print('Running time: %s Seconds' % (end - start))

if __name__=='__main__':
    #英文
    # english.create_file('D:\\AI\\email\\ham', 'D:\\AI\\email\\spam')#创建完一次就能获得特征矩阵，换数据集才需要重新创建
    x=english.getEnglishData()
    y=english.getEnglishLabel()
    # 随机打乱所有样本
    index = np.arange(len(y))  # 根据样本数生成按顺序的下标数组
    np.random.shuffle(index)  # 打乱下标顺序
    # 把样本词向量矩阵及标签一起打乱
    x = x[index]
    y = y[index]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)#训练集：测试集=8：2

    BernoulliNBClassify(x_train, x_test, y_train, y_test)
    LinearSVCClassify(x_train, x_test, y_train, y_test)
    KNeighborsClassify(x_train, x_test, y_train, y_test)

    #中文
    np.random.seed(1)
    # chinese.create_file('D:\\AI\\svm\\emailTest\\data',#获得数据集数据路径
    #                     'D:\\AI\\svm\\emailTest\\index',#获得数据集标签路径
    #                     'D:\\AI\\svm\\emailTest\\all_email.txt',#保存特征的路径
    #                     'D:\\AI\\svm\\emailTest\\label.txt'#保存标签的路径
    #                     )#创建完一次就够了
    email_file_name = 'D:\\AI\\svm\\emailTest\\all_email.txt'
    label_file_name = 'D:\\AI\\svm\\emailTest\\label.txt'
    x, vectoring = chinese.get_data_tf_idf(email_file_name)
    y = chinese.get_label_list(label_file_name)
    index = np.arange(len(y))
    np.random.shuffle(index)
    x = x[index]
    y = np.array(y)
    y = y[index]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    BernoulliNBClassify(x_train, x_test, y_train, y_test)
    LinearSVCClassify(x_train, x_test, y_train, y_test)
    KNeighborsClassify(x_train, x_test, y_train, y_test)
