# -*- coding: UTF-8 -*-
import numpy as np
from numpy import *
import random
import re
import os
import time
def createVocabList(dataSet):
    vocabSet = set()                      #创建一个空的不重复列表
    for document in dataSet:
        if document!='':
          vocabSet.add(document)
    return list(vocabSet)

#读取训练集的非垃圾邮件，并生成该邮件的词汇表
def readAhemail(n,filelist):
    reader = []
    list1 = []
    #filelist = os.listdir('D:\\AI\\email\\ham')
    with open("D:\\AI\\email\\ham\\"+filelist[n-1], 'r')as f:
        reader = str(f.read())
    s = re.sub('[^A-Za-z]', ' ', reader)  # 只字母,若要保留数字，正则表达式加上0-9.
    list1 = s.split(' ')  # 拆分
    list1 = [x.lower() for x in list1]  # 大写变小写
    sss = createVocabList(list1)
    return sss

#读取训练集的垃圾邮件，并生成该邮件的词汇表
def readAsemail(n,filelist):
    reader = []
    list1 = []
    #filelist = os.listdir('D:\\AI\\email\\spam')
    with open("D:\\AI\\email\\spam\\"+filelist[n-1], 'r')as f:
        reader = str(f.read())
    s = re.sub('[^A-Za-z]', ' ', reader)  # 只字母
    list1 = s.split(' ')  # 拆分
    list1 = [x.lower() for x in list1]  # 大写变小写
    sss = createVocabList(list1)
    return sss

#创建n封非垃圾邮件和m封垃圾垃圾的总词汇表
def TotalVocabList(A,B,filelist0,filelist1):
    Total=set()
    for i in A:#非垃圾邮件
        s1=readAhemail(i,filelist0)
        for j in s1:
            Total.add(j)
    for i in B:#垃圾邮件
        s2=readAsemail(i,filelist1)
        for j in s2:
            Total.add(j)
    return list(Total)

#将一封邮件的词汇表转换成01的词向量
def setWordsVec(vocabList, inputSet):
    Vec = [0] * len(vocabList)									#创建一个其中所含元素都为0的向量
    for word in inputSet:												#遍历每个词条
        if word in vocabList:											#如果词条存在于词汇表中，则置1
             Vec[vocabList.index(word)] = 1
    return Vec													#返回文档向量


#垃圾邮件求各种概率
def getP(trainMatrix,trainLable):
    #trainMatrix是一个列表的列表，一个列表为一个样本的词向量（词向量长度为总词汇表的长度）
    #trainLable是一个01向量，对应着样本的标签

    numTrain = len(trainMatrix)							#计算训练矩阵的样本数
    numWords = len(trainMatrix[0])							#计算每个样本的词数

    #求垃圾邮件的先验概率P（垃圾）
    #标签向量元素为1表示是垃圾邮件，求和就是垃圾邮件的数量
    Pspam = sum(trainLable)/float(numTrain)

    #求条件概率
    p1NumVec = zeros(numWords)      #记录垃圾邮件频数向量
    p0NumVec = zeros(numWords)      #记录非垃圾邮件频数向量
    p1Sum = 0.0                        #记录垃圾邮件总词数
    p0Sum = 0.0                        #记录非垃圾邮件总词数
    for i in range(numTrain):
        if trainLable[i] == 1:
            p1NumVec += trainMatrix[i]                 #向量和向量相加，最终所得的向量的元素对应各特征在垃圾邮件样本中占的频数
            p1Sum += sum(trainMatrix[i])          #每次都加上该样本的总词数，最终所得为垃圾邮件样本的总词数
        else:
            p0NumVec += trainMatrix[i]               #同上统计非垃圾邮件
            p0Sum += sum(trainMatrix[i])
    p1Vect = p1NumVec/p1Sum						#垃圾邮件频数向量除与垃圾邮件总词数得到垃圾邮件条件概率向量P(词1|垃圾),P(词2|垃圾),P(词3|垃圾)···
    p0Vect = p0NumVec/p0Sum                     #非垃圾邮件频数向量除与非垃圾邮件总词数得到非垃圾邮件条件概率向量P(词1|非垃圾),P(词2|非垃圾),P(词3|非垃圾)···
    return [p1Vect,p0Vect,Pspam]							#返回垃圾邮件条件概率向量，非垃圾邮件条件概率向量，先验概率P（垃圾）

#测试分类
def classify(VecTest, p1Vec, p0Vec, PClass):
    p1 = sum(VecTest * p1Vec) * PClass    			#是垃圾邮件的概率
    p0 = sum(VecTest * p0Vec) * (1.0 - PClass) #是非垃圾邮件的概率
    if p1 > p0:
        return 1
    else:
        return 0

if __name__ == '__main__':
   t1=time.time()
   filelist0 = os.listdir('D:\\AI\\email\\ham')
   filelist1 = os.listdir('D:\\AI\\email\\spam')
   long = len(filelist0)  # 两个文件夹中的非垃圾邮件数等于垃圾邮件数
   ratio = eval(input("输入训练集所占比例："))
   TrainNum = int(long * ratio)  # 分别从非垃圾邮件和垃圾邮件中取的训练集个数
   Index0 = random.sample(range(1, long + 1), TrainNum)  # 随机抽取TrainNum个样本
   Index1 = random.sample(range(1, long + 1), TrainNum)
   V = TotalVocabList(Index0, Index1, filelist0, filelist1)  # 生成总词汇表
   train = []  # 训练矩阵
   lable = []  # 训练标签
   # 0样本词向量加入训练矩阵
   for i in Index0:  # 根据随机抽取的非垃圾邮件下标选取非垃圾邮件
       a = readAhemail(i, filelist0)  # 根据非垃圾邮件名列表读取非垃圾邮件
       a = setWordsVec(V, a)  # 对照总词汇表生成词向量
       train.append(a)  # 将生成的词汇量加进训练集
       lable.append(0)  # 同时为词向量加上标签
   # 1样本词向量加入训练矩阵
   for i in Index1:  # 根据随机抽取的垃圾邮件下标选取垃圾邮件
       a = readAsemail(i, filelist1)  # 根据垃圾邮件名列表读取非垃圾邮件
       a = setWordsVec(V, a)  # 对照总词汇表生成词向量
       train.append(a)  # 将生成的词汇量加进训练集
       lable.append(1)  # 同时为词向量加上标签

   TestIndex0 = []  # 0测试集下标
   TestIndex1 = []  # 1测试集下标
   for i in range(1,long+1):
       if i not in Index0:
           TestIndex0.append(i)
       if i not in Index1:
           TestIndex1.append(i)
   #测试集生成词向量
   Test0=[]
   Test1=[]
   for i in TestIndex0:
       Test0.append(setWordsVec(V,readAhemail(i,filelist0)))
   for i in TestIndex1:
       Test1.append(setWordsVec(V, readAsemail(i,filelist1)))

   #计算判断正确率
   total=len(Test0)+len(Test1)
   num0=0
   num1=0
   for i in Test0:
       if classify(i,*getP(train,lable))==0:
           num0+=1
   for i in Test1:
       if classify(i,*getP(train,lable))==1:
           num1+=1
   accuracy = (num0 + num1) / total
   '''
   print("训练集下标")
   print(Index0)
   print(Index1)
   print("测试下标")
   print(TestIndex0)
   print(TestIndex1)
   '''
   print("测试集邮件数为{}".format(total))
   print("非垃圾邮件分类正确数为{}".format(num0))
   print("垃圾邮件分类正确数为{}".format(num1))
   print("分类正确率为:{}%".format(accuracy*100))
   t2=time.time()
   print('用时：',t2-t1)
