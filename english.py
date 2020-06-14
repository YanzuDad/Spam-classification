# -*- coding: UTF-8 -*-
import numpy as np
from numpy import *
import random
import re
import os
def createVocabList(dataSet):
    vocabSet = set()                      #创建一个空的不重复列表
    for document in dataSet:
        if document!='':
          vocabSet.add(document)
    return list(vocabSet)

#读取训练集的非垃圾邮件，并生成该邮件的词汇表
def readAhemail(n,filelist,pathHam):
    reader = []
    list1 = []
    with open(pathHam+'\\'+filelist[n-1], 'r')as f:
        reader = str(f.read())
    s = re.sub('[^A-Za-z]', ' ', reader)  # 只字母,若要保留数字，正则表达式加上0-9.
    list1 = s.split(' ')  # 拆分
    list1 = [x.lower() for x in list1]  # 大写变小写
    sss = createVocabList(list1)
    return sss

#读取训练集的垃圾邮件，并生成该邮件的词汇表
def readAsemail(n,filelist,pathSpam):
    reader = []
    list1 = []
    with open(pathSpam+'\\'+filelist[n-1], 'r')as f:
        reader = str(f.read())
    s = re.sub('[^A-Za-z]', ' ', reader)  # 只字母
    list1 = s.split(' ')  # 拆分
    list1 = [x.lower() for x in list1]  # 大写变小写
    sss = createVocabList(list1)
    return sss

#创建n封非垃圾邮件和m封垃圾垃圾的总词汇表
def TotalVocabList(A,B,filelist0,filelist1,pathHam,pathSpam):
    Total=set()
    for i in A:#非垃圾邮件
        s1=readAhemail(i,filelist0,pathHam)
        for j in s1:
            Total.add(j)
    for i in B:#垃圾邮件
        s2=readAsemail(i,filelist1,pathSpam)
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


def create_file(pathHam,pathSpam):#将提取的特征和标签存进文件里
   filelist0 = os.listdir(pathHam)#获得非垃圾邮件文件夹的文件名列表
   filelist1 = os.listdir(pathSpam)#获得垃圾邮件文件夹的文件名列表
   long0 = len(filelist0)#非垃圾邮件文件夹文件数
   long1 = len(filelist1)#垃圾邮件文件夹文件数
   Index0 = [x for x in range(long0)]#非垃圾邮件下标
   Index1 = [x for x in range(long1)]#垃圾邮件下标
   V = TotalVocabList(Index0, Index1, filelist0, filelist1,pathHam,pathSpam)  # 生成总词汇表
   x = []#样本词向量矩阵
   y = []#标签列表
   # 0样本词向量加入样本矩阵
   for i in Index0:
       a = readAhemail(i, filelist0,pathHam)#根据非垃圾邮件文件名列表读取一封非垃圾邮件
       a = setWordsVec(V, a)#根据总词汇表生成词向量
       x.append(a)#加入词向量矩阵
       y.append(0)#加上非垃圾邮件标签
   # 1样本词向量加入样本矩阵
   for i in Index1:
       a = readAsemail(i, filelist1,pathSpam)#根据垃圾邮件文件名列表读取一封垃圾邮件
       a = setWordsVec(V, a)#根据总词汇表生成词向量
       x.append(a)#加入词向量矩阵
       y.append(1)#加上垃圾邮件标签
   #转化为numpy数组方便处理
   x = np.array(x)
   y = np.array(y)
   np.savetxt('特征矩阵',x)
   np.savetxt('标签列表',y)
   # # 随机打乱所有样本
   # index = np.arange(len(y))#根据样本数生成按顺序的下标数组
   # np.random.shuffle(index)#打乱下标顺序
   # #把样本词向量矩阵及标签一起打乱
   # x = x[index]
   # y = y[index]
   # return x,y

def getEnglishData():#从文件中提取特征
    x=np.loadtxt('特征矩阵')
    return x
def getEnglishLabel():#从文件中提取标签
    y=np.loadtxt('标签列表')
    return y

