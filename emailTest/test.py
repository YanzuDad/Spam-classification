'''每一个邮件样本，除了邮件文本外，还包含其他信息，如发件人邮箱、收件人邮箱等。
因为我是想把垃圾邮件分类简单地作为一个文本分类任务来解决，所以这里就忽略了这些
信息。用递归的方法读取所有目录里的邮件样本，用 jieba 分好词后写入到一个文本中，
一行文本代表一个邮件样本：


'''

import re
import jieba
import codecs
import os 
# 去掉非中文字符
def clean_str(string):
    #加r防止转义，\u‘开头就基本表明是跟unicode编码相关的，
    #“\u”后的16进制字符串是相应汉字的utf-16编码
    #4E00～9FFF：中日韩认同表意文字区，总计收容20,902个中日韩汉字。
    #\s匹配任何空白字符，包括空格、制表符、换页符等等。等价于 [ \f\n\r\t\v]。
    #注意 Unicode 正则表达式会匹配全角空格符。^表示非

    string = re.sub(r"[^\u4e00-\u9fff]", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    # strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列
    return string.strip()

def get_data_in_a_file(original_path, save_path='all_email.txt'):
    #os.listdir()用于返回一个由文件名和目录名组成的列表
    files = os.listdir(original_path)
    for file in files:
        #os.path.isdir()用于判断对象是否为一个目录
        if os.path.isdir(original_path + '/' + file):
                get_data_in_a_file(original_path + '/' + file, save_path=\
                                   save_path)
        else:
            email = ''
            # 注意要用 'ignore'，不然会报错
            #codecs.open(filepath,method,encoding),method--打开方式，
            #r为读，w为写，rw为读写,encoding--文件的编码，中文文件使用utf-8
            f = codecs.open(original_path + '/' + file, 'r', 'gbk',errors=\
                            'ignore')
            # lines = f.readlines()
            for line in f:
                line = clean_str(line)
                email += line
            f.close()
            #'a'打开一个文件用于追加。如果该文件已存在，文件指针将会放在文件
            #的结尾。也就是说，新的内容将会被写入到已有内容之后。如果该文件不
            #存在，创建新文件进行写入。
            """
            发现在递归过程中使用 'a' 模式一个个写入文件比 在递归完后一次性用
            'w' 模式写入文件快很多
            """
            f = open(save_path, 'a', encoding='utf8')
            email = [word for word in jieba.cut(email) if word.strip() != '']
            f.write(' '.join(email) + '\n')

'''
jieba.cut 方法接受三个输入参数: 需要分词的字符串；
cut_all 参数用来控制是否采用
全模式；
HMM 参数用来控制是否使用 HMM 模型
jieba.cut_for_search 方法接受两个参数：需要分词的字符串；是否使用 HMM 模型。
该方法适合用于搜索引擎构建倒排索引的分词，粒度比较细待分词的字符串可以是
unicode 或 UTF-8 字符串、GBK 字符串。注意：不建议直接输入
GBK 字符串，可能无法预料地错误解码成 UTF-8
jieba.cut 以及 jieba.cut_for_search 返回的结构都是一个可迭代的 generator，
可以使用 for 循环来获得分词后得到的每一个词语(unicode)，或者用
jieba.lcut 以及 jieba.lcut_for_search 直接返回 list
jieba.Tokenizer(dictionary=DEFAULT_DICT) 新建自定义分词器，可用于同时使用
不同词典。jieba.dt 为默认分词器，所有全局分词相关函数都是该分词器的映射。
'''
print('Storing emails in a file ...')
get_data_in_a_file('C:\\Users\\zzp\\Desktop\\emailTest\\data', save_path=\
                   'C:\\Users\\zzp\\Desktop\\emailTest\\all_email.txt')
print('Store emails finished !')


#然后将样本标签写入单独的文件中，0 代表垃圾邮件，1 代表非垃圾邮件。代码如下：
def get_label_in_a_file(original_path, save_path='all_email.txt'):
    f = open(original_path, 'r')
    label_list = []
    for line in f:
        # spam
        if line[0] == 's':
            label_list.append('0')
        # ham
        elif line[0] == 'h':
            label_list.append('1')

    f = open(save_path, 'w', encoding='utf8')
    f.write('\n'.join(label_list))
    f.close()

print('Storing labels in a file ...')
get_label_in_a_file('C:\\Users\\zzp\\Desktop\\emailTest\\index', \
                    save_path='C:\\Users\\zzp\\Desktop\\emailTest\\label.txt')
print('Store labels finished !')

def get_label_list(label_file_name):
    f = open(label_file_name, 'r')
    label_list = []
    for line in f:
        if line[0]=='0':
            label_list.append('0')
        else:
            label_list.append('1')

    f.close()
    return label_list

import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

def tokenizer_jieba(line):
    # 结巴分词
    return [li for li in jieba.cut(line) if li.strip() != '']

def tokenizer_space(line):
    # 按空格分词
    return [li for li in line.split() if li.strip() != '']

def get_data_tf_idf(email_file_name):
    # 邮件样本已经分好了词，词之间用空格隔开，所以 tokenizer=tokenizer_space
    vectoring = TfidfVectorizer(input='content', tokenizer=tokenizer_space, \
                                analyzer='word')
    content = open(email_file_name, 'r', encoding='utf8').readlines()
    x = vectoring.fit_transform(content)
    return x, vectoring


from sklearn.linear_model import LogisticRegression
from sklearn import svm, ensemble, naive_bayes
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np

if __name__ == "__main__":
    np.random.seed(1)
    email_file_name = 'C:\\Users\\zzp\\Desktop\\emailTest\\all_email.txt'
    label_file_name = 'C:\\Users\\zzp\\Desktop\\emailTest\\label.txt'
    x, vectoring = get_data_tf_idf(email_file_name)
    y = get_label_list(label_file_name)

    #x为稀疏矩阵,
    #shape()函数为读取矩阵的维度，没有参数则输出全部维度大小，反之如shape[0]
    #则输出第一维度长度
    #对矩阵使用len()得到结果为矩阵行数
    #print('x.shape : ', x.shape)
    #x.shape :  (64622, 159590)
    #print('y.shape : ', y.shape)
    
    # 随机打乱所有样本
    index = np.arange(len(y))  
    np.random.shuffle(index)
    x = x[index]
    y=np.array(y)
    y = y[index]

    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    #clf = svm.LinearSVC()
    clf = LogisticRegression()
    #clf = ensemble.RandomForestClassifier()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print('classification_report\n', metrics.classification_report(y_test, \
                            y_pred, digits=4))
    print('Accuracy:', metrics.accuracy_score(y_test, y_pred))

