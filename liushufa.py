# coding: UTF-8

import pandas as pd
import numpy as np
import time
from sklearn.metrics import roc_auc_score
import re
from bs4 import BeautifulSoup

st = time.time()
# 数据清洗函数
def review_to_wordlist(review):
    '''
    吧IMDB的评论转换成次序列
    '''
    # 去掉HTML标签，拿到内容
    review_text = BeautifulSoup(review, "html.parser").get_text()
    #用正则表达式取出符合规范的部分
    review_text = re.sub("[^a-zA-Z]"," ",review_text)
    # 小写化所有的词，并转成词list
    #return review_text
    words = review_text.lower().split()
    # 返回words
    return words

# 载入数据集

#train = pd.read_csv('labeledTrainData.head1000.tsv', header=0, delimiter="\t", quoting=3)
#test =  pd.read_csv('testData.head1000.tsv', header=0, delimiter="\t", quoting=3)
#train = pd.read_csv('labeledTrainData.head100.tsv', header=0, delimiter="\t", quoting=3)
#test =  pd.read_csv('testData.head100.tsv', header=0, delimiter="\t", quoting=3)
train = pd.read_csv('labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
test =  pd.read_csv('testData.tsv', header=0, delimiter="\t", quoting=3)
print train.head()
print test.head()

# 预处理数据
label = train['sentiment']
train_data = []
c = 0
for i in range(len(train['review'])):
    c += 1
    if c % 1000 == 0:
        print c
    train_data.append(' '.join(review_to_wordlist(train['review'][i])))

test_data = []
c = 0
for i in range(len(test['review'])):
    c += 1
    if c % 1000 == 0:
        print c
    test_data.append(' '.join(review_to_wordlist(test['review'][i])))


# 预览数据
#print train_data[0]
#print test_data[0]
#
#print type(train_data[0])
#print type(test_data[0])
#
#print len(train_data)
#print len(test_data)

# 特征处理
from sklearn.feature_extraction.text import CountVectorizer

# bag of words tool

vectorizer = CountVectorizer(analyzer = "word",
                          tokenizer = None,
                          preprocessor = None,
                          stop_words = 'english',
                          max_features = 5000)



data_all = train_data + test_data
len_train = len(train_data)

data_all = vectorizer.fit_transform(data_all)

data_all = data_all.toarray()

#print type(data_all)
#print data_all[0]

train_x = data_all[:len_train]
test_x = data_all[len_train:]
print 'bag of words处理结束！'

from xgboost import XGBClassifier
#from xgboost import XGBRegressor
xgbc = XGBClassifier(max_depth=2, learning_rate=0.1, n_estimators=500,
                     subsample=0.8,
                     colsample_btree=0.8,
                     objective='binary:logitraw')

### 交叉验证
#from sklearn.cross_validation import cross_val_score
#scores = cross_val_score(xgbc, train_x, label, cv=3, scoring='accuracy')
#print scores
#print scores.mean()

# 留出法
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_x, label, test_size=0.2, random_state=31)
xgbc.fit(X_train, y_train)
pred_test  = xgbc.predict(X_test)
auc = roc_auc_score(y_test, pred_test)
print 'xgbc测试集表现',auc
pred_train = xgbc.predict(X_train)
auc = roc_auc_score(y_train, pred_train)
print 'xgbc训练集表现',auc
print '保存结果..'
print '保存结果..'

# 预测测试集
xgbc.fit(train_x, label)
test_predicted = xgbc.predict(test_x)


output = open('xgbc.result.csv', 'wb')
output.write('id,sentiment\n')
for i in range(len(test_predicted)):
    output.write('%s,%s\n' % (test['id'][i], test_predicted[i]))

print '结束！'

print time.time() - st
