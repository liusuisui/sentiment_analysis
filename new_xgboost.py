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
    # 去掉单个字母的元素
    words = filter(lambda x: len(x) >= 2, words)
    # 返回words
    return words

# 载入数据集

train = pd.read_csv('labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
test =  pd.read_csv('testData.tsv', header=0, delimiter="\t", quoting=3)
#train = pd.read_csv('labeledTrainData.head1000.tsv', header=0, delimiter="\t", quoting=3)
#test =  pd.read_csv('testData.head1000.tsv', header=0, delimiter="\t", quoting=3)
#train = pd.read_csv('labeledTrainData.head100.tsv', header=0, delimiter="\t", quoting=3)
#test =  pd.read_csv('testData.head100.tsv', header=0, delimiter="\t", quoting=3)
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

print type(data_all)
print data_all[0]

train_x = data_all[:len_train]
test_x = data_all[len_train:]
fmap = dict(map(lambda x: ('f%d' % (x[1]), x[0]), vectorizer.vocabulary_.items()))
print 'bag of words处理结束！'

#from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
#
#tfidf = TFIDF(min_df=2, #最小支持度2
#           max_features=None,
#            strip_accents='unicode',
#            analyzer='word',
#            token_pattern=r'\w{1,}',
#            ngram_range=(1,3),
#            use_idf=1,
#            smooth_idf=1,
#            sublinear_tf=1,
#            stop_words = 'english')
#
##合并训练街和测试集以便进行TFIDF向量化操作
#data_all = train_data + test_data
#print type(data_all)
#
#len_train = len(train_data)
#
#tfidf.fit(data_all)
#data_all = tfidf.transform(data_all)
#print type(data_all)
##恢复成训练集和测试集部分
#
#train_x = data_all[:len_train]
#test_x = data_all[len_train:]
##print len(train_x)
##print 'bag of words处理结束！'
#print 'TFIDF处理结束'
#fmap = dict(map(lambda x: ('f%d' % (x[1]), x[0]), tfidf.vocabulary_.items()))

import xgboost as xgb
params = {
    'booster':'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eta': 0.025,
    'alpha': 1,
    'seed': 0,
    #'nthread': 8,
    'silent': 1
}

# 留出法
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_x, label, test_size=0.2, random_state=31)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
watchlist = [(dtrain,'train'), (dtest, 'test')]
bst = xgb.train(params, dtrain, num_boost_round=1000, evals=watchlist)
ypred = bst.predict(dtest)
y_pred = (ypred >= 0.5) * 1
from sklearn import metrics
print 'AUC: %.4f' % metrics.roc_auc_score(y_test, ypred)
print 'ACC: %.4f' % metrics.accuracy_score(y_test, y_pred)
print 'Recall: %.4f' % metrics.recall_score(y_test, y_pred)
print 'F1-score: %.4f' %metrics.f1_score(y_test, y_pred)
print 'Precesion: %.4f' %metrics.precision_score(y_test, y_pred)
for key, value in sorted(bst.get_fscore().items(), key=lambda x: -x[1])[:200]:
    print fmap.get(key), value

dtest = xgb.DMatrix(test_x)
ypred = bst.predict(dtest)
test_predicted = (ypred >= 0.5) * 1

output = open('xgbc.result.csv', 'wb')
output.write('id,sentiment\n')
for i in range(len(test_predicted)):
    output.write('%s,%s\n' % (test['id'][i], test_predicted[i]))

print '结束！'

print time.time() - st
