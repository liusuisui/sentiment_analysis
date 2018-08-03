# coding: UTF-8

import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup

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

review = '''With all this stuff going down at the moment with MJ i've started
listening to his music, watching the odd documentary here and there, watched
The Wiz and watched Moonwalker again. Maybe i just want to get a certain
insight into this guy who i thought was really cool in the eighties just to
maybe make up my mind whether he is guilty or innocent. Moonwalker is part
biography, part feature film which i remember going to see at the cinema when
it was originally released. Some of it has subtle messages about MJ's feeling
towards the press and also the obvious message of drugs are bad m'kay. <br/><br/>... '''
print review_to_wordlist(review=review)
print type(review_to_wordlist(review))
#exit(0)
# 载入数据集
train = pd.read_csv('labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
test =  pd.read_csv('testData.tsv', header=0, delimiter="\t", quoting=3)
#train = pd.read_csv('labeledTrainData.head1000.tsv', header=0, delimiter="\t", quoting=3)
#test =  pd.read_csv('testData.head1000.tsv', header=0, delimiter="\t", quoting=3)
print train.head()
print test.head()
#exit(0)
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

#tfidf.fit(data_all)

data_all = vectorizer.fit_transform(data_all)

data_all = data_all.toarray()
#print type(data_all)
print data_all[0]
#exit(0)
#from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF

#3tfidf = TFIDF(min_df=2, #最小支持度2
#           max_features=None,
#            strip_accents='unicode',
#            analyzer='word',
#            token_pattern=r'\w{1,}',
#            ngram_range=(1,3),
#            use_idf=1,
#            smooth_idf=1,
#            sublinear_tf=1,
#            stop_words = 'english')
# 合并训练街和测试集以便进行TFIDF向量化操作
#data_all = train_data + test_data
#print type(data_all)

#len_train = len(train_data)

#tfidf.fit(data_all)
#data_all = tfidf.transform(data_all)
#print type(data_all)
# 恢复成训练集和测试集部分

train_x = data_all[:len_train]
test_x = data_all[len_train:]

#print len(train_x)
print 'bag of words处理结束！'
#print 'TFIDF处理结束'


from xgboost import XGBClassifier
#from xgboost import XGBRegressor
xgbc = XGBClassifier(max_depth=2, learning_rate=0.1, n_estimators=150,
                     subsample=0.8,
                     colsample_btree=0.8,
                     objective='binary:logitraw')

### 交叉验证
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(xgbc, train_x, label, cv=3, scoring='accuracy')
print scores
print scores.mean()
xgbc.fit(train_x, label)

test_predicted = xgbc.predict(test_x)

print '保存结果..'
xgbc_output = pd.DataFrame(data=test_predicted, columns=['sentiment'])
xgbc_output['id'] = test['id']
xgbc_output = xgbc_output[['id','sentiment']]
xgbc_output.to_csv('xgbc_output.csv', index=False)

print '结束！'
