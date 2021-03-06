#!/usr/bin/env python
# -*- coding:utf-8 -*-  
__time__ = '2020-06-31 17:27'

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.layers import Convolution1D,BatchNormalization,concatenate,Flatten
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from data_loader import*

from keras.models import load_model
from keras.models import *

#读取数据
base_dir = '/content/drive/My Drive/colab/news_classfication/data'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

save_dir = '/content/drive/My Drive/colab/news_classfication/checkpoints/textcnn'

sentences = build_vocab(train_dir, vocab_dir, )

# #构建词汇表
# if not os.path.exists(vocab_dir):
#     build_vocab(train_dir, vocab_dir,)

# 创建数据字典
categories, cat_to_id = read_category()
words, word_to_id = read_vocab(vocab_dir)
id_to_word = read_vocab_reverse(vocab_dir)
vocab_size = len(words)

seq_length = 600  # 序列长度
x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, seq_length)
x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, seq_length)

model1 = load_model(os.path.join(save_dir,'w2v_train_model_tem.h5'))
## 对测试集进行预测
y_pre = model1.predict(x_val)

print(x_val.shape)
print(x_val)

print(np.argmax(y_pre,axis=1).shape)
print(np.argmax(y_pre,axis=1))
f = open('emd_res.txt','w')

for i in range(len(x_val)):
  if(np.argmax(y_pre,axis=1)[i] != np.argmax(y_val,axis=1)[i]):
    news=''
    for j in x_val[i]:
      if j !=0:
        news += id_to_word[j]
      
    f.write('true label :'+ ' '+ str(categories[np.argmax(y_val,axis=1)[i]])+ ' ; predicted label : '+ str(categories[np.argmax(y_pre,axis=1)[i]])+'\n')
    f.write(news)
    f.write('\n')



metrics.classification_report(np.argmax(y_val,axis=1),np.argmax(y_pre,axis=1), digits=4, output_dict=True)

## 评价预测效果，计算混淆矩阵
confm = metrics.confusion_matrix(np.argmax(y_val,axis=1),np.argmax(y_pre,axis=1))


plt.figure(figsize=(8,8))
sns.heatmap(confm.T, square=True, annot=True,
            fmt='d', cbar=False,linewidths=.8,
            cmap="YlGnBu")
plt.xlabel('true label',size = 14)
plt.ylabel('predicted label',size = 14)
categories_num=[str(i) for i in range(1,11)]
plt.xticks(np.arange(10)+0.5,categories_num,size = 12)
plt.yticks(np.arange(10)+0.3,categories_num,size = 12)
# plt.xticks(np.arange(10)+0.5,categories,fontproperties = fonts,size = 12)
# plt.yticks(np.arange(10)+0.3,categories,fontproperties = fonts,size = 12)
plt.savefig('matrix')
plt.show()


print(metrics.classification_report(np.argmax(y_val,axis=1),np.argmax(y_pre,axis=1), digits=4))