#!/usr/bin/env python
# -*- coding:utf-8 -*-
__time__ = '2020-07-02 16:29'
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
from data_loader import *
from gensim.models import Word2Vec
#读取数据
base_dir = '/content/drive/My Drive/colab/news_classfication/data'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')

save_dir = '/content/drive/My Drive/colab/news_classfication/checkpoints/textcnn'

w2v_save_path = os.path.join(base_dir,'w2v.model')

# sentences = None
# #构建词汇表
# if not os.path.exists(vocab_dir) :
#     sentences = build_vocab(train_dir, vocab_dir,)
sentences = build_vocab(train_dir, vocab_dir, )
print(sentences[:3])

# 创建数据字典
categories, cat_to_id = read_category()
words, word_to_id = read_vocab(vocab_dir)
vocab_size = len(words)

seq_length = 600  # 序列长度
x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, seq_length)
x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, seq_length)

# if sentences is None:
if not os.path.exists(w2v_save_path):
  model = Word2Vec(sentences, size=256, window=5, min_count=1, workers=2, sg=1)
  model.save(w2v_save_path)

w2v_model=Word2Vec.load(w2v_save_path)
embedding_matrix = np.zeros((vocab_size + 1 , 256))
for word in words:
    try:
        embedding_vector = w2v_model[str(word)]
        embedding_matrix[word_to_id[word]] = embedding_vector
    except KeyError:
        continue

#构建模型
main_input = Input(shape=(600,), dtype='float64')
# embedder = Embedding(vocab_size + 1, 256, input_length = 600, weights=[embedding_matrix], trainable=True)
embedder = Embedding(vocab_size + 1, 256, input_length = 600, weights=[embedding_matrix],trainable = True)
embed = embedder(main_input)
block1 = Convolution1D(128, 1, padding='same')(embed)
conv2_1 = Convolution1D(256, 1, padding='same')(embed)
bn2_1 = BatchNormalization()(conv2_1)
relu2_1 = Activation('relu')(bn2_1)
block2 = Convolution1D(128, 3, padding='same')(relu2_1)
inception = concatenate([block1, block2], axis=-1)
flat = Flatten()(inception)
fc = Dense(128)(flat)
drop = Dropout(0.5)(fc)
bn = BatchNormalization()(drop)
relu = Activation('relu')(bn)
main_output = Dense(10, activation='softmax')(relu)
model = Model(inputs = main_input, outputs = main_output)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

from keras.utils import plot_model
from PIL import Image
    #生成一个模型图，第一个参数为模型，第二个参数为要生成图片的路径及文件名，还可以指定两个参数：
    #show_shapes:指定是否显示输出数据的形状，默认为False
    #show_layer_names:指定是否显示层名称，默认为True
plot_model(model,to_file='model.png',show_shapes=True,show_layer_names=False)

#模型训练
history = model.fit(x_train, y_train,
          batch_size=32,
          epochs=30,
          validation_data=(x_val, y_val))


#绘制准确率和损失值
def plot_acc_loss(history):
    plt.subplot(211)
    plt.title("Accuracy")
    plt.plot(history.history["accuracy"], color="g", label="Train")
    plt.plot(history.history["val_accuracy"], color="b", label="Test")
    plt.legend(loc="best")

    plt.subplot(212)
    plt.title("Loss")
    plt.plot(history.history["loss"], color="g", label="Train")
    plt.plot(history.history["val_loss"], color="b", label="Test")
    plt.legend(loc="best")

    plt.tight_layout()
    plt.savefig('w2v_t_acc_loss')
    plt.show()


plot_acc_loss(history)

## 模型的保存和导入
from keras.models import load_model
#保存模型
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
model.save(os.path.join(save_dir,'w2v_train_model_tem.h5'))
# model.save('w2v_model.h5')
print('模型保存成功')

 #对测试集进行预测
y_pre = model.predict(x_val)

metrics.classification_report(np.argmax(y_val,axis=1),np.argmax(y_pre,axis=1), digits=4, output_dict=True)



## 评价预测效果，计算混淆矩阵
confm = metrics.confusion_matrix(np.argmax(y_val,axis=1),np.argmax(y_pre,axis=1))


## 混淆矩阵可视化
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
plt.savefig('w2v_t_matrix')
plt.show()


print(metrics.classification_report(np.argmax(y_val,axis=1),np.argmax(y_pre,axis=1), digits=4))