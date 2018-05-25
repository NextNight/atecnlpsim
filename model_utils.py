#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
  @Time    : 13:47
  @Author  : chengxin
  @Site    :
  @File    : MLpythonLearn
  @Software: PyCharm
  @Contact : 1031329190@qq.com
"""
# pip install python-Levenshtein
from sklearn.metrics import jaccard_similarity_score
import os
import numpy as np
import pandas as pd
import model_config
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec


def tf_idf_similarity(data):
    """
    tfidf相似度：
    :return:
    """
    # 合并获取词袋
    data_all = data['s1'].append(data['s2'], ignore_index=True).to_frame(name='sentence')
    print(data_all.head())

    # ifidf向量化
    tfidf_v = TfidfVectorizer(min_df=1, token_pattern='(?u)\\b\\w+\\b', ngram_range=(1, 8), analyzer='word')
    tfidf_v.fit(data_all['sentence'].values)
    vs1 = tfidf_v.transform(data['s1'].values)
    vs2 = tfidf_v.transform(data['s2'].values)
    print(vs1.shape)

    # 存放向量相似度
    vsm_sm = vsm_similary(vs1,vs2)
    return vsm_sm

def vsm_similary(vs1,vs2):
    """
    欧式距离
    """
    vsm_sm = np.zeros(vs1.shape[0], dtype='float64')
    for i in range(vs1.shape[0]):
        d = 0.0
        for a, b in zip(vs1[i], vs2[i]):
            d += (a - b) ** 2;
        vsm_sm[i] = d ** 0.5
    return vsm_sm


def jaccard_dis(data):
    """
    杰拉德相似度：计算两个共同词的占比
    """
    jaccard_sm =np.zeros(data.shape[0])
    for i,(s1,s2) in enumerate(zip(data['s1'],data['s2'])):
        words1 = s1.split()
        words2 = s2.split()
        union = words1 + words2
        intersection = [ w.strip() for w in words1 if w in words2]
        jaccard_sm[i] = len(intersection)/len(union)
    return jaccard_sm


def levenshteinl():
    """
    编辑距离：计算一个字符串变换多少次到另一个字符串
    """
    pass



def remove_stop_words(s, stop_words, similar_words):
    """
    去除停用词，同时替换同义词
    :param s:
    :param stop_words:
    :param similar_words:
    :return:
    """
    word_split = ''
    sentence_list = jieba.cut(s, cut_all=False, HMM=True)
    for word in sentence_list:
        for similar_word in similar_words:
            if word in similar_word.split(' '):
                word = similar_word.split(' ')[0].strip()
        if word not in stop_words:
            if word != '\t':
                word_split += word.strip() + ' '
    return word_split.strip()


def train_word_vector(data):
    """
    训练词向量
    :param data_all:
    :return:
    """
    data_all = data['s1'].append(data['s2'], ignore_index=True).to_frame(name='sentence')
    data_all = data_all.apply(lambda x: str(x).split(' ')).values()
    for i in [1, 2, 3, 4, 5, 6]:
        model = Word2Vec(data_all, window=i, sg=0, size=100, min_count=1, negative=3, sample=0.001, hs=1, workers=4,
                         cbow_mean=1)
        model.save(os.path.join(model_config.data_path,'model.model'))
        model.wv.save_word2vec_format(os.path.join(model_config.data_path,"model.ve" + str(i)), binary=False)

def build_embeddings_dict():
    embeddings_dict = {}
    with open(os.path.join(model_config.data_path, 'model.ve1'), encoding='utf-8') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float64')
            embeddings_dict[word] = coefs
    return embeddings_dict

def word_embedding_vector(s,embeddings_dict):
    """
    词嵌入构建句子向量
    :return:
    """
    svt = np.zeros(100)
    for word in s.split(' '):  # ==x.split(' ')
        word_v = embeddings_dict.get(word)
        if word_v is not None:
            svt += word_v
    return svt

def word_embedding_smiliary(data):
    """
    相似度：100维的句子向量的相似度
    :param data:
    :return:
    """
    if not os.path.exists(os.path.join(model_config.data_path, 'model.ve1')):
        train_word_vector(data)
    embeddings_dict = build_embeddings_dict()
    vs1 = data['s1'].apply(lambda s:word_embedding_vector(s,embeddings_dict)).values()
    vs2 = data['s2'].apply(lambda s:word_embedding_vector(s,embeddings_dict)).values()
    # 欧式距离相似度
    sm1 = vsm_similary(vs1,vs2)
    # TODO：其他距离计算
    sm2 = []

    return sm1,sm2

def load_data():
    """
    加载数据：
    :return:
    """
    columns = ['index', 's1', 's2', 'similarity']
    data = pd.read_csv(
        model_config.path_train,
        sep='\t',
        header=None,
        names=columns,
        index_col=0,
        encoding='utf-8',
    )
    data.describe()
    return data


def cut_words(data):
    """
    预处理：
        分词，去停词，合并同义词
    """
    # 加载自定义词典
    jieba.load_userdict(os.path.join(model_config.data_path, 'atec_dict.dict'))
    # 见了鬼了，自定义辞典不起做用
    # ['花呗','借呗','借呗','蚂蚁借呗','支付宝']
    jieba.add_word('花呗')
    jieba.add_word('借呗')
    jieba.add_word('蚂蚁借呗')
    jieba.add_word('蚂蚁花呗')
    jieba.add_word('支付宝')
    jieba.add_word('余额宝')
    jieba.add_word('是不是')
    jieba.add_word('不是')
    jieba.add_word('怎么还款')
    jieba.add_word('怎么开通')
    jieba.add_word('还能')
    jieba.add_word('开不了')
    jieba.add_word('开通不了')
    jieba.add_word('要还')
    # 停用词
    stop_words_path = os.path.join(model_config.data_path, 'stop_words.dict')
    stop_words = [line.strip() for line in open(stop_words_path, 'r', encoding='utf-8').readlines()]
    # 同义词
    similar_words_path = os.path.join(model_config.data_path, 'atec_similarity.dict')
    similar_words = [line.strip() for line in open(similar_words_path, 'r', encoding='utf-8').readlines()]
    # 去停用词，合并同义词
    data['s1'] = data['s1'].apply(lambda s: remove_stop_words(str(s), stop_words, similar_words))
    data['s2'] = data['s1'].apply(lambda s: remove_stop_words(str(s), stop_words, similar_words))

    return data


# 特征工程
def feature_enginee(data):
    """
    特征工程：构建特征
    """
    # 分词，替换同义词
    data = cut_words(data)
    # tf_idf相似度
    data['tfidf_vsmm'] = tf_idf_similarity(data)
    # 长度差
    data['lengths_cha'] = np.abs(
        data['s1'].apply(lambda x: len(str(x))) - data['s2'].apply(lambda x: len(str(x))))
    # jaccard_dis
    data['jaccard'] = jaccard_dis(data)

    # 词嵌入相似度
    data['wes1'],data['wes2'] = word_embedding_smiliary(data)

    #
    pass


# 构建模型
def build_model():
    pass

def main():
    data = load_data()
    data = feature_enginee(data)


if __name__ == '__main__':
    main()
