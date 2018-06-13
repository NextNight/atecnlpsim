#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
import networkx as nx
import matplotlib.pyplot as plt

from gensim.models import Word2Vec
from gensim import corpora, similarities
from gensim.models.tfidfmodel import TfidfModel
from gensim.models.ldamodel import LdaModel
from gensim.models.lsimodel import LsiModel

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances


def word_graph():
    G = nx.DiGraph()
    G.add_edge('a', 'e')
    G.add_edge('b', 'e')
    G.add_edge('c', 'e')
    G.add_edge('d', 'e')
    nx.draw(G, with_labels=True, width=0.5)
    plt.savefig("youxiangtu.png")
    plt.show()


def lda_similarity(cps, cps1, cps2, dic):
    # 计算s1,s2词频LDA相似度
    print('starting lda similarity...')
    lda = LdaModel(corpus=cps, num_topics=100, id2word=dic)
    s1_lda = lda[cps1]
    s2_lda = lda[cps2]
    sm = similarities.MatrixSimilarity(corpus=s1_lda, num_features=lda.num_topics)
    lda_sm = np.diag(sm[s2_lda])

    return lda_sm


def lsi_similarity(cps, cps1, cps2, dic):
    # 计算s1,s2词频LSI相似度
    print("starting lsi similarity....")
    lsi = LsiModel(corpus=cps, num_topics=100, id2word=dic)
    s1_lsi = lsi[cps1]
    s2_lsi = lsi[cps2]
    sm = similarities.MatrixSimilarity(corpus=s1_lsi, num_features=lsi.num_topics)
    lsi_sm = np.diag(sm[s2_lsi])
    return lsi_sm


def gensim_similarity(data_c):
    """
    使用Gensim包计算相似度：
        词频
            COUNT
            LDA
            LSI
        Tfidf:
            TFIDF
            LDA
            LSI
    """
    # 合并获取词袋
    data_c['s1'] = data_c['s1'].apply(lambda text: list(text))
    data_c['s2'] = data_c['s2'].apply(lambda text: list(text))
    data_c_all = data_c['s1'].append(data_c['s2'], ignore_index=True).to_frame(name='s')

    # 构建词典
    print("starting create dic....")
    dic = corpora.Dictionary(data_c['s1'].values)
    dic.add_documents(data_c['s2'].values)

    print("文档数：", dic.num_docs)
    print("starting create count bow...")
    data_c['s1'] = data_c['s1'].apply(lambda text: dic.doc2bow(text))
    data_c['s2'] = data_c['s2'].apply(lambda text: dic.doc2bow(text))
    data_c_all['s'] = data_c_all['s'].apply(lambda text: dic.doc2bow(text))

    # cps1 = [dic.doc2bow(text) for text in list(data_c['s1'].values)]
    # cps2 = [dic.doc2bow(text) for text in list(data_c['s2'].values)]

    cps1 = list(data_c['s1'])
    cps2 = list(data_c['s2'])
    cps = list(data_c_all['s'])

    # 计算s1,s2词频相似度
    print("starting count similarity....")
    sm = similarities.SparseMatrixSimilarity(corpus=cps1, num_features=10000)
    count_sm = np.diag(sm[cps2])

    # 计算s1,s2词频LDA,LSI相似度
    count_lda_sm = lda_similarity(cps, cps1, cps2, dic)
    # count_lsi_sm= lsi_similarity(cps,cps1,cps2,dic)

    # 计算s1,s2 tfidf相似度
    print("starting tfidf similarity....")
    tfidf = TfidfModel(corpus=cps, id2word=dic)
    cps1_tfidf = tfidf[cps1]
    cps2_tfidf = tfidf[cps2]
    cps_tfidf = tfidf[cps]

    # 计算s1,s2 TFIDF相似度
    sm = similarities.SparseMatrixSimilarity(corpus=cps1_tfidf, num_features=10000)
    tfidf_sm = np.diag(sm[cps2_tfidf])

    # 计算s1,s2词频LDA,LSI相似度
    tfidf_lda_sm = lda_similarity(cps_tfidf, cps1_tfidf, cps2_tfidf, dic)
    tfidf_lsi_sm = lda_similarity(cps_tfidf, cps1_tfidf, cps2_tfidf, dic)

    return count_sm, count_lda_sm, tfidf_sm, tfidf_lda_sm, tfidf_lsi_sm


def jaccard_dis(data):
    """
    杰拉德相似度：计算两个共同词的占比
    """
    jaccard_sm = np.zeros(data.shape[0], dtype='float64')
    for i, (s1, s2) in enumerate(zip(data['s1'], data['s2'])):
        words1 = s1.split()
        words2 = s2.split()
        union = words1 + words2
        intersection = [w.strip() for w in words1 if w in words2]
        jaccard_sm[i] = len(intersection) / len(union)
    return jaccard_sm


def levenshteinl():
    """
    编辑距离：计算一个字符串变换多少次到另一个字符串
    """
    pass


def remove_stop_words(s, stop_words, similar_words):
    """
    去除停用词，同时替换同义词
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
    # if word_split.strip()=='':
    #     word_split = word_split.strip()+'NAN'
    return word_split.strip()


def train_word_vector(data):
    """
    训练词向量
    :param data_all:
    :return:
    """
    data_all = data['s1'].append(data['s2'], ignore_index=True).to_frame(name='sentence')
    data_all = list(np.array(data_all.apply(lambda x: str(x).split(' '))))
    for i in [1, 2, 3, 4, 5, 6]:
        model = Word2Vec(data_all, window=i, sg=0, size=100, min_count=1, negative=3, sample=0.001, hs=1, workers=4,
                         cbow_mean=1)
        model.save(os.path.join(model_config.data_path, 'model.model'))
        model.wv.save_word2vec_format(os.path.join(model_config.data_path, "model.ve" + str(i)), binary=False)


def build_embeddings_dict():
    embeddings_dict = {}
    with open(os.path.join(model_config.data_path, 'model.ve1'), encoding='utf-8') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float64')
            embeddings_dict[word] = coefs
    return embeddings_dict


def word_embedding_vector(s, embeddings_dict):
    """
    词嵌入构建句子向量
    :return:
    """
    svt = np.zeros(100, dtype='float64')
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
    if not os.path.exists(os.path.join(model_config.data_path, 'model.ve5')):
        train_word_vector(data)
    embeddings_dict = build_embeddings_dict()
    vs1 = data['s1'].apply(lambda s: word_embedding_vector(s, embeddings_dict)).values
    vs2 = data['s2'].apply(lambda s: word_embedding_vector(s, embeddings_dict)).values
    print(vs1[0:3])
    # 余弦相似度
    pairwise_distances(vs1,)
    # # TODO：其他距离计算
    # sm2 = []

    return sm1, sm2


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
    print(data.describe())
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
    data['s1'] = data['s1'].apply(lambda s: remove_stop_words(str(s), stop_words, similar_words)).fillna('NA')
    data['s2'] = data['s1'].apply(lambda s: remove_stop_words(str(s), stop_words, similar_words)).fillna('NA')
    return data


# 特征工程
def feature_enginee(data):
    """
    特征工程：构建特征
    """
    data_c = data.copy(deep=True)
    count_sm, count_lda_sm, tfidf_sm, tfidf_lda_sm, tfidf_lsi_sm = gensim_similarity(data_c=data_c)
    del data_c
    #
    data['count_sm'] = count_sm
    data['count_lda_sm'] = count_lda_sm
    data['tfidf_sm'] = tfidf_sm
    data['tfidf_lda_sm'] = tfidf_lda_sm
    data['tfidf_lsi_sm'] = tfidf_lsi_sm

    # 长度差
    data['lengths_cha'] = np.abs(
        data['s1'].apply(lambda x: len(str(x))) - data['s2'].apply(lambda x: len(str(x))))
    # jaccard_dis
    data['jaccard'] = jaccard_dis(data)

    # 词嵌入相似度
    # data['wes1'], data['wes2'] = word_embedding_smiliary(data)

    #
    data.to_csv("output_data/vector.csv",index=False,encoding='utf-8')

# 构建模型
def build_model():
    pass


def main():
    data = load_data()
    # 分词，替换同义词
    data = cut_words(data)
    data = feature_enginee(data)


if __name__ == '__main__':
    main()
    # graph()
