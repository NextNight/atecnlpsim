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

import os

data_path = "input_data"
out_path = "output_data"

if not os.path.exists(out_path):
    os.mkdir(out_path)

path_train = os.path.join(data_path, 'atec_nlp_sim_train.csv')
path_test = os.path.join(data_path, 'atec_nlp_sim_train.csv')
path_model = os.path.join(out_path, 'best_model.h5')
path_result = os.path.join(out_path, 'result_pre.csv')


