# !/usr/bin/python
# -*- coding: utf-8 -*-
# created by: hongyu.shao@qunar.com
# date: 2017-03-09 11:56

"""
执行beam search查询
见：https://arxiv.org/pdf/1408.2873.pdf
"""

import os
from collections import namedtuple

import numpy as np
import kenlm
import random
import time
import heapq
from char_map import char_map, index_map


WORD_BLANK = 0  # 表示语音空白
LANG_ATTENUATION_PARAM = 0.25  # 语言模型的衰减参数
LONG_TAIL_PARAM = 0.9  # 网络输出值只处理概率前x的
BETA = 0.25  # softmax的衰减参数
GAMA=0.6

TOP_SUM = 0.0
SORT_SUM = 0.0
MAIN_SUM = 0.0
M_SUM = 0.0


def load_lm(lm_model_path):
    model = kenlm.Model(lm_model_path)
    return model


# 根据语言模型，中文之间用空格隔开
# kenlm score的输出为log10(P)
def __lm_prob(lm_model, next_prefix, prefix):
    if '' == next_prefix:
        return 1.0
    score = -10.0
    try:
        next_prefix = ' '.join(next_prefix)
        all_score = [x for (x, _, _) in lm_model.full_scores(next_prefix, bos=False, eos=False)]
        score = all_score[-1]
    except Exception, e:
        print next_prefix + 'a'
        print e
    #write_to_file(next_prefix + " " + str(10 ** score))
    return 10 ** score


def __top_k(probs, k):
    if len(probs) <= k:
        return probs
    pivot = probs[-1]
    right = [x for x in probs[:-1] if x[1] > pivot[1]]
    rlen = len(right)
    if rlen == k:
        return right
    if rlen > k:
        return __top_k(right, k)
    else:
        left = [x for x in probs[:-1] if x[1] <= pivot[1]]
        right.append(pivot)
        left_result = __top_k(left, k - rlen - 1)
        left_result.extend(right)
        return left_result


def __top_k_from_map(prefix_probs, k):
    """
    采用quick select算法求出概率最大的前k个prefix返回
    时间复制度 O(n)
    :param prefix_probs:  前缀及对应的概率 dict
    :param k:  取概率最大的前k个数
    :return: 返回前k个对应的prefix
    """
    prefix_arr = [i for i in prefix_probs.items()]
    if len(prefix_arr) <= k:
        return prefix_arr
    # 转换为数组，便于处理， 这个比较耗时
    top_list = __top_k(prefix_arr, k)
    return top_list


# 每个timestamp里, 抛弃小概率的词
def __top_props_np(timestep, prop_limit):
    IndexVal = namedtuple('IndexVal', ['index', 'proba'])
    timestep = [IndexVal(index=i, proba=proba) for i, proba in enumerate(timestep)]
    timestep = sorted(timestep, key=lambda x: x.proba, reverse=True)
    filter_indexes = []
    added_proba = 0
    for index_val in timestep:
        added_proba += index_val.proba
        filter_indexes.append(index_val.index)
        if added_proba >= prop_limit:
            break
    return filter_indexes


def __add_prob(next_prefixes, prefix, prob_no_blanks, prob_blanks, i):
    if not prob_no_blanks[i + 1].has_key(prefix):
        prob_no_blanks[i + 1][prefix] = 0
    if not prob_blanks[i + 1].has_key(prefix):
        prob_blanks[i + 1][prefix] = 0
    next_prefixes[prefix] = prob_no_blanks[i + 1][prefix] + prob_blanks[i + 1][prefix]


def __softmax(timestep):
    timestep = np.array(timestep)
    e_vals = np.e ** timestep
    total = np.sum(e_vals)
    return e_vals/total


def __get_prefixes(prefix_probs):
    prefixes = []
    for item in prefix_probs:
        prefixes.append(item[0])
    return prefixes


def prefix_beam_search(lang_model, prediction, beam_width, result_num):
    """
    前缀束搜索
    :param lm_model_path:语言模型路径
    :param prediction: 网络输出概率. 包含map的list
    :param beam_width: 束搜索宽度限制
    :param result_num: 返回前k的概率
    :return: 概率最大的前K个句子
    """
    prob_blanks = [{'': 1.}]  # 记录序列结尾为blank的概率
    prob_no_blanks = [{'': 0.}]  # 记录序列结尾不为blank的概率
    prev_prefixes = [('', 0.)]  # 当前序列及其对应的概率
    for i, timestep in enumerate(prediction):
        next_prefixes = {}
        # 只处理概率前x%的网络输出
        softmax_timestep = __softmax(timestep)
        choose_word_index = __top_props_np(softmax_timestep, LONG_TAIL_PARAM)
        prefixes = __get_prefixes(prev_prefixes)
        if len(prob_blanks) < i + 2:
            prob_blanks.append({})
        if len(prob_no_blanks) < i + 2:
            prob_no_blanks.append({})
        # 循环当前序列
        for prefix in prefixes:
            # 遍历所有词表中所有的词
            for word_index in choose_word_index:
                if WORD_BLANK == word_index:
                    # 计算下一轮序列结尾空白的概率
                    prefix_black_prob = ((softmax_timestep[WORD_BLANK]) * (prob_blanks[i][prefix] + prob_no_blanks[i][prefix]))
                    if len(prefix) == 0:
                        prefix_black_prob = prefix_black_prob
                    else:
                        prefix_black_prob = prefix_black_prob * float((len(prefix))**GAMA)

                    prob_blanks[i + 1][prefix] = prefix_black_prob
                    __add_prob(next_prefixes, prefix, prob_no_blanks, prob_blanks, i)
                else:
                    next_prefix = prefix + index_map[word_index]
                    # 语言模型概率
                    next_lang_prob = __lm_prob(lang_model, next_prefix, prefix)
                    # 语言模型衰减后概率
                    atte_prob = next_lang_prob ** LANG_ATTENUATION_PARAM
                    # 语言模型结合网络输出概率**BETA
                    prefix_word_prob = (softmax_timestep[word_index]**BETA) * (
                        prob_blanks[i][prefix] + prob_no_blanks[i][prefix]) * atte_prob
#                   # positive weight for length
                    if len(prefix) == 0:
                        prefix_word_prob = prefix_word_prob
                    else:
                        prefix_word_prob = prefix_word_prob * float((len(prefix))**GAMA)

                    prob_no_blanks[i + 1][next_prefix] = prefix_word_prob
                    __add_prob(next_prefixes, next_prefix, prob_no_blanks, prob_blanks, i)
        # 取概率值最大的前K个为下次迭代的prev_prefixes
        top_next_prefixes = __top_k_from_map(next_prefixes, beam_width)
        prev_prefixes = top_next_prefixes
     #   print top_next_prefixes
    return __top_k(prev_prefixes, result_num)


