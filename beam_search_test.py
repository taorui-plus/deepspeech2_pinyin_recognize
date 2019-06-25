#encoding:utf-8
"""
Test a trained speech model over a dataset
"""

from __future__ import absolute_import, division, print_function
import kenlm
import json
import numpy as np
import edit_distance
import time
from beam_search import prefix_beam_search
import argparse
from data_generator import DataGenerator
from config_util import get_value
from model import compile_test_fn
from utils import argmax_decode, load_model, ctc_input_length, conv_output_length
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


lm_model = kenlm.Model("language_model/zhongjin.bin")

def test(model, test_fn, datagen, result_file, mb_size=16, conv_context=11,
         conv_border_mode='valid', conv_stride=2):
# def test(model, test_fn, datagen, result_file, mb_size=16):
    
    total_distance = 0
    total_length = 0
    wf = open(result_file, 'w')
    for batch in datagen.iterate_test(mb_size):
        inputs = batch['x']
        labels = batch['y']
        input_lengths = batch['input_lengths']
        label_lengths = batch['label_lengths']
        ground_truth = batch['texts']

        output_lengths = [conv_output_length(l, conv_context,
                                             conv_border_mode, conv_stride)
                          for l in input_lengths]
        predictions, ctc_cost = test_fn([inputs, output_lengths, labels,
                                        label_lengths, True])

        # ctc_in_length = ctc_input_length(model, input_lengths)
        # predictions, ctc_cost = test_fn([inputs, ctc_in_length, labels,
        #                                 label_lengths, False])
        predictions = np.swapaxes(predictions, 0, 1)
        for i, prediction in enumerate(predictions):
            truth = ground_truth[i]
            # 最佳结果
            pre_prediction = argmax_decode(prediction)
            # 前三结果
            preds = prefix_beam_search(lm_model, matrix_same_delete(prediction), 100, 3)

            max_pred_precision = []
            for pred in preds:
                max_pred_precision.append(pred[1])
            # 求三个中的最大概率
            max_index = max_pred_precision.index(max(max_pred_precision))
            # 获取三个中概率最大的字符串
            best_pred_str = preds[max_index][0]
            # 计算标签和概率最大字符串的编辑距离
            sm = edit_distance.SequenceMatcher(a=truth,b=best_pred_str)
            sm2 = edit_distance.SequenceMatcher(a=truth,b=pre_prediction)
            total_distance += sm.distance()
            total_length += len(truth)
            content = json.loads('{}')
            content['label'] = truth
            content['text'] = best_pred_str
            content['lm_distance'] = sm.distance()/len(truth)
            content['no_lm'] = pre_prediction
            content['no_lm_distance'] = sm2.distance()/len(truth)
            __write_and_print(wf, json.dumps(content, ensure_ascii=False))

    total_distance_rate = -1 if total_length == 0 else float(total_distance)/total_length
    print ('total_distance_rate:%s'%total_distance_rate)
    wf.close()


def matrix_same_delete(prediction):
    result_prediction = []
    int_sequence = []
    for timestep in prediction:
        int_sequence.append(np.argmax(timestep))
    c_prev = -1
    for i in range(len(prediction)):
        if int_sequence[i] == c_prev:
            continue
        if int_sequence[i] != 0:
            result_prediction.append(prediction[i])
            c_prev = int_sequence[i]
    return np.array(result_prediction)


def main(test_desc_file, train_desc_file, load_dir, result_file='test_result'):
    datagen = DataGenerator()
    datagen.load_test_data(test_desc_file)
    datagen.load_train_data(train_desc_file)
    datagen.fit_train(100)
    # load model
    model = load_model(load_dir)
    test_fn = compile_test_fn(model)
    # test begin
    start = time.time()
    test(model, test_fn, datagen, result_file)
    print ('elapsed: %s'%(time.time() - start))


def __write_and_print(wf, content):
    wf.write(content.strip('\n'))
    wf.write('\n')
    wf.flush()
    print (content)
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('test_desc_file', type=str,
                        help='Path to a JSON-line file that contains '
                             'test labels and paths to the audio files. ')
    parser.add_argument('train_desc_file', type=str,
                        help='Path to the training JSON-line file. This will '
                             'be used to extract feature means/variance')
    parser.add_argument('load_dir', type=str,
                        help='Directory where a trained model is stored.')
    args = parser.parse_args()
    main(args.test_desc_file, args.train_desc_file, args.load_dir)
