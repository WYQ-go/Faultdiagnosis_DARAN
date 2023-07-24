#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Monday Aug 30 15:17:00 2019
school:HUST
@author: KJ.Zhou
"""

# 改进取样方法，随机选取样本的起始位置，然后从起始位置开始截取样本长度个采样点得到一个样本

from scipy.io import loadmat
import os
import random
import numpy as np
import pandas as pd


def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True


def get_records(data, label, sample_length, sample_number):
    records = []
    # begins = random.sample(range(0, int(len(data) * 100000 / sample_length / sample_number)),
    #                        int(len(data) / sample_length))  # 随机从100000种采样sample_number个点
    random.seed(12354)
    begins = random.sample(range(0, 100000), sample_number)
    for begin in begins:
        end = begin + sample_length
        sample = data[begin:end].reshape(1, 800).tolist()[0]
        record = [label] + sample
        records.append(record)
    print(len(records))
    return records


def get_filenames_by_rpm(data_dir, rpm):
    filenames = [rpm + '-0.000-Normal.mat']
    degrees, faults = ['0.007', '0.014', '0.021'], ['Ball', 'InnerRace', 'OuterRace6']
    for degree in degrees:
        for fault in faults:
            path = rpm + '-' + degree + '-' + fault + '.mat'
            if os.path.exists(data_dir + '/' + path):
                filenames.append(path)
            else:
                raise FileNotFoundError
    return filenames


def get_position_data(data_dir, filenames, position):
    """
    :param data_dir:str
    :param filenames:str
    :param sample_length:int
    :param sample_number:int sample_number*sample_length<121556
    :param position: str DE_time/ FE_time
    :return:
    """
    time_series, Label_keys = [], []
    for filename in filenames:
        Label_keys.append(filename[5:-4])  # 解析文件名到指定标签
        filepath = data_dir + '/' + filename
        m = loadmat(filepath)
        keys = list(m.keys())
        # print(keys) #可以查看.mat文件里面都有什么数据
        for key in keys:
            if position in key:  # 选择驱动端加速度计数据，如有需要可以得到其他加速度计数据
                index = key
        time_series.append(m[index][0:120000])
        print(len(m[index][0:120000]))
    return time_series, Label_keys


def split_dataset_by_rpm(data_dir, filenames, sample_length, sample_number, rate, position, label_dict):
    time_series, Label_list = get_position_data(data_dir, filenames, position)
    for i in range(len(Label_list)):
        print(Label_list[i])
        idx = label_dict[Label_list[i]]
        print(idx)
        records = get_records(time_series[i], idx, sample_length, sample_number)
        temp = np.array(records)

        train_num = int(rate * sample_number)  # rate数据划分成训练集和测试集的比例

        indices = np.random.permutation(temp.shape[0])  # 随机shuffle np.arrange(temp.shape[0])
        train_idx, test_idx = indices[:train_num], indices[train_num:]  # 随机划分成训练集和测试集
        train, test = temp[train_idx, :], temp[test_idx, :]
        if i == 0:
            trains = train
            tests = test
        else:
            trains = np.r_[trains, train]  # 类似list extend操作
            tests = np.r_[tests, test]
    trains, tests = pd.DataFrame(trains), pd.DataFrame(tests)
    trains[0], tests[0] = trains[0].astype(int), tests[0].astype(int)
    return trains, tests


def rpm_diff(rpm_source, rpm_target, position):
    source_filenames = get_filenames_by_rpm(data_dir, rpm_source)
    target_filenames = get_filenames_by_rpm(data_dir, rpm_target)

    source_train_dataset, source_test_dataset = split_dataset_by_rpm(data_dir, source_filenames, sample_len, sample_num,
                                                                     0.7, position, label_dict)
    target_train_dataset, target_test_dataset = split_dataset_by_rpm(data_dir, target_filenames, sample_len, sample_num,
                                                                     0.7, position, label_dict)
    mkdir(save_path)
    source_train_dataset.to_csv(save_path + '/' + rpm_source + 'source_train.csv', index=False, header=False,
                                columns=None)
    source_test_dataset.to_csv(save_path + '/' + rpm_source + 'source_test.csv', index=False, header=False,
                               columns=None)
    print('source save ')
    target_train_dataset.to_csv(save_path + '/' + rpm_target + 'target_train.csv', index=False, header=False,
                                columns=None)
    target_test_dataset.to_csv(save_path + '/' + rpm_target + 'target_test.csv', index=False, header=False,
                               columns=None)
    print('target save ')


def position_diff(position_source, position_target, rpm):
    filenames = get_filenames_by_rpm(data_dir, rpm)
    source_train_dataset, source_test_dataset = split_dataset_by_rpm(data_dir, filenames, sample_len, sample_num,
                                                                     0.7, position_source, label_dict)
    target_train_dataset, target_test_dataset = split_dataset_by_rpm(data_dir, filenames, sample_len, sample_num,
                                                                     0.7, position_target, label_dict)

    mkdir(save_path)
    source_train_dataset.to_csv(save_path + '/' + rpm + position_source + 'source_train.csv', index=False,
                                header=False, columns=None)
    source_test_dataset.to_csv(save_path + '/' + rpm + position_source + 'source_test.csv', index=False,
                               header=False, columns=None)
    print('source save')
    target_train_dataset.to_csv(save_path + '/' + rpm + position_target + 'target_train.csv', index=False,
                                header=False, columns=None)
    target_test_dataset.to_csv(save_path + '/' + rpm + position_target + 'target_test.csv', index=False,
                               header=False, columns=None)
    print('target save')


if __name__ == '__main__':
    root = "D:/Projects/ssl_fault_diagnosis/CWRU_master"
    data_dir = root + '/' + 'dataset/cwru_10'
    save_path = root + '/' + 'dataset/'
    label_dict = {'0.000-Normal': 0,
                  '0.007-Ball': 1, '0.007-InnerRace': 2, '0.007-OuterRace6': 3,
                  '0.014-Ball': 4, '0.014-InnerRace': 5, '0.014-OuterRace6': 6,
                  '0.021-Ball': 7, '0.021-InnerRace': 8, '0.021-OuterRace6': 9}
    sample_len, sample_num = 800, 300
    #############################################################################################
    # rpm_diff(rpm_source='1730', rpm_target='1797', position='DE_time')
    position_diff(position_source='FE_time', position_target='DE_time', rpm='1797')

    # rpm = {'source': '1730', 'target': '1797'}
    # source_filenames = get_filenames_by_rpm(data_dir, rpm['source'])
    # target_filenames = get_filenames_by_rpm(data_dir, rpm['target'])
    #
    # source_train_dataset, source_test_dataset = split_dataset_by_rpm(data_dir, source_filenames, sample_len, sample_num,
    #                                                                  0.7, 'DE_time', label_dict)
    # target_train_dataset, target_test_dataset = split_dataset_by_rpm(data_dir, target_filenames, sample_len, sample_num,
    #                                                                  0.7, 'DE_time', label_dict)
    # mkdir(save_path)
    # source_train_dataset.to_csv(save_path + '/' + rpm['source'] + 'source_train.csv', index=False, header=False,
    #                             columns=None)
    # source_test_dataset.to_csv(save_path + '/' + rpm['source'] + 'source_test.csv', index=False, header=False,
    #                            columns=None)
    # print('source save ')
    # target_train_dataset.to_csv(save_path + '/' + rpm['target'] +'target_train.csv', index=False, header=False, columns=None)
    # target_test_dataset.to_csv(save_path + '/' + rpm['target'] +'target_test.csv', index=False, header=False, columns=None)
    # print('target save ')
    ######################################################################################################
    # position = {'source': 'DE_time', 'target': 'FE_time'}
    # rpm = '1797'
    # filenames = get_filenames_by_rpm(data_dir, rpm)
    # source_train_dataset, source_test_dataset = split_dataset_by_rpm(data_dir, filenames, sample_len, sample_num,
    #                                                                  0.7, position['source'], label_dict)
    # target_train_dataset, target_test_dataset = split_dataset_by_rpm(data_dir, filenames, sample_len, sample_num,
    #                                                                  0.7, position['target'], label_dict)
    #
    # mkdir(save_path)
    # source_train_dataset.to_csv(save_path + '/' + rpm + position['source'] + 'source_train.csv', index=False,
    #                             header=False, columns=None)
    # source_test_dataset.to_csv(save_path + '/' + rpm + position['source'] + 'source_test.csv', index=False,
    #                            header=False, columns=None)
    # print('source save ')
    # target_train_dataset.to_csv(save_path + '/' + rpm + position['target'] + 'target_train.csv', index=False,
    #                             header=False, columns=None)
    # target_test_dataset.to_csv(save_path + '/' + rpm + position['target'] + 'target_test.csv', index=False,
    #                            header=False, columns=None)
    # print('target save ')

    # data_dir = 'D:/desktop/dataset/CRWU/12k Drive End Bearing Fault Data/'
    # fault_types = ['Ball', 'Inner Race', 'Outer Race']
    # fault_degrees = ['0007', '0014', '0021']
    # loads = ['0', '1', '2', '3']
    # for fault_type in fault_types:
    #     for fault_degree in fault_degrees:
    #         for load in loads:
    #             path = data_dir + fault_type + '/' + fault_degree
    #             if fault_type == 'Ball':
    #                 data_path = path + '/' + 'B' + fault_degree[1:] + '_' + load + '.mat'
    #             elif fault_type == 'Inner Race':
    #                 data_path = path + '/' + 'IR' + fault_degree[1:] + '_' + load + '.mat'
    #             else:
    #                 data_path = path + '/' + 'Centered' + '/' + 'OR' + fault_degree[1:] + '@6_' + load + '.mat'
    #             if not os.path.exists(data_path):
    #                 raise FileNotFoundError
    #             m = loadmat(data_path)
    #
    #             print(m[])
    #             a = []
    pass
