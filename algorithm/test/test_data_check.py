#!/usr/bin/env python
#-*- coding:utf-8 -*-

import pandas
from data_check import DataCheck

if __name__ == '__main__':
    file_path_data = '/home/login01/Workspaces/python/dataset/qiubin/stg2_checked_dummied.csv'
    file_path_stats = '/home/login01/Workspaces/python/dataset/qiubin/stg2_checked_stats.csv'
    info = pandas.read_csv(file_path_data)
    check_data = DataCheck(info,file_path_stats)
    check_data.check()
    check_data.dummies()
    check_data.del_columns()
    #check_data.info.to_csv('/home/login01/Workspaces/python/dataset/qiubin/stg2_checked_dummied.csv')
    check_data.info.to_csv('/home/login01/Workspaces/python/dataset/qiubin/stg2_checked_dummied_tmp.csv')
