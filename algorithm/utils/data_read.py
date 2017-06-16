#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os
import re
import numpy as np
import pandas as pd
from preprocessing import DataCheck,DataPreprocessing
from sklearn import model_selection

#def data_read(data_path,form=0,attributes=None,all_labels=None,target_key=None,test_size=0.3,cut_point=0,random_state=99):
def data_read(data_path,form=0,all_labels=None,test_size=0.3,cut_point=0,random_state=99):
    '''
    :param data_path: string, the data's path 
    :param form: int, indicate that what type of data to process:
            0: data isn't split to train and test set, one whole file
            1: data has been split to train and test set, two files
            2: data is split to multiple directories, each directory exists one or multiple files
    :param attributes: list of string, labels of X
    :param all_labels: list of string, labels of data, including all X and y, even the label isn't used, should use when form=2
    :param target_key: string, label of target
    :param stats_file_path: string, the path of statistical files 
    :param test_size: float, the size ratio of test 
    :param cut_point: int, the path of statistical files 
    :param random_state: int, random state 
    :return: DataFrame: df,train,test
    '''
    df = None
    train = None
    test = None
    try:
        if form == 0:
            for dirpath,dirnames,filenames in os.walk(data_path):
                # for dirname in dirnames:
                #    print dirname
                for filename in filenames:
                    file_path = os.path.join(dirpath,filename)
                    print(file_path)
            data_df = pd.read_csv(file_path,sep=',',na_values='NA',low_memory=False)
            #X = df[attributes]
            #y = df[target_key]
            #X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=test_size,random_state=random_state)
            train,test = model_selection.train_test_split(data_df,test_size=test_size,random_state=random_state)
            df = pd.concat([train,test],axis=0)
            df = df.reset_index(drop=True)
        if form == 1:
            files=[]
            for dirpath,dirnames,filenames in os.walk(data_path):
                for filename in filenames:
                    file_path = os.path.join(dirpath,filename)
                    files.append(file_path)
            if len(files) == 2:
                for file in files:
                    if not isinstance(file,str):
                        print("Error: files parameter should be string list type")
                train = pd.read_csv(files[0],sep=',',na_values='NA',low_memory=False)
                test = pd.read_csv(files[1],sep=',',na_values='NA',low_memory=False)
                df = pd.concat([train,test],axis=0)
                df = df.reset_index(drop=True)
            else:
                print("Error: please split the data in the directory %s into two files: train and test set, like train_csv.csv, test_csv.csv" % data_path)
            return df,train,test
        if form == 2:
            dict_files={}
            files = []
            for dirpath,dirnames,filenames in os.walk(data_path):
                for filename in filenames:
                    file_path = os.path.join(dirpath,filename)
                    if file_path==(data_path+'/'+filename): #there is noly files, no subdirectory in the directory data_path
                        files.append(file_path)
                    else: #there are only subdirectories in the directory data_path, one subdirectory is also included
                        dict_files.setdefault(dirpath,file_path)
            if len(dict_files)>0:
                files_sort_key = sorted(dict_files.items(),key=lambda item:item[0]) # sort by key(dirpath, or subdirectory)
                for i in range(len(files_sort_key)):
                    files.append(files_sort_key[i][1])
            df_lst = []
            for i in range(len(files)):
                df_lst.append(pd.read_csv(files[i],header=None,names=all_labels,sep=',',na_values='NA',low_memory=False))
            train = None
            test = None
            if cut_point >= len(df_lst):
                print("Error: the cut_point should be less then ",len(df_lst))
            for i in range(len(df_lst)):
                if i<=cut_point:
                    train = pd.concat([train,df_lst[i]],axis=0)
                else:
                    test = pd.concat([test,df_lst[i]],axis=0)
            train = train.reset_index(drop=True)
            test = test.reset_index(drop=True)
            df = pd.concat([train,test],axis=0)
            df = df.reset_index(drop=True)
        else:
            pass
    except Exception as e:
        print(e)
    finally:
        return df,train,test

if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    attributes_dir = '../resources/attributes'

    attribute_file_path = attributes_dir+'/'+'user_portrait_info_v1'
    data_path= '/home/login01/Workspaces/python/dataset/module_data_stg2_tt'
    #data_path= '/home/login01/Workspaces/python/dataset/module_data_stg2'

    #attribute_file_path = attributes_dir+'/'+'user_portrait_info_v1_20170608'
    #data_path= '/home/login01/Workspaces/python/dataset/module_data_20170608'

    #attribute_file_path = attributes_dir+'/'+'user_portrait_info_v2_20170612'
    #data_path= '/home/login01/Workspaces/python/dataset/module_data_stg2_20170612'

    attribute_file = open(attribute_file_path,'r')
    attributes = []
    all_labels = []
    while True:
        line = attribute_file.readline()
        if not line:
            break
        matchObj = re.match(r'\#(.*)',line.strip('\n'),re.M | re.I)
        if matchObj:
            all_labels.append(matchObj.group(1))
        else:
            attributes.append(line.strip('\n'))
            all_labels.append(line.strip('\n'))
    target_key="user_own_overdue"
    #target_key="user_own_ninety_overdue_order"

    to_binary_attrs = ['user_live_address','user_rela_name','user_relation','user_rela_phone','user_high_edu','user_company_name']
    area_attrs = ['user_live_province','user_live_city']
    df,train,test = data_read(data_path=data_path,form=1)
    print(df.info())
    print("------------")
    print(train.info())
    print("------------")
    print(test.info())
    df.to_csv('./df.csv',sep=',')
    train.to_csv('./train.csv',sep=',')
    test.to_csv('./test.csv',sep=',')
