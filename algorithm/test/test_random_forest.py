#!/usr/bin/env python
#-*- coding:utf-8 -*-

import re
import pandas as pd
import preprocessing
import arno_random_forest

if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    attributes_dir = '../resources/attributes'

    #attribute_file_path = attributes_dir+'/'+'user_portrait_info_v1'
    #data_path= '/home/login01/Workspaces/python/dataset/module_data_stg2_tt'
    #data_path= '/home/login01/Workspaces/python/dataset/module_data_stg2'

    #attribute_file_path = attributes_dir+'/'+'user_portrait_info_v1_20170608'
    #data_path= '/home/login01/Workspaces/python/dataset/module_data_20170608'

    attribute_file_path = attributes_dir+'/'+'user_portrait_info_v2_20170612'
    data_path= '/home/login01/Workspaces/python/dataset/module_data_stg2_20170612'

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
    #X_train,X_test,y_train,y_test = preprocessing.data_preprocess(data_path=data_path,form=1,attributes=attributes,target_key=target_key,to_binary_attrs=to_binary_attrs,area_attrs=area_attrs,show=False)
    X_train,X_test,y_train,y_test = preprocessing.data_preprocess(data_path=data_path,form=2,attributes=attributes,all_labels=all_labels,target_key=target_key,to_binary_attrs=to_binary_attrs,area_attrs=area_attrs,show=False,cut_point=6)
    #arno_random_forest.train_test(X_train,X_test,y_train,y_test)
    arno_random_forest.train_test1(X_train,X_test,y_train,y_test)

