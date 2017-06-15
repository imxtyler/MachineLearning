#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import os
import csv
import numpy as np
import pandas as pd
import codecs
from sklearn.ensemble import RandomForestRegressor

#float regex pattern
float_regex_patterns = [
    r'[-+][0-9]+\.[0-9]+[eE][-+]?[0-9]+',
    r'[-+][0-9]+\.[0-9]+',
    r'\A[-+]?[0-9]+\.[0-9]+\Z',
    r'\A[-+]?[0-9]*\.[0-9]+\Z',
    r'\A[-+]?([0-9]+(\.[0-9]+)?|\.[0-9]+)\Z',
    r'\A[-+]?([0-9]+(\.[0-9]*)?|\.[0-9]+)\Z',
    r'\A[-+]?([0-9]+(\.[0-9]+)?|\.[0-9]+)([eE][-+]?[0-9]+)?\Z',
    r'\A[-+]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][-+]?[0-9]+)?\Z',
    r'\A[-+]?(\b[0-9]+(\.[0-9]+)?|\.[0-9]+)([eE][-+]?[0-9]+\b)?\Z'
    ]
#int regex pattern
int_regex_patterns = [
    r'[-+][0-9]+[eE][+]?[0-9]+',
    r'[-+][0-9]+',
    r'\A[-+]?[0-9]+\Z',
    r'\A[-+]?[0-9]*[0-9]+\Z',
    r'\A[-+]?[0-9]+([eE][+]?[0-9]+)?\Z',
    r'\A[+-]?\b[0-9]+\b\Z',
    r'\A[-+]?\b[0-9]+([eE][+]?[0-9]+\b)?\Z',
    r'(?:^|(?<=\s))[0-9]+(?=$|\s)',
    r'([+-] *)\b[0-9]+\b'
    ]

number_type = [np.int,np.int8,np.int16,np.int32,np.int64,
               np.float,np.float16,np.float32,np.float64,np.float128,
               np.double]
def is_dtype_numberical(x):
    '''
    :param x: the dtype that need to be judge whether it is numberical 
    :return: boolean
    '''
    try:
        if x in number_type:
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False

class DataCheck():
    def __init__(self,df,attributes,key):
        '''
        :param df: the dataframe containing the attriute and target
        :param key: the target's label in the classification problem
        '''
        self.sample_num,self.attr_num = df.shape
        self.df = df
        self.attributes = attributes
        self.key = key

    def check_type(self):
        '''
        :param strategy: string, the strategy to process the abnormal type, it should be in {"null","mean","discard"}
        :return: 
        '''
        try:
            for item in self.attributes:
                if (is_dtype_numberical(self.df[item].dtype)):
                    pass
                else:
                    is_abnormal,col_type = self.check_and_choose_col_type(list(self.df[item]),item)
                    if is_abnormal == True:
                        if col_type is float:
                            #self.df[item] = self.df[item].apply(self.convert_to_float(strategy))
                            #self.df[item] = self.df[item].map(self.convert_to_float(strategy))
                            self.df[item] = self.df[item].apply(self.convert_to_float)
                            self.df[item] = self.df[item].astype(float)
                        else:
                            pass
                    else:
                        pass
        except Exception as e:
            print(e)

        #return self.df

    def check_value(self):
        pass

    def is_numberical_type(self,item):
        matchF = False
        matchI = False
        for pattern in float_regex_patterns:
            matchO = re.match(pattern, str(item), re.M | re.I)
            matchF = matchF or (matchO is not None)
        for pattern in int_regex_patterns:
            matchO = re.match(pattern, str(item), re.M | re.I)
            matchI = matchI or (matchO is not None)

        return (matchI == True or matchF == True)

    def check_and_choose_col_type(self,col=[],col_name=None):
        '''
        :param col:list
        :return: is_abnormal:bool, indicate that there is abnormal value type or not
                 type, the data type that the column should be
        '''
        num_cnt = 0
        str_cnt = 0
        obj_cnt = 0
        nan_cnt = 0
        for item in col:
            if self.is_numberical_type(item):
                num_cnt = num_cnt+1
            elif isinstance(item,str):
                str_cnt = str_cnt+1
            elif np.isnan(item):
                nan_cnt = nan_cnt+1
            else:
                obj_cnt = obj_cnt+1
        type_list = [num_cnt,str_cnt,obj_cnt]
        is_abnormal = (((num_cnt+nan_cnt)==len(col) or (str_cnt+nan_cnt)==len(col) or (obj_cnt+nan_cnt)==len(col))==False)
        if num_cnt == max(type_list):
            return is_abnormal,float
        if str_cnt == max(type_list):
            return is_abnormal,str
        if obj_cnt == max(type_list):
            return is_abnormal,object

    def convert_to_float(self,item,strategy="null"):
        '''
        :param strategy: string, the strategy to process the abnormal type, it should be in {"null","mean"}
        :param item: 
        :return: 
        '''
        if self.is_numberical_type(item):
            float(item)
        else:
            if strategy == "null":
                item = np.nan
            #if strategy == "mean":
            #    pass #fixme

        return item

class DataPreprocessing():
    def __init__(self,df,attributes,key):
        '''
        :param df: the dataframe containing the attriute and target
        :param attributes: the X's attributes that need to be calculated the gini score and is type 1
        :param key: the target's label in the classification problem
        '''
        self.sample_num,self.attr_num = df.shape
        self.df = df
        self.attributes = attributes
        self.key = key
        self.x_df = None

    def data_summary(self,show=True,stats=True,file_path_stats='/tmp/data_statistics.csv'):
        '''
        :param show: bool, indicate that it should print the data's summary information
        :param stats: bool, indicate whether it does the data's statistical work
        :param file_path_stats: string, the file full path of statistical file
        :return:
        '''
        if show==True:
            print("number of attributes:%d, number of samples:%d" % (self.attr_num,self.sample_num))
            print(self.df.info())
            #print(self.df.describe())
            print("ratio of null values of each attribute:")
            print(self.df.isnull().sum()/self.sample_num)
            #print("targe key, size of non-null values:",self.df.groupby(self.key).size())
            print("targe key, size of negetive values:",self.key, (self.sample_num-self.df[self.key].sum()))
            print("targe key, size of positive values:",self.df[self.key].sum())
            print("targe key, rate of positive values:",(self.df[self.key].sum()/self.sample_num))
        else:
            pass
        if stats==True:
            self.value_counts(file_path_stats=file_path_stats)
        else:
            pass

    def value_counts(self,output=True,file_path_stats='/tmp/data_statistics.csv'):
        stats = {}
        for key in list(self.df.columns.values):
            #print type(self.df['user_marriage_nan'])
            stats[key] = self.df[key].value_counts(dropna=False)
        if output:
            MAX_LEN = max([len(stats[key].index) for key in self.df.columns])
            csvfile = open(file_path_stats, 'w')
            #csvfile.write(codecs.BOM_UTF8)
            #csvfile.write(codecs.UTF-8)
            writer = csv.writer(csvfile)
            header = []
            for key in self.df.columns:
                header.append(key)
                header.append('number')
            writer.writerow(header)
            for line in range(MAX_LEN):
                row = []
                for key in self.df.columns:
                    if line < len(stats[key]):
                        row.append(stats[key].index[line])
                        row.append(stats[key].values[line])
                    else:
                        row.append(' ')
                        row.append(' ')
                writer.writerow(row)
            csvfile.close()
        return stats

    #def set_x_missing_label(self,numberical_attributes,label):
    #    '''
    #    :param numberical_attributes: the numberical attributes that need to be calculated
    #    :param label: the label containing null value
    #    '''
    #    # Put all the numberical features into Random Forest Regressor
    #    label_df = self.df[numberical_attributes]

    #    # Divide the customers into two parts: who's label is known and unknown
    #    known_label = label_df[label_df[label].notnull()].as_matrix()
    #    unknown_label = label_df[label_df[label].isnull()].as_matrix()

    #    # Target label
    #    label_y = known_label[:, 0]

    #    # X are feature attributes
    #    label_X = known_label[:, 1:]

    #    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    #    rfr.fit(label_X, label_y)
    #    predictedlabels = rfr.predict(unknown_label[:, 1::])

    #    # Fill the null value via the predict labels
    #    self.df.loc[(self.df.label.isnull()),label] = predictedlabels

    #    return self.df,rfr

    def set_x_missing_label_mean(self,attributes,allnull=False,nullvalue=-1):
        '''
        :param attributes: the attributes who's null value need to be filled with mean value
        :param allnull: bool, sign that if need to process the attribute whose value is all null
        :param nullvalue: the value that using fill the attribute whose value is all null
        '''
        for attr in attributes:
            mean_attr = self.df[attr].mean(skipna=True)
            if allnull==True:
                if (len(self.df[self.df[attr].isnull()])==self.sample_num):
                    self.df.loc[(self.df[attr].isnull()),attr] = nullvalue
                else:
                    #df[attr] = df[attr].fillna(mean_attr,inplace=True)
                    self.df.loc[(self.df[attr].isnull()),attr] = mean_attr
            else:
                self.df.loc[(self.df[attr].isnull()),attr] = mean_attr

        if self.key in (list(self.df.columns.values)):
            self.x_df = self.df.drop(self.key,axis=1,inplace=False)
        else:
            self.x_df = self.df
        return self.x_df

    def filter_numberical_attr(self):
        numberical_attrs = []
        nonnumberical_attrs = []
        #df._get_numeric_data().columns # another way
        #list(set(cols) - set(num_cols)) # another way
        try:
            for item in self.attributes:
                if (is_dtype_numberical(self.df[item].dtype)):
                    numberical_attrs.append(item)
                else:
                    nonnumberical_attrs.append(item)
            return numberical_attrs,nonnumberical_attrs
        except Exception as e:
            print(e)
            return numberical_attrs,nonnumberical_attrs

    def transform_x_dtype(self,attributes,d_type,uniform_type=False):
        '''
        :param attributes: list, the attributes that need to be transformed its' dtype 
        :param d_type: list, the destination dtypes of attributes, the entry's value must be int32, int64, float32, float64, etc.
        :param uniform_type: bool, indicate that the types of attributes are the same or not. the length of attributes and d_type must be the same if not
        '''
        try:
            if uniform_type == True:
                for attr in attributes:
                    self.df[attr] = self.df[attr].notnull().astype(d_type[0])
            else:
                attrs_types = dict(zip(attributes,d_type))
                for key,val in attrs_types.items():
                    self.df[key] = self.df[key].notnull().astype(val)
        except Exception as e:
            print(e)

        if self.key in (list(self.df.columns.values)):
            self.x_df = self.df.drop(self.key,axis=1,inplace=False)
        else:
            self.x_df = self.df
        return self.x_df

    def transform_x_to_dummies(self,attributes):
        '''
        :param attributes: the attributes that need to be transformed to dummies
        '''
        dummies_attrs = []
        for item in attributes:
            dummies_item = pd.get_dummies(self.df[item],prefix=item)
            dummies_attrs.append(dummies_item)

        if self.key in (list(self.df.columns.values)):
            self.x_df = self.df.drop(self.key,axis=1,inplace=False)
        else:
            self.x_df = self.df
        return self.x_df,dummies_attrs

    def transform_x_to_binary(self,attributes):
        '''
        :param attributes: the attributes that need to be transformed to binary 
        '''
        for item in attributes:
            #self.df.loc[(self.df[item].isnull()),item] ='No'
            #self.df.loc[(self.df[item].notnull()),item]='Yes'
            self.df.loc[(self.df[item].isnull()),item] =0
            self.df.loc[(self.df[item].notnull()),item]=1

        if self.key in (list(self.df.columns.values)):
            self.x_df = self.df.drop(self.key,axis=1,inplace=False)
        else:
            self.x_df = self.df
        return self.x_df

    def china_area_number_mapping(self,attributes,resource_dir):
        '''
        :param attributes: the attributes that need to be transformed to area number 
        :param resource_dir: the directory including the resource files
        '''
        area_dict = {}
        try:
            for dirpath,dirnames,filenames in os.walk(resource_dir):
                for file in filenames:
                    if file=='china_area_number.csv':
                        filepath = dirpath+'/'+file
                        res_file = open(filepath,'r')
                        while True:
                            line = res_file.readline()
                            if not line:
                                break
                            area_number,area = line.strip().split(',')
                            area_dict.setdefault(area,area_number)
        except Exception as e:
            print(e)
        area_mapping = area_dict
        for attr in attributes:
            self.df[attr] = self.df[attr].map(area_mapping)

        #return area_dict
        if self.key in (list(self.df.columns.values)):
            self.x_df = self.df.drop(self.key,axis=1,inplace=False)
        else:
            self.x_df = self.df
        return self.x_df

    def single_attr_binning(self,attribute,bin_num=1,labels=None):
        '''
        :param attribute: the attribute that needs to be binned 
        :param bin_num: the number of bins
        :param labels: the labels of bins
        :return: bins, type:Series 
        '''
        break_points = []
        minval = self.df[attribute].min()
        maxval = self.df[attribute].max()
        bin_length = (maxval-minval)/bin_num
        for i in range(0,bin_num+1):
            if (minval+bin_length*i)<maxval:
                break_points.append(minval+bin_length*i)
        break_points.append(maxval)
        if not labels:
            labels = range(len(break_points)-1)
        #attr_bin = pd.cut(self.df[attribute],bins=break_points,labels=labels,right=False)
        #attr_bin = pd.qcut(self.df[attribute],bin_num)
        attr_bin = pd.cut(self.df[attribute],bins=break_points,labels=labels,include_lowest=True)

        return attr_bin

    def multi_attrs_binning(self,attributes,bin_num=1,labels=None):
        '''
        :param attributes: the attributes that need to be binned 
        :param bin_num: the number of bins
        :param labels: the labels of bins
        :return: bins, type:Series 
        '''
        attributes_bin = []
        for attr in attributes:
            attr_bin = self.single_attr_binning(attr,bin_num=bin_num,labels=labels)
            attributes_bin.append(attr_bin)

        return attributes_bin

    def transform_to_binning_rate(self,key,attributes,bin_num=1,labels=None):
        '''
        :param key: the target's label in the classification problem, that is y
        :param attributes: the attributes that need to transform to binning_rate 
        :param bin_num: the number of bins
        :param labels: the labels of bins
        :param allnull: bool, sign that if need to process the attribute whose value is all null
        :param nullvalue: the value that using fill the attribute whose value is all null
        :return: df 
        '''
        pass  # fixme

    def x_dummies_and_fillna(self,allnull=False,nullvalue=-1):
        numberical_attrs,nonnumberical_attrs = self.filter_numberical_attr()
        self.x_df,dummies_attrs = self.transform_x_to_dummies(nonnumberical_attrs)
        for item in dummies_attrs:
            self.x_df = pd.concat([self.x_df,item],axis=1)
        self.x_df.drop(nonnumberical_attrs,axis=1,inplace=True)
        self.df = pd.concat([self.x_df,self.df[self.key]],axis=1)
        self.x_df = self.set_x_missing_label_mean(list(self.df.columns),allnull=allnull,nullvalue=nullvalue)

        return self.x_df