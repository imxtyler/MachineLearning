#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas
from sklearn.ensemble import RandomForestRegressor

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

    def data_summary(self):
        print("number of attributes:%d, number of samples:%d" % (self.attr_num,self.sample_num))
        print(self.df.info())
        #print(self.df.describe())
        print("ratio of null values of each attribute:")
        print(self.df.isnull().sum()/self.sample_num)
        print("targe key, size of non-null values:",self.df.groupby(self.key).size())
        print("targe key, size of negetive values:",(self.sample_num-self.df[self.key].sum()))
        print("targe key, size of positive values:",self.df[self.key].sum())
        print("targe key, rate of positive values:",(self.df[self.key].sum()/self.sample_num))

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
        try:
            for item in self.attributes:
                if (self.is_dtype_numberical(self.df[item].dtype)):
                    numberical_attrs.append(item)
                else:
                    nonnumberical_attrs.append(item)
            return numberical_attrs,nonnumberical_attrs
        except Exception as e:
            print(e)
            return numberical_attrs,nonnumberical_attrs

    def is_dtype_numberical(self,x):
        '''
        :param x: the dtype that need to be judge whether it is float
        :return: boolean
        '''
        try:
            return (x=='int32') or (x=='int64') or (x=='float32') or (x=='float64')
        except Exception as e:
            print(e)
            return False

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
            dummies_item = pandas.get_dummies(self.df[item],prefix=item)
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
        #attr_bin = pandas.cut(self.df[attribute],bins=break_points,labels=labels,right=False)
        #attr_bin = pandas.qcut(self.df[attribute],bin_num)
        attr_bin = pandas.cut(self.df[attribute],bins=break_points,labels=labels,include_lowest=True)

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
            self.x_df = pandas.concat([self.x_df,item],axis=1)
        self.x_df.drop(nonnumberical_attrs,axis=1,inplace=True)
        self.df = pandas.concat([self.x_df,self.df[self.key]],axis=1)
        self.df = self.set_x_missing_label_mean(list(self.df.columns),allnull=allnull,nullvalue=nullvalue)

        if self.key in (list(self.df.columns.values)):
            self.x_df = self.df.drop(self.key,axis=1,inplace=False)
        else:
            self.x_df = self.df
        return self.x_df