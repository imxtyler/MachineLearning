#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import os
import csv
import numpy as np
import pandas as pd
import codecs
from sklearn.ensemble import RandomForestRegressor
from time import time

import data_read

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
    def __init__(self,df,key):
        '''
        :param df: the dataframe containing the attriute and target
        :param key: the target's label in the classification problem
        '''
        self.sample_num,self.attr_num = df.shape
        self.df = df
        self.key = key

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
            #csvfile = open(file_path_stats, 'wb') # used for python2
            #csvfile.write(codecs.BOM_UTF8) # used for python2
            csvfile = codecs.open(file_path_stats, 'w+', 'utf_8_sig') # used for python3
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

    def check_type(self,show=True,stats=True,file_path_stats='/tmp/data_statistics.csv'):
        '''
        :param file_path_stats: string, the file full path of statistical file
        :return self.df: DataFrame
        '''
        self.data_summary(show=show,stats=stats,file_path_stats=file_path_stats)
        abnormal_lst = []
        try:
            for item in self.df.columns:
                if (is_dtype_numberical(self.df[item].dtype)):
                    pass
                else:
                    is_abnormal,col_type = self.check_and_choose_col_type(list(self.df[item]))
                    if is_abnormal == True:
                        abnormal_lst.append(item)
                        if col_type is float:
                            #self.df[item] = self.df[item].apply(self.convert_to_float)
                            self.df[item] = self.df[item].map(self.convert_to_float)
                            self.df[item] = self.df[item].astype(float)
                        else:
                            pass
                    else:
                        pass
        except Exception as e:
            print(e)
        abnormal_col_str = ','.join(abnormal_lst)
        print("The columns "+abnormal_col_str+" have abnormal type value, please see the statistical file:",file_path_stats)

        return self.df

    def check_value(self):
        try:
            for column in list(self.df.columns.values):
                column = str(column)
                if re.match(r'.*\_name', column, re.I):
                    self.df[column] = self.df[column].map(self.check_name)
                if re.match(r'.*\_gender|.*\_sex', column, re.I):
                    self.df[column] = self.df[column].map(self.check_gender)
                if re.match(r'.*\_age', column, re.I):
                    self.df[column] = self.df[column].map(self.check_age)
                if re.match(r'.*\_idcard|.*id\_card', column, re.I):
                    self.df[column] = self.df[column].map(self.check_idcard)
                if re.match(r'.*\_phone', column, re.I):
                    self.df[column] = self.df[column].map(self.check_phone)
                if re.match(r'.*\_mail|.*\_email', column, re.I):
                    self.df[column] = self.df[column].map(self.check_mail)
                if re.match(r'.*\_province|.*\_prov', column, re.I):
                    self.df[column] = self.df[column].map(self.check_province)
                if re.match(r'.*\_city', column, re.I):
                    self.df[column] = self.df[column].map(self.check_city)
        except Exception as e:
            print(e)
        finally:
            return self.df

    def check_name(self,name):
        try:
            if np.isnan(name):
                return 0
        except:
            #for ch in name.decode('utf-8'): # for python2
            for ch in name:
                if u'\u4e00' <= ch <= u'\u9fff' or 'a' <= ch <= 'z':
                    return 1
                else:
                    return 0

    def check_gender(self,gender):
        if gender not in ['男', '女', 'male', 'female']:
            return np.nan
        else:
            return gender

    def check_age(self,age):
        try:
            if np.isnan(age):
                return age
            age = float(age)
            if age > 100 and age < 10:
                return -100
            else:
                return age
        except:
            return np.nan

    def check_idcard(self,idcard):
        idcard = str(idcard)
        if re.match('^([1-9]\d{5}[12]\d{3}(0[1-9]|1[012])(0[1-9]|[12][0-9]|3[01])\d{3}[0-9xX])$', idcard):
            return 1
        else:
            return 0

    def check_phone(self,phone):
        phone = str(phone)
        if re.match('^0\d{2,3}\d{7,8}$|^1[358]\d{9}$|^147\d{8}', phone):
            return 1
        else:
            return 0

    def check_mail(self,mail):
        mail = str(mail)
        if re.match(r'^[0-9a-zA-Z_]{0,19}@[0-9a-zA-Z]{1,13}\.[com,cn,net]{1,3}$', mail):
            return 1
        else:
            return 0

    def check_province(self,province):
        try:
            if np.isnan(province):
                return np.nan
            else:
                return np.nan
        except:
            #for ch in province.decode('utf-8'): # for python2
            for ch in province:
                if u'\u4e00' <= ch <= u'\u9fff' or 'a' <= ch <= 'z':
                    continue
                else:
                    return np.nan
        return province

    def check_city(self,city):
        try:
            if np.isnan(city):
                return np.nan
            else:
                return np.nan
        except:
            #for ch in city.decode('utf-8'): # for python2
            for ch in city:
                if u'\u4e00' <= ch <= u'\u9fff' or 'a' <= ch <= 'z':
                    continue
                else:
                    return np.nan
        return city

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

    def check_and_choose_col_type(self,col=[]):
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
            if strategy == "mean":
                pass #fixme

        return item

#china area regex pattern
china_area_regex_patterns = [
    r'(.*)特别行政区.*\Z',
    r'(.*)各族.*\Z',
    r'(.*)左翼蒙古族.*\Z',
    r'(.*)满族蒙古族.*\Z',
    r'(.*)蒙古族藏族.*\Z',
    r'(.*)土家族苗族.*\Z',
    r'(.*)苗族土家族.*\Z',
    r'(.*)苗族侗族.*\Z',
    r'(.*)壮族瑶族.*\Z',
    r'(.*)壮族苗族.*\Z',
    r'(.*)黎族苗族.*\Z',
    r'(.*)藏族羌族.*\Z',
    r'(.*)仡佬族苗族.*\Z',
    r'(.*)布依族苗族.*\Z',
    r'(.*)苗族布依族.*\Z',
    r'(.*)彝族回族苗族.*\Z',
    r'(.*)回族彝族.*\Z',
    r'(.*)彝族回族.*\Z',
    r'(.*)彝族傣族.*\Z',
    r'(.*)傣族彝族.*\Z',
    r'(.*)哈尼族彝族傣族.*\Z',
    r'(.*)哈尼族彝族.*\Z',
    r'(.*)彝族哈尼族拉祜族.*\Z',
    r'(.*)傣族拉祜族佤族.*\Z',
    r'(.*)拉祜族佤族布朗族傣族.*\Z',
    r'(.*)傣族佤族.*\Z',
    r'(.*)苗族瑶族傣族.*\Z',
    r'(.*)傣族景颇族.*\Z',
    r'(.*)独龙族怒族.*\Z',
    r'(.*)白族普米族.*\Z',
    r'(.*)保安族东乡族撒拉族.*\Z',
    r'(.*)回族土族.*\Z',
    r'(.*)蒙古族.*\Z',
    r'(.*)回族.*\Z',
    r'(.*)藏族.*\Z',
    r'(.*)维吾尔族.*\Z',
    r'(.*)苗族.*\Z',
    r'(.*)彝族.*\Z',
    r'(.*)壮族.*\Z',
    r'(.*)布依族.*\Z',
    r'(.*)朝鲜族.*\Z',
    r'(.*)满族.*\Z',
    r'(.*)侗族.*\Z',
    r'(.*)瑶族.*\Z',
    r'(.*)白族.*\Z',
    r'(.*)土家族.*\Z',
    r'(.*)哈尼族.*\Z',
    r'(.*)哈萨克族.*\Z',
    r'(.*)傣族.*\Z',
    r'(.*)黎族.*\Z',
    r'(.*)僳僳族.*\Z',
    r'(.*)佤族.*\Z',
    r'(.*)畲族.*\Z',
    r'(.*)高山族.*\Z',
    r'(.*)拉祜族.*\Z',
    r'(.*)水族.*\Z',
    r'(.*)东乡族.*\Z',
    r'(.*)纳西族.*\Z',
    r'(.*)景颇族.*\Z',
    r'(.*)柯尔克孜族.*\Z',
    r'(.*)土族.*\Z',
    r'(.*)达斡尔族.*\Z',
    r'(.*)仫佬族.*\Z',
    r'(.*)羌族.*\Z',
    r'(.*)布朗族.*\Z',
    r'(.*)撒拉族.*\Z',
    r'(.*)毛南族.*\Z',
    r'(.*)仡佬族.*\Z',
    r'(.*)锡伯族.*\Z',
    r'(.*)阿昌族.*\Z',
    r'(.*)普米族.*\Z',
    r'(.*)塔吉克族.*\Z',
    r'(.*)怒族.*\Z',
    r'(.*)乌孜别克族.*\Z',
    r'(.*)俄罗斯族.*\Z',
    r'(.*)鄂温克族.*\Z',
    r'(.*)德昂族.*\Z',
    r'(.*)保安族.*\Z',
    r'(.*)裕固族.*\Z',
    r'(.*)京族.*\Z',
    r'(.*)塔塔尔族.*\Z',
    r'(.*)独龙族.*\Z',
    r'(.*)鄂伦春族.*\Z',
    r'(.*)赫哲族.*\Z',
    r'(.*)门巴族.*\Z',
    r'(.*)珞巴族.*\Z',
    r'(.*)基诺族.*\Z',
    r'(.*)回族自治区\Z',
    r'(.*)壮族自治区\Z',
    r'(.*)自治区\Z',
    r'(.*)自治州\Z',
    r'(.*)自治县\Z',
    r'(.*)自治旗\Z',
    r'(.*)地区\Z',
    r'(.*)新区\Z',
    r'(.*)市\Z',
    r'(.*)县\Z'
]
china_city_regex_patterns = [
    r'(.*)市\Z',
    r'(.*)盟\Z',
    r'(.*)地区\Z',
    r'(.*)朝鲜族自治州\Z',
    r'(.*)土家族苗族自治州\Z',
    r'(.*)藏族羌族自治州\Z',
    r'(.*)藏族自治州\Z',
    r'(.*)彝族自治州\Z',
    r'(.*)布依族苗族自治州\Z',
    r'(.*)苗族侗族自治州\Z',
    r'(.*)哈尼族彝族自治州\Z',
    r'(.*)壮族苗族自治州\Z',
    r'(.*)傣族自治州\Z',
    r'(.*)白族自治州\Z',
    r'(.*)傣族景颇族自治州\Z',
    r'(.*)傈僳族自治州\Z',
    r'(.*)回族自治州\Z',
    r'(.*)蒙古族藏族自治州\Z',
    r'(.*)蒙古自治州\Z',
    r'(克孜勒苏柯尔克孜)自治州\Z',
    r'(伊犁哈萨克)自治州\Z'
]

class DataPreprocessing():
    def __init__(self,df,attributes,key,resource_dir):
        '''
        :param df: the dataframe containing the attriute and target
        :param attributes: list, the X's attributes that need to be calculated the gini score and is type 1
        :param key: the target's label in the classification problem
        '''
        self.sample_num,self.attr_num = df.shape
        self.df = df
        self.attributes = attributes
        self.key = key
        self.x_df = None
        self.resource_dir = resource_dir
        self.area_mapping = None
        self.province_mapping = None
        self.city_mapping = None

    def discard_trivial_attrs(self,null_ratio_threshold=0.95):
        '''
        :param null_ratio_threshold: the threshold that one attribute's null value ratio 
        :return: DataFrame, self.df 
        '''
        attr_null_ratio = self.df.isnull().sum()/self.sample_num
        dict_attr_null_ratio = dict(zip(self.df.columns.values,attr_null_ratio.values))
        for key,val in dict_attr_null_ratio.items():
            if val > null_ratio_threshold:
                rm_key = key
                self.df.drop(key,axis=1,inplace=True)
                if (self.attributes is not None) and (rm_key in self.attributes):
                    self.attributes.remove(rm_key)

        return self.df

    def china_province_number_mapping(self):
        province_dict = {}
        try:
            for dirpath,dirnames,filenames in os.walk(self.resource_dir):
                for file in filenames:
                    if file=='china_province_area_number.csv':
                        filepath = dirpath+'/'+file
                        res_file = open(filepath,'r')
                        while True:
                            line = res_file.readline()
                            if not line:
                                break
                            province_number,province = line.strip().split(',')
                            #province_dict.setdefault(province_number,province)
                            province_dict.setdefault(province,province_number)
        except Exception as e:
            print(e)

        self.province_mapping = province_dict

    def china_city_number_mapping(self):
        city_dict = {}
        try:
            for dirpath,dirnames,filenames in os.walk(self.resource_dir):
                for file in filenames:
                    if file=='china_city_area_number.csv':
                        filepath = dirpath+'/'+file
                        res_file = open(filepath,'r')
                        while True:
                            line = res_file.readline()
                            if not line:
                                break
                            city_number,city = line.strip().split(',')
                            #city_dict.setdefault(city_number,city)
                            matchO = None
                            for pattern in china_city_regex_patterns:
                                if matchO is not None:
                                    break
                                matchO = re.match(pattern, str(city), re.I)
                            if matchO:
                                new_city = matchO.group(1)
                                #city_dict.setdefault(city_number,new_city)
                                city_dict.setdefault(new_city,city_number)
                            else:
                                #city_dict.setdefault(city_number,city)
                                city_dict.setdefault(new_city,city_number)
        except Exception as e:
            print(e)

        self.city_mapping = city_dict

    def do_china_province_mapping(self,item):
        item_val = np.nan
        #if np.isnan(item):
        if str(item) == 'nan':
            pass
        else:
            for key,val in self.province_mapping.items():
                if key in item:
                    item_val = val
                    break
        item = item_val

        return item

    def do_china_city_mapping(self,item):
        item_val = np.nan
        #if np.isnan(item):
        if str(item) == 'nan':
            pass
        else:
            for key,val in self.city_mapping.items():
                if key in item:
                    item_val = val
                    break
        item = item_val

        return item

    def check_province_city_consistency(self,province,city,do_mapping=True):
        '''
        :param province:
        :param city:
        :param do_mapping: bool, indicate that if do_mapping on the column province and city
        :return:
        '''
        print("Performing check province and city's consistency...")
        t = time()
        area_df = self.df
        self.china_province_number_mapping()
        self.china_city_number_mapping()
        area_df[province] = self.df[province].map(self.do_china_province_mapping)
        area_df[city] = self.df[city].map(self.do_china_city_mapping)
        col_name = 'province_city_consistency'
        #area_df[col_name] = Series(np.random.randn(self.sample_num),index=area_df.index)
        #area_df[col_name] = area_df[province].map(lambda x: np.nan if str(x) == 'nan' else np.nan)
        #area_df = area_df.assign(col_name=pd.Series(np.random.randn(self.sample_num)).values)
        area_df[col_name] = area_df[province].map(lambda z: np.nan if str(z) == 'nan' else (area_df[province].map(lambda x: str(x)[0:1]) == area_df[city].map(lambda y: str(y)[0:1])))
        if do_mapping == True:
            self.df = area_df
        else:
            self.df = pd.concat([self.df,area_df[col_name]],axis=1)
        self.df = self.df.reset_index(drop=True)
        self.attributes = self.attributes.append(col_name)
        print("Check province and city's consistency done in %0.3fs" % (time() - t))
        print()

        return self.df

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

    def set_x_null_attr_mean(self,attributes,allnull=False,nullvalue=-1):
        '''
        :param attributes: the attributes who's null value need to be filled with mean value
        :param allnull: bool, indicate that if need to process the attribute whose value is all null
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

        self.x_df = self.df[attributes]
        return self.x_df

    def filter_numberical_attrs(self):
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

        self.x_df = self.df[self.attributes]
        return self.x_df

    def transform_x_to_dummies(self,attributes):
        '''
        :param attributes: the attributes that need to be transformed to dummies
        '''
        dummies_df_lst = []
        for item in attributes:
            dummies_item = pd.get_dummies(self.df[item],prefix=item)
            dummies_df_lst.append(dummies_item)

        #if self.key in (list(self.df.columns.values)):
        #    self.x_df = self.df.drop(self.key,axis=1,inplace=False)[self.attributes]
        #else:
        #    self.x_df = self.df[self.attributes]
        self.x_df = self.df[self.attributes]
        return self.x_df,dummies_df_lst

    def transform_x_to_binary(self,attributes):
        '''
        :param attributes: the attributes that need to be transformed to binary 
        '''
        for item in attributes:
            #self.df.loc[(self.df[item].isnull()),item] ='No'
            #self.df.loc[(self.df[item].notnull()),item]='Yes'
            self.df.loc[(self.df[item].isnull()),item] =0
            self.df.loc[(self.df[item].notnull()),item]=1

        #if self.key in (list(self.df.columns.values)):
        #    self.x_df = self.df.drop(self.key,axis=1,inplace=False)
        #else:
        #    self.x_df = self.df
        self.x_df = self.df[self.attributes]
        return self.x_df

    def china_area_number_mapping(self,attributes):
        '''
        :param attributes: the attributes that need to be transformed to area number 
        '''
        area_dict = {}
        try:
            for dirpath,dirnames,filenames in os.walk(self.resource_dir):
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
        #if self.key in (list(self.df.columns.values)):
        #    self.x_df = self.df.drop(self.key,axis=1,inplace=False)
        #else:
        #    self.x_df = self.df
        self.x_df = self.df[self.attributes]
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
        numberical_attrs,nonnumberical_attrs = self.filter_numberical_attrs()
        self.x_df,dummies_df_lst = self.transform_x_to_dummies(nonnumberical_attrs)
        for item in dummies_df_lst:
            self.x_df = pd.concat([self.x_df,item],axis=1)
        self.x_df.drop(nonnumberical_attrs,axis=1,inplace=True)
        self.df = pd.concat([self.x_df,self.df[self.key]],axis=1)
        self.x_df = self.set_x_null_attr_mean(list(self.x_df.columns),allnull=allnull,nullvalue=nullvalue)

        return self.x_df

def data_preprocess(data_path,form=0,attributes=None,all_labels=None,target_key=None,to_binary_attrs=None,area_attrs=None,discard_threshold=1.0,show=True,stats=True,stats_file_path='/tmp',test_size=0.3,cut_point=0,random_state=99):
    '''
    :param data_path: string, the data's path
    :param form: int, indicate that what type of data to process:
            0: data isn't split to train and test set, one whole file
            1: data has been split to train and test set, two files
            2: data is split to multiple directories, each directory exists one or multiple files
    :param attributes: list of string, labels of X
    :param all_labels: list of string, labels of data, including all X and y, even the label isn't used, should use when form=2
    :param target_key: string, label of target
    :param to_binary_attrs: list of string, attributes to be converted to binary value:1 or 0
    :param area_attrs: list of string, attributes about china area, like province, city, etc.
    :param discard_threshold: float, indicate that it will be discarded when one attribute's null value ratio > discard_threshold
    :param show: bool, indicate that the data's summary information should be printed
    :param stats: bool, indicate that the data's detail statistical information should be done
    :param stats_file_path: string, the path of statistical files
    :param test_size: float, the size ratio of test
    :param cut_point: int, indicate that which files are used to train model, which files are used to test model, it should be used when form=2
    :param random_state: int, random state
    :return: composite of DataFrame: X_train,X_test,y_train,y_test
    '''
    print("Performing data's preprocessing...")
    t = time()
    X_train = None
    y_train = None
    X_test = None
    y_test = None
    try:
        df,train,test = data_read.data_read(data_path,form,all_labels,test_size,cut_point,random_state)
        if attributes is None:
            attributes = list(df.columns.values)
        if show==True:
            print("BEFORE DATA PREPROCESS, SUMMARY INFORMATION OF THE DATA:")
        bef_file_path_stats = stats_file_path+'/'+'bef_data_statistics.csv'

        resource_dir = '../resources'
        datacheck = DataCheck(df,target_key)
        df = datacheck.check_type(show=show,stats=stats,file_path_stats=bef_file_path_stats)
        df = datacheck.check_value()
        df_datapreprocessing = DataPreprocessing(df,attributes,target_key,resource_dir)
        df = df_datapreprocessing.discard_trivial_attrs(null_ratio_threshold=discard_threshold)
        df = df_datapreprocessing.check_province_city_consistency('user_live_province','user_live_city')

        if to_binary_attrs is not None:
            df_datapreprocessing.transform_x_to_binary(to_binary_attrs)
            df_datapreprocessing.transform_x_dtype(to_binary_attrs,d_type=[int],uniform_type=True)
        if area_attrs is not None:
            df_datapreprocessing.china_area_number_mapping(area_attrs)
            df_datapreprocessing.transform_x_dtype(area_attrs,d_type=[int],uniform_type=True)
        X_df = df_datapreprocessing.x_dummies_and_fillna()

        X_train = X_df.loc[train.index.values,:]
        df_index_lst = df.index.values.tolist()
        train_index_lst = train.index.values.tolist()
        # If the index isn't continous, should use for-loop
        #for item in train_index_lst:
        #    df_index_lst.remove(item)
        X_test_index_lst = df_index_lst[len(train_index_lst):]
        X_test = X_df.loc[np.array(X_test_index_lst),:]
        X_test = X_test.reset_index(drop=True)

        y_train = train[target_key]
        y_test = test[target_key]

        X = pd.concat([X_train,X_test],axis=0)
        X = X.reset_index(drop=True)
        y = pd.concat([y_train,y_test],axis=0)
        y = y.reset_index(drop=True)
        df = pd.concat([X,y],axis=1)
        df = df.reset_index(drop=True)

        if show==True:
            print("AFTER DATA PREPROCESS, SUMMARY INFORMATION OF THE DATA:")
        aft_file_path_stats = stats_file_path+'/'+'aft_data_statistics.csv'
        datacheck = DataCheck(df,target_key)
        datacheck.data_summary(show=show,stats=stats,file_path_stats=aft_file_path_stats)
        print("Please see the statistical files %s and %s in directory %s if you want to see the detail information" % (bef_file_path_stats,aft_file_path_stats,stats_file_path))

    except Exception as e:
        print(e)
    finally:
        print("Data's preprocessing done in %0.3fs" % (time() - t))
        print()
        return X_train,X_test,y_train,y_test

