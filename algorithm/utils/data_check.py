#coding=utf-8
'''
Created on 2017/6/6

@author: lenovo
'''
import pandas as pd
import numpy as np
import csv
import codecs
import re

''' this script is used to clean row data '''

class DataCheck():
    def __init__(self,info,file_path_stats):
        self.keys = info.columns
        self.info = info
        self.provinces = self.china_prov_city()
        self.file_path_stats = file_path_stats
    # do some statistics on data
    def value_counts(self,info,output=True):
        stats = {}
        for key in self.info.columns:
            #print type(self.info['user_marriage_nan'])
            stats[key] = self.info[key].value_counts(dropna=False)
        if output:
            MAX_LEN = max([len(stats[key].index) for key in self.info.columns])
            csvfile = open(self.file_path_stats, 'wb')
            csvfile.write(codecs.BOM_UTF8)
            writer = csv.writer(csvfile)
            header = []
            for key in self.info.columns:
                header.append(key)
                header.append('number')
            writer.writerow(header)
            for line in range(MAX_LEN):
                row = []
                for key in self.info.columns:
                    if line < len(stats[key]):
                        row.append(stats[key].index[line])
                        row.append(stats[key].values[line])
                    else:
                        row.append(' ')
                        row.append(' ')
                writer.writerow(row)
            csvfile.close()
        return stats
    # check data
    def check(self):
        def check_age(age):
            try:
                if np.isnan(age):
                    return age
                age = float(age)
                if age>100 and age<10:
                    return -100
                else:
                    return age
            except:
                return np.nan
        def check_name(name):
            try:
                if np.isnan(name):
                    return 0
            except:
                for ch in name.decode('utf-8'):
                    if u'\u4e00' <= ch <= u'\u9fff' or 'a'<=ch<='z':
                        return 1
                    else:
                        return 0
        def check_phone(phone):
            phone = str(phone)
            if re.match('^0\d{2,3}\d{7,8}$|^1[358]\d{9}$|^147\d{8}',phone):
                return 1
            else:
                return 0
        def check_sex(sex):
            if sex not in ['男','女']:
                return np.nan
            else:
                return sex
        def check_province_city(province_city):
            try:
                if np.isnan(province_city):
                    return np.nan
                else:
                    return np.nan
            except:
                for ch in province_city.decode('utf-8'):
                    if u'\u4e00' <= ch <= u'\u9fff' or 'a'<=ch<='z':
                        continue
                    else:
                        return np.nan
            return province_city  
        def check_marriage(marry):
            #print type(marry)
            if marry not in [1,2,0]:
                return np.nan
            else:
                return str(int(marry))
        def check_relation(rela):
            if rela not in ['配偶','父母','朋友','同事',
                            '同学','子女']:
                return np.nan
            return rela
        def check_high_edu(edu):
            if str(edu) not in '012345':
                return 0
            else:
                return edu
        def check_indu_type(indu_type):
            if indu_type not in ['制造业','物流运输','政府及事业单位',
                                 '建筑工程','软件及咨询服务','金融业',
                                 '住宿及旅游','零售业','餐饮业','其他']:
                return np.nan
            else:
                return indu_type
        def check_work_time(time):
            try:
                float(time)
                if np.isnan(time):
                    return time
                elif float(time)<0 or float(time)>50:
                    return np.nan
                else:
                    return time
            except:
                return np.nan
        def check_income(income_range):
            for i,income in enumerate(['5000-10000元','10000-15000元','5000元以下',
                                    '20000元以上','15000-20000元']):
                if income_range == income:
                    return i
            return 0                
        def check_amount(amount):
            try:
                if np.isnan(amount):
                    return 0 
            except: 
                pass
            try:
                float(amount)
                if amount<0:
                    return 0
                else:
                    return amount
            except:
                return 0
        def check_01(data_01):
            if data_01 not in [0,1]:
                return 0
            else:
                return data_01 
        def check_mail(mail):
            mail = str(mail)
            if re.match(r'^[0-9a-zA-Z_]{0,19}@[0-9a-zA-Z]{1,13}\.[com,cn,net]{1,3}$',mail):
                return 1
            else:
                return 0
        def check_idcard(idcard):
            if re.match(
                '^([1-9]\d{5}[12]\d{3}(0[1-9]|1[012])(0[1-9]|[12][0-9]|3[01])\d{3}[0-9xX])$',
                idcard):
                return 1
            else:
                return 0        
        def check_company(company):
            try:
                if np.isnan(company):
                    return 0
            except:
                return 1
        def check_all(self):
            self.info['user_sex'] = self.info['user_sex'].map(check_sex)
            age_sex_False = map(self.articul_compare,self.info['user_id_card'],self.info['user_sex'],self.info['user_age'],self.info['create_date'])
            age_sex_False = np.array(age_sex_False)
            self.info['age_False'] = pd.Series(age_sex_False[:,0])
            self.info['sex_False'] = pd.Series(age_sex_False[:,1])
            self.info['city_False'] = map(lambda province,city: self.province_city_compare(province, city), 
                                          self.info['user_live_province'],self.info['user_live_city'])            
            self.info['user_age'] = self.info['user_age'].map(check_age)
            self.info['user_marriage'] = self.info['user_marriage'].map(check_marriage)
            self.info['user_relation'] = self.info['user_relation'].map(check_relation)
            self.info['user_high_edu'] = self.info['user_high_edu'].map(check_high_edu)
            self.info['user_indu_type'] = self.info['user_indu_type'].map(check_indu_type)
            self.info['user_work_time'] = self.info['user_work_time'].map(check_work_time)
            self.info['user_income_range'] = self.info['user_income_range'].map(check_income)
            self.info['user_mailbox'] = self.info['user_mailbox'].map(check_mail)
            self.info['user_id_card'] = self.info['user_id_card'].map(check_idcard)
            self.info['user_company_name'] = self.info['user_company_name'].map(check_company)
            for key in ['user_name','user_rela_name']:
                self.info[key] = self.info[key].map(check_name)
            for key in ['user_phone','user_rela_phone',
                        'user_work_phone','user_company_phone']:
                self.info[key] = self.info[key].map(check_phone)
            for key in ['user_live_province','user_live_city']:
                self.info[key] = self.info[key].map(check_province_city)
            for key in ['user_last_consume','user_ave_six_consume',
                        'user_ave_twelve_consume','user_credit_limit',
                        'user_base_fund','user_loan_amount','user_own_ninety_overdue_num',
                        'user_own_sixty_overdue_num','user_own_thirty_overdue_num']:
                self.info[key] = self.info[key].map(check_amount)
            for key in ['user_house_mortgage','user_car_mortgage','user_other_overdue',
                        'user_own_overdue','user_other_overdue_yet','user_own_overdue_yet',
                        'user_own_fpd_overdue_order','user_own_ninety_overdue_order',
                        'user_own_sixty_overdue_order','user_own_thirty_overdue_order',
                        'user_credit_ninety_overdue','user_loan_pass','user_four_ident',
                        'user_face_ident','user_base_fund_ident','user_center_ident',
                        'user_card_ident','user_loan_ident']:
                self.info[key] = self.info[key].map(check_01)
                   
        check_all(self)
    # find conflict between idcard and sex,age
    def articul_compare(self,idcard,sex,age,create_date):  
        if sex == '男':
            sex = 1
        if sex == '女':
            sex = 0 
                 
        age_False = 0
        sex_False = 0
        try:
            if np.isnan(idcard):
                return -1,-1
        except:
            ID_birth=idcard[6:14]
            ID_sex=int(idcard[16])
            year=ID_birth[0:4]
        try:
            if np.isnan(sex):
                sex_False = -1
        except:
            if ID_sex%2!=sex:
                sex_False = 1
        if np.isnan(age):
            age_False = -1
        else:          
            create_year = create_date[:4]
            if abs((int(create_year)-int(year))-age)>2:
                age_False = 1   
        return [age_False,sex_False]
    # find conflict between province city
    def province_city_compare(self,province,city):
        city_pro_False = 0  
        try:
            if np.isnan(province):
                return city_pro_False 
        except:
            province = province.replace('省','').replace('市','').replace('自治区','')
            try:
                if np.isnan(city):
                    return city_pro_False
            except:
                city = city.replace('市','') 
                if len(city)>15:
                    return city_pro_False
                if city not in self.provinces[province]:
                    city_pro_False = 1
        return city_pro_False        
    def china_prov_city(self):
        f_pro = open('../data/CityFile/province/province.json')
        provinces = f_pro.readlines()[0].strip('[]').replace(r'"','').split(',')
        provinces = dict(zip(provinces[::2],provinces[1::2]))
        f_pro.close()
        for prov in provinces.keys():
            f_city = open('../data/CityFile/city/%s.json'%provinces[prov])
            provinces[prov] = f_city.readlines()[0].strip('[]').replace(r'"','').split(',')[::2]
            f_city.close()
        return provinces 
    def del_columns(self):
        stats = self.value_counts(self,output=True)
        for key in stats.keys():
            if len(stats[key])<2:
                del self.info[key]
        del_list = ['create_date','user_live_city','user_live_address','user_regi_address']
        for dl in del_list:
            del self.info[dl]          
    def dummies(self):
        dummies_list = ['user_live_province','user_marriage','user_relation','user_indu_type','user_sex']
        dummies = pd.get_dummies(self.info[dummies_list],dummy_na=True)
        for dum_l in dummies_list:
            #print(dum_l)
            del self.info[dum_l]
        self.info = pd.concat([self.info,dummies],axis=1)
