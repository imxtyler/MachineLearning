#!/usr/bin/env python
#-*- coding:utf-8 -*-

import re
import pandas
import str_processing

if __name__ == "__main__":
    #pprint.pprint(sys.path)
    #file_fullpath = '/home/login01/Workspaces/python/dataset/module_data_app_hl_calls_stg1/app_hl_stg1.csv'
    file_fullpath = '/home/login01/Workspaces/python/dataset/module_data_stg2/md_data_stg2.csv'
    df = pandas.read_csv(file_fullpath,sep=',',na_values='NA',low_memory=False)
    #attributes=[
    #    "id",
    #    "method",
    #    #"object_id",
    #    #"true_name",
    #    #"id_card_no",
    #    #"phone",
    #    "status",
    #    #"create_time",
    #    #"update_time",
    #    #"real_name",
    #    #"idcard",
    #    "sex",
    #    "age",
    #    "in_net_time",
    #    #"gmt_create",
    #    #"reg_time",
    #    "allavgamt",
    #    "12avgamt",
    #    "9avgamt",
    #    "6avgamt",
    #    "3avgamt",
    #    "planavgamt",
    #    #"phone_balance",
    #    "12zhuavgcount",
    #    "12zhutime",
    #    "9zhuavgcount",
    #    "9zhutime",
    #    "6zhuavgcount",
    #    "6zhutime",
    #    "3zhuavgcount",
    #    "3zhutime",
    #    "12beiavgcount",
    #    "12beitime",
    #    "9beiavgcount",
    #    "9beitime",
    #    "6beiavgcount",
    #    "6beitime",
    #    "3beiavgcount",
    #    "3beitime",
    #    "12hutongavgcount",
    #    "12hutongtime",
    #    "9hutongavgcount",
    #    "9hutongtime",
    #    "6hutongavgcount",
    #    "6hutongtime",
    #    "3hutongavgcount",
    #    "3hutongtime",
    #    "12receiveavg",
    #    "9receiveavg",
    #    "6receiveavg",
    #    "3receiveavg",
    #    "12sendavg",
    #    "9sendavg",
    #    "6sendavg",
    #    "3sendavg",
    #    "12avgflow",
    #    "9avgflow",
    #    "6avgflow",
    #    "3avgflow",
    #    "12avgnettime",
    #    "9avgnettime",
    #    "6avgnettime",
    #    "3avgnettime",
    #    "12contactavgcount",
    #    "12contacttime",
    #    "9contactavgcount",
    #    "9contacttime",
    #    "6contactavgcount",
    #    "6contacttime",
    #    "3contactavgcount",
    #    "3contacttime",
    #    "12zhuplace",
    #    "9zhuplace",
    #    "6zhuplace",
    #    "3zhuplace",
    #    #"user_own_overdue", #y, target
    #    #"user_own_overdue_yet",
    #    #"user_own_fpd_overdue_order",
    #    #"user_own_ninety_overdue_order",
    #    #"user_own_sixty_overdue_order",
    #    #"user_own_thirty_overdue_order",
    #    #"user_own_ninety_overdue_num",
    #    #"user_own_sixty_overdue_num",
    #    #"user_own_thirty_overdue_num",
    #    #"user_credit_ninety_overdue",
    #    "1zhuplace"
    #]
    #target_key="user_own_overdue"

    attributes=[
        #"create_date",
        #"user_name",
        #"user_phone",
        "user_age",
        "user_sex",
        #"user_id_card",
        "user_live_province",
        "user_live_city",
        "user_live_address",
        #"user_regi_address",
        #"user_mailbox",
        "user_marriage",
        "user_rela_name",
        "user_relation",
        "user_rela_phone",
        "user_high_edu",
        #"user_indu_type",
        "user_company_name",
        #"user_company_phone",
        #"user_work_time",
        #"user_work_phone",
        "user_income_range",
        "user_last_consume",
        "user_ave_six_consume",
        "user_ave_twelve_consume",
        "user_house_mortgage",
        "user_car_mortgage",
        "user_base_fund",
        "user_credit_limit",
        "user_other_overdue",
        #"user_own_overdue", #y, target
        #"user_other_overdue_yet",
        "user_own_overdue_yet",
        "user_own_fpd_overdue_order",
        "user_own_ninety_overdue_order",
        "user_own_sixty_overdue_order",
        "user_own_thirty_overdue_order",
        "user_own_ninety_overdue_num",
        "user_own_sixty_overdue_num",
        "user_own_thirty_overdue_num",
        "user_credit_ninety_overdue",
        "user_loan_pass",
        "user_loan_amount",
        "user_four_ident",
        "user_face_ident",
        "user_base_fund_ident",
        "user_center_ident",
        "user_card_ident",
        "user_loan_ident"
    ]
    target_key="user_own_overdue"
    #print(df['user_income_range'].head(20))
    #print(df['user_last_consume'].head(20))
    df.convert_objects(convert_numeric=True)
    #df['user_last_consume'].str.replace('','')
    df.info()

    extract_str = str_processing.regexp_extract('foothebar','foo(.*?)(bar)',1)
    print(extract_str)
    replace_str = str_processing.regexp_replace('foobar','oo|ar','')
    print(replace_str)

