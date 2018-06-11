#!/usr/bin/env python
#-*- coding:utf-8 -*-
import sys
import pandas as pd
import numpy as np
import preprocessing
import arno_linear_regression

if __name__ == "__main__":
    #file_path = "D:\\04analysis\\TQA-TQR\\result_trial_teacher_quality_trial_count_more1.csv"
    #file_path = "D:\\04analysis\\TQA-TQR\\result_trial_teacher_quality_trial_count_more2.csv"
    #file_path = "D:\\04analysis\\TQA-TQR\\result_trial_teacher_quality_trial_count_more3.csv"
    #file_path = "D:\\04analysis\\TQA-TQR\\result_trial_teacher_quality_trial_student_more6.csv"
    #file_path = "D:\\04analysis\\TQA-TQR\\result_trial_teacher_quality_trial_student_more10.csv"

    #file_path = "D:\\04analysis\\TQA-TCR\\new_result_trial_teacher_quality_trial_count_more1.csv"
    #file_path = "D:\\04analysis\\TQA-TCR\\new_result_trial_teacher_quality_trial_count_more2.csv"
    #file_path = "D:\\04analysis\\TQA-TCR\\new_result_trial_teacher_quality_trial_count_more3.csv"
    #file_path = "D:\\04analysis\\TQA-TCR\\new_result_trial_teacher_quality_trial_student_more6.csv"
    #file_path = "D:\\04analysis\\TQA-TCR\\new_result_trial_teacher_quality_trial_student_more10.csv"

    #file_path = "D:\\04analysis\\TQA-TCR\\teacher_main_channel_bigbear_cr_trial_student_number_more1.csv"
    #file_path = "D:\\04analysis\\TQA-TCR\\teacher_main_channel_bigbear_cr_trial_student_number_more2.csv"
    #file_path = "D:\\04analysis\\TQA-TCR\\teacher_main_channel_bigbear_cr_trial_student_number_more3.csv"
    file_path = "D:\\04analysis\\TQA-TCR\\teacher_main_channel_bigbear_cr_trial_student_number_more6.csv"
    #file_path = "D:\\04analysis\\TQA-TCR\\teacher_main_channel_bigbear_cr_trial_student_number_more10.csv"


    data_df = pd.read_csv(file_path, sep=',', na_values='NA', low_memory=False)
    attributes = ['id','total_course_count','qa_level','qa_score','language','attire','environment','instruction','feedback','student_engagement','tpr','props','efficient','cr']
    dataset = np.array(data_df)
    X = dataset[:,1:dataset.shape[1]-1]
    y = dataset[:,dataset.shape[1]-1]
    print('before normalization:')
    arno_linear_regression.analysis_corr(dataset[1:dataset.shape[1]],X,y,constant=False)

    e = pd.DataFrame(dataset[:,0:dataset.shape[1]-1],
           columns=attributes[0:dataset.shape[1]-1])
    print(e.info())
    attrs = e.columns.tolist()[1:dataset.shape[1]]
    data_preprocesssing = preprocessing.DataPreprocessing(e,attrs,None,None)
    data_preprocesssing.max_min_normalization(attrs)
    e = np.array(e)

    X1 = e[:,1:dataset.shape[1]-1]
    y1 = dataset[:,dataset.shape[1]-1]
    print('after normalization,do not add constant:')
    arno_linear_regression.analysis_corr(dataset[1:dataset.shape[1]],X1,y1,constant=False)
    print('after normalization,add constant:')
    arno_linear_regression.analysis_corr(dataset[1:dataset.shape[1]],X1,y1,constant=True)

    #plt.scatter(X2,y2,alpha=0.3)
    #plt.xlabel('language')
    #plt.ylabel('tqr')
    #plt.plot(X2,y2,'r',alpha=0.9)
    #plt.show()