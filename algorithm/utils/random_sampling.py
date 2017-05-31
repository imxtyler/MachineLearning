#!/usr/bin/env python
# -*- coding:utf-8 -*-
import random
import bisect

'''
from http://www.cnblogs.com/zywscq/p/5469661.html
'''
def extend_weight_random(number_list,weight):
    '''
    :param number_list: the number list to choice
    :param weight: the weight list of the corresponding number_list
    :return: the number selected in the number_list 
    '''
    new_list = []
    for i,val in enumerate(number_list):
        new_list.extend(val*weight[i])
    return random.choice(new_list)

def weight_random(number_list,weight):
    '''
    :param number_list: the number list to choice
    :param weight: the weight list of the corresponding number_list
    :return: the number selected in the number_list 
    '''
    t = random.randint(0,sum(weight)-1)
    for i,val in enumerate(weight):
        t-=val
        if t<0:
            return number_list[i]

def sort_weight_random(number_list,weight):
    '''
    :param number_list: the number list to choice
    :param weight: the weight list of the corresponding number_list
    :return: the number selected in the number_list 
    '''
    weight_sum = []
    sum = 0
    for a in weight:
        sum += a
        weight_sum.append(sum)
    t = random.randint(0, sum - 1)
    return number_list[bisect.bisect_right(weight_sum, t)]

#if __name__ == "__main__":
#    #print(extend_weight_random(['A','B','C','D'],[5,2,2,1]))
#    #print(weight_random(['A','B','C','D'],[5,2,2,1]))
#    print(sort_weight_random(['A','B','C','D'],[5,2,2,1]))
