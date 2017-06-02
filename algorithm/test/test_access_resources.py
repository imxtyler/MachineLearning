#!/usr/bin/env python
#-*- coding:utf-8 -*-

import os

if __name__ == "__main__":
    area_dict = {}
    resource_dir = '../resources'
    for dirpath,dirnames,filenames in os.walk(resource_dir):
        for file in filenames:
            filepath = dirpath+'/'+file
            res_file = open(filepath,'r')
            while True:
                line = res_file.readline()
                if not line:
                    break
                area_number,area = line.strip().split(',')
                area_dict.setdefault(area,area_number)
            print(area_dict)
            #for item in area_dict.items():
            #    print(item)
            #for key,val in area_dict.items():
            #    print(str(key)+','+str(val))
