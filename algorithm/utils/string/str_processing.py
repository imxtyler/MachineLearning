#!/usr/bin/env python
#-*- coding:utf-8 -*-

def regexp_extract(src_string,pattern,index):
    '''
    :param src_string: the source string
    :param pattern: the pattern in regexp
    :param index: the index in the pattern if match
    '''
    is_match = re.match(pattern=pattern, string=src_string)
    if(is_match):
        try:
            substr = is_match.group(index)
        except Exception as e:
            print(e)
    return substr

def regexp_replace(init_string,pattern,replacement):
    '''
    :param init_string: the initial string that to be processed
    :param pattern: the pattern in regexp
    :param replacement: the replaced string after matching
    '''
    try:
        is_match = re.match(pattern='|',string=init_string)
        if(is_match):
            pattern_list = str(pattern).split('|')
            for item in pattern_list:
                init_string = init_string.replace(item,replacement)
        else:
            init_string = init_string.replace(pattern,replacement)
    except Exception as e:
        print(e)
    replace_string = init_string

    return replace_string