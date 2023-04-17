'''
Some functions and classes of tools used in project.
'''

import time

__all__ = ["get_time", "dict_to_str", "is_chinese"]

def get_time():
    localTime = time.localtime(time.time()) 
    strTime = '[' + time.strftime("%Y-%m-%d %H:%M:%S", localTime) + '] ' 
    return strTime


def dict_to_str(src_dict: dict[str, float]):
    """
    turn results to string, results' value will round to .4f
    """
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " %(key, src_dict[key]) 
    return dst_str

def is_chinese(uchar):
    """if unicode char is a chinese character"""
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    else:
        return False
                
