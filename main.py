#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial transformer net
"""
__author__ = 'Taiki Horiuchi'
__version__ = '1.0'
__date__    = "2019/06/25 13:00"


#import numpy as np

class Net(object):
    """
    Netインスタンスを生成する。

    Attributes
    ----------
    INPUT_SIZE : int
        入力の大きさ。
        ex:)入力画像サイズが572×572の場合
            INPUT_SIZE = 572
    """
    
    def __init__(self, input_size):
        """
        入力画像サイズに合わせたUNetインスタンスを生成する。
        
        Parameters:
        ----------
        input_size : int
            入力の大きさ。
        """
        self.INPUT_SIZE = input_size
    
    def net_func():
        print("HelloWorld")


def func():
    _func()
    
    
def _func():
    print("HelloWorld")
    
    
if __name__ == '__main__':
    func()
