
# -*- coding: utf-8 -*-
"""
Created on Jul 21 2017

@author: J. C. Vasquez-Correa
"""


import numpy as np

def Hz2semitones(freq):
    A4=440.
    C0=A4*2**(-4.75)
    if freq>0:
        h=12*np.log2(freq/C0)
        octave=h//12.
        return h+octave
    else:
        return C0


def multi_find(s, r):
    s_len = len(s)
    r_len = len(r)
    _complete = []
    if s_len < r_len:
        n = -1
    else:
        for i in range(s_len):
            # search for r in s until not enough characters are left
            if s[i:i + r_len] == r:
                _complete.append(i)
            else:
                i = i + 1
    return(_complete)
