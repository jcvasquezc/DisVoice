
# -*- coding: utf-8 -*-
"""
Created on Jul 21 2017

@author: J. C. Vasquez-Correa
"""


import numpy as np
import scipy.stats as st
import kaldi_io



def dynamic2static(feat):

    me=np.mean(feat,0)

    std=np.std(feat,0)
    sk=st.skew(feat)
    ku=st.kurtosis(feat)

    return np.hstack((me,std,sk,ku))

def dynamic2statict(feat):

    me=[]
    std=[]
    sk=[]
    ku=[]
    for k in feat:
        me.append(np.mean(k,0))
        std.append(np.std(k,0))
        sk.append(st.skew(k))
        ku.append(st.kurtosis(k))
    return np.hstack((me,std,sk,ku))


def dynamic2statict_artic(feat):

    me=[]
    std=[]
    sk=[]
    ku=[]
    for k in feat:
        if k.shape[0]>1:
            me.append(np.mean(k,0))
            std.append(np.std(k,0))
            sk.append(st.skew(k))
            ku.append(st.kurtosis(k))
        elif k.shape[0]==1:
            me.append(k[0,:])
            std.append(np.zeros(k.shape[1]))
            sk.append(np.zeros(k.shape[1]))
            ku.append(np.zeros(k.shape[1]))
        else:
            me.append(np.zeros(k.shape[1]))
            std.append(np.zeros(k.shape[1]))
            sk.append(np.zeros(k.shape[1]))
            ku.append(np.zeros(k.shape[1]))

    return np.hstack((np.hstack(me),np.hstack(std),np.hstack(sk),np.hstack(ku)))




def get_dict(feat_mat, IDs):
    uniqueids=np.unique(IDs)
    df={}
    for k in uniqueids:
        p=np.where(IDs==k)[0]
        featid=feat_mat[p,:]
        df[str(k)]=featid
    return df

def save_dict_kaldimat(dict_feat, temp_file):
    ark_scp_output='ark:| copy-feats --compress=true ark:- ark,scp:'+temp_file+'.ark,'+temp_file+'.scp'
    with kaldi_io.open_or_fd(ark_scp_output,'wb') as f:
        for key,mat in dict_feat.items(): 
            kaldi_io.write_mat(f, mat, key=key)

def multi_find(s, r):
    s_len = len(s)
    r_len = len(r)
    _complete = []
    if s_len < r_len:
        return -1
    for i in range(s_len):
        # search for r in s until not enough characters are left
        if s[i:i + r_len] == r:
            _complete.append(i)
        else:
            i = i + 1
    return _complete


def fill_when_empty(array):
    if len(array) == 0:
        return np.zeros((0,1))
    return array