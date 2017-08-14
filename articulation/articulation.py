
# -*- coding: utf-8 -*-
"""
Created on Jul 21 2017

@author: J. C. Vasquez-Correa


Compute articulation features from continuous speech.

122 descriptors are computed:

1-22. Bark band energies in onset transitions (22 BBE).
23-34. Mel frequency cepstral coefficients in onset transitions (12 MFCC onset)
35-46. First derivative of the MFCCs in onset transitions (12 DMFCC onset)
47-58. Second derivative of the MFCCs in onset transitions (12 DDMFCC onset)
59-80. Bark band energies in offset transitions (22 BBE).
81-92. MFCCC in offset transitions (12 MFCC offset)
93-104. First derivative of the MFCCs in offset transitions (12 DMFCC offset)
105-116. Second derivative of the MFCCs in offset transitions (12 DMFCC offset)
117. First formant Frequency
118. First Derivative of the first formant frequency
119. Second Derivative of the first formant frequency
120. Second formant Frequency
121. First derivative of the Second formant Frequency
122. Second derivative of the Second formant Frequency


Static or dynamic matrices can be computed:

Static matrix is formed with 488 features formed with (122 descriptors) x (4 functionals: mean, std, skewness, kurtosis)

For dynamic analysis, two matrices are created for onset and offset based features

Dynamic matrices are formed with the 58 descriptors (22 BBEs, 12 MFCC, 12DMFCC, 12 DDMFCC ) computed for frames of 40 ms.

The first two frames of each recording are not considered for dynamic analysis to be able to stack the derivatives of MFCCs



Notes:
The fundamental frequency is used to detect the transitions and it is computed using the RAPT algorithm
The formant frequencies are computed using Praat

Script is called as follows

python articulation.py <file_or_folder_audio> <file_features.txt> [dynamic_or_static (default static)] [plots (true or false) (default false)]


examples:


python articulation.py "./001_ddk1_PCGITA.wav" "featuresDDKst.txt" "static" "true"
python articulation.py "./001_ddk1_PCGITA.wav" "featuresDDKdyn.txt" "dynamic" "true"
python articulation.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Rainbow/" "featuresDDKdynFolder.txt" "dynamic" "false"
python articulation.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Rainbow/" "featuresDDKstatFolder.txt" "static" "false"

"""


from scipy.io.wavfile import read
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import pysptk
import scipy.stats as st
from articulation_functions import V_UV, extractTrans, decodeFormants
import uuid


sys.path.append('../')
from utils import Hz2semitones

def plot_art(data_audio,fs,F0,F1,F2,segmentsOn,segmentsOff):
    plt.figure(1)
    plt.subplot(211)
    t=np.arange(0, float(len(data_audio))/fs, 1.0/fs)
    if len(t)>len(data_audio):
        t=t[:len(data_audio)]
    elif len(t)<len(data_audio):
        data_audio=data_audio[:len(t)]
    plt.plot(t, data_audio, 'k')
    plt.ylabel('Amplitude', fontsize=14)
    plt.xlim([0, t[-1]])
    plt.grid(True)

    plt.subplot(212)
    fsp=int(len(F1)/t[-1])
    t2=np.arange(0.0, t[-1], 1.0/fsp)
    if len(t2)>len(F1):
        t2=t2[:len(F1)]
    elif len(F1)>len(t2):
        F1=F1[:len(t2)]
        F2=F2[:len(t2)]
    plt.plot(t2, F1, color='k', linewidth=2.0, label='F1')
    plt.plot(t2, F2, color='g', linewidth=2.0, label='F2')

    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Frequency (Hz)', fontsize=14)
    plt.ylim([0,np.max(F2)+10])
    plt.xlim([0, t2[-1]])
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(2)
    plt.title("Onset segments")
    for j in range(len(segmentsOn)):
        plt.subplot(int(np.sqrt(len(segmentsOn)))+1, len(segmentsOn)/int(np.sqrt(len(segmentsOn))), j+1)
        t=np.arange(0, float(len(segmentsOn[j]))/fs, 1.0/fs)
        plt.plot(t, segmentsOn[j], linewidth=2.0)

    plt.show()

    plt.figure(3)
    plt.title("Offset segments")
    for j in range(len(segmentsOff)):
        plt.subplot(int(np.sqrt(len(segmentsOff)))+1, len(segmentsOff)/int(np.sqrt(len(segmentsOff))), j+1)
        t=np.arange(0, float(len(segmentsOff[j]))/fs, 1.0/fs)
        plt.plot(t, segmentsOff[j], linewidth=2.0)

    plt.show()


def articulation_continuous(audio_filename, flag_plots,sizeframe=0.04,step=0.02,nB=22,nMFCC=12,minf0=60,maxf0=350, voice_bias=-0.2,len_thr_miliseconds=270.0):

    fs, data_audio=read(audio_filename)
    data_audio=data_audio-np.mean(data_audio)
    data_audio=data_audio/float(np.max(np.abs(data_audio)))
    size_frameS=sizeframe*float(fs)
    size_stepS=step*float(fs)
    overlap=size_stepS/size_frameS
    data_audiof=np.asarray(data_audio*(2**15), dtype=np.float32)
    F0=pysptk.sptk.rapt(data_audiof, fs, int(size_stepS), min=minf0, max=maxf0, voice_bias=voice_bias, otype='f0')
    segmentsOn=V_UV(F0, data_audio, fs, 'onset')
    segmentsOff=V_UV(F0, data_audio, fs, 'offset')

    BBEon, MFCCon=extractTrans(segmentsOn, fs, size_frameS, size_stepS, nB, nMFCC)
    BBEoff, MFCCoff=extractTrans(segmentsOff, fs, size_frameS, size_stepS, nB, nMFCC)

    DMFCCon=np.asarray([np.diff(MFCCon[:,nf], n=1) for nf in range(MFCCon.shape[1])]).T
    DDMFCCon=np.asarray([np.diff(MFCCon[:,nf], n=2) for nf in range(MFCCon.shape[1])]).T

    DMFCCoff=np.asarray([np.diff(MFCCoff[:,nf], n=1) for nf in range(MFCCoff.shape[1])]).T
    DDMFCCoff=np.asarray([np.diff(MFCCoff[:,nf], n=2) for nf in range(MFCCoff.shape[1])]).T

    # TODO: Make parameters configurable. (If worth it)
    temp_uuid=str(uuid.uuid4().get_hex().upper()[0:6])
    temp_filename='./tempForm'+temp_uuid+'.txt'
    os.system('praat FormantsPraat.praat ' + audio_filename + ' ' + temp_filename +' 5 5500 '+str(float(sizeframe)/2)+' '+str(float(step))) #formant extraction praat
    [F1, F2]=decodeFormants(temp_filename)
    os.remove(temp_filename)

    if len(F0)<len(F1):
        F0=np.hstack((F0, np.zeros(len(F1)-len(F0))))
    else:
        F1=np.hstack((F1, np.zeros(len(F0)-len(F1))))
        F2=np.hstack((F2, np.zeros(len(F0)-len(F2))))

    pos0=np.where(F0==0)[0]
    dpos0=np.hstack(([1],np.diff(pos0)))
    f0u=np.split(pos0, np.where(dpos0>1)[0])

    # TODO: Why 270???

    thr_sil=int(len_thr_miliseconds/step)

    sil_seg=[]
    for l in range(len(f0u)):
        if len(f0u[l])>=thr_sil:
            F1[f0u[l]]=0
            F2[f0u[l]]=0
        sil_seg.append(f0u)

    sil_seg=np.hstack(sil_seg)

    F1nz=F1[F1!=0]
    F2nz=F2[F2!=0]
    DF1=np.diff(F1, n=1)
    DF2=np.diff(F2, n=1)
    DDF1=np.diff(F1, n=2)
    DDF2=np.diff(F2, n=2)

    if flag_plots:
        plot_art(data_audio,fs,F0,F1,F2,segmentsOn,segmentsOff)

    return BBEon, MFCCon, DMFCCon, DDMFCCon, BBEoff, MFCCoff, DMFCCoff, DDMFCCoff, F1nz, DF1, DDF1, F2nz, DF2, DDF2


if __name__=="__main__":

    if len(sys.argv)==5:
        audio=sys.argv[1]
        file_features=sys.argv[2]
        flag_static=sys.argv[3]
        if sys.argv[3]=="static" or sys.argv[3]=="dynamic":
            flag_static=sys.argv[3]
        else:
            print('python '+sys.argv[0]+' <file_or_folder_audio> <file_features.txt> [dynamic_or_static (default static)] [plots (true or false) (default false)]')
            sys.exit()
        if sys.argv[4]=="false" or sys.argv[4]=="False":
            flag_plots=False
        elif sys.argv[4]=="true" or sys.argv[4]=="True":
            flag_plots=True
        else:
            print('python '+sys.argv[0]+' <file_or_folder_audio> <file_features.txt> [dynamic_or_static (default static)] [plots (true or false) (default false)]')
            sys.exit()
    elif len(sys.argv)==4:
        audio=sys.argv[1]
        file_features=sys.argv[2]
        if sys.argv[3]=="static" or sys.argv[3]=="dynamic":
            flag_static=sys.argv[3]
        else:
            print('python '+sys.argv[0]+' <file_or_folder_audio> <file_features.txt> [dynamic_or_static (default static)] [plots (true or false) (default false)]')
            sys.exit()
        flag_plots=False
    elif len(sys.argv)==3:
        audio=sys.argv[1]
        file_features=sys.argv[2]
        flag_static="static"
        flag_plots=False
    elif len(sys.argv)<3:
        print('python '+sys.argv[0]+' <file_or_folder_audio> <file_features.txt> [dynamic_or_static (default static)] [plots (true or false) (default false)]')
        sys.exit()


    if audio.find('.wav')!=-1:
        nfiles=1
        hf=['']
    else:
        hf=os.listdir(audio)
        hf.sort()
        nfiles=len(hf)

    FeaturesOnset=[]
    IDon=[]
    FeaturesOffset=[]
    IDoff=[]
    Features=[]
    for k in range(nfiles):
        audio_file=audio+hf[k]
        print("Processing audio "+str(k+1)+ " from " + str(nfiles)+ " " +audio_file)

        BBEon, MFCCon, DMFCCon, DDMFCCon, BBEoff, MFCCoff, DMFCCoff, DDMFCCoff, F1, DF1, DDF1, F2, DF2, DDF2=articulation_continuous(audio_file, flag_plots)

        if flag_static=="static":
            Features_mean=np.hstack([BBEon.mean(0), MFCCon.mean(0), DMFCCon.mean(0), DDMFCCon.mean(0), BBEoff.mean(0), MFCCoff.mean(0), DMFCCoff.mean(0), DDMFCCoff.mean(0), F1.mean(0), DF1.mean(0), DDF1.mean(0), F2.mean(0), DF2.mean(0), DDF2.mean(0)])
            Features_std=np.hstack([BBEon.std(0), MFCCon.std(0), DMFCCon.std(0), DDMFCCon.std(0), BBEoff.std(0), MFCCoff.std(0), DMFCCoff.std(0), DDMFCCoff.std(0), F1.std(0), DF1.std(0), DDF1.std(0), F2.std(0), DF2.std(0), DDF2.std(0)])
            Features_sk=np.hstack([st.skew(BBEon), st.skew(MFCCon), st.skew(DMFCCon), st.skew(DDMFCCon), st.skew(BBEoff), st.skew(MFCCoff), st.skew(DMFCCoff), st.skew(DDMFCCoff), st.skew(F1), st.skew(DF1), st.skew(DDF1), st.skew(F2), st.skew(DF2), st.skew(DDF2)])
            Features_ku=np.hstack([st.kurtosis(BBEon, fisher=False), st.kurtosis(MFCCon, fisher=False), st.kurtosis(DMFCCon, fisher=False), st.kurtosis(DDMFCCon, fisher=False), st.kurtosis(BBEoff, fisher=False), st.kurtosis(MFCCoff, fisher=False), st.kurtosis(DMFCCoff, fisher=False), st.kurtosis(DDMFCCoff, fisher=False), st.kurtosis(F1, fisher=False), st.kurtosis(DF1, fisher=False), st.kurtosis(DDF1, fisher=False), st.kurtosis(F2, fisher=False), st.kurtosis(DF2, fisher=False), st.kurtosis(DDF2, fisher=False)])
            feat_vec=np.hstack((Features_mean, Features_std, Features_sk, Features_ku))
            Features.append(feat_vec)


        if flag_static=="dynamic":
            feat_onset=np.hstack((BBEon[2:,:], MFCCon[2:,:], DMFCCon[1:,:], DDMFCCon))
            IDson=np.ones(feat_onset.shape[0])*(k+1)
            FeaturesOnset.append(feat_onset)
            IDon.append(IDson)
            feat_offset=np.hstack((BBEoff[2:,:], MFCCoff[2:,:], DMFCCoff[1:,:], DDMFCCoff))
            IDsoff=np.ones(feat_offset.shape[0])*(k+1)
            FeaturesOffset.append(feat_offset)
            IDoff.append(IDsoff)

    if flag_static=="static":
        Features=np.asarray(Features)
        print(Features.shape)
        np.savetxt(file_features, Features)

    if flag_static=="dynamic":
        FeaturesOnset=np.vstack(FeaturesOnset)
        print(FeaturesOnset.shape)
        np.savetxt(file_features.replace('.txt', 'onset.txt'), FeaturesOnset)
        FeaturesOffset=np.vstack(FeaturesOffset)
        print(FeaturesOffset.shape)
        np.savetxt(file_features.replace('.txt', 'offset.txt'), FeaturesOffset)
        IDon=np.hstack(IDon)
        print(IDon.shape)
        IDoff=np.hstack(IDoff)
        print(IDoff.shape)
        np.savetxt(file_features.replace('.txt', 'IDonset.txt'), IDon, fmt='%i')
        np.savetxt(file_features.replace('.txt', 'IDoffset.txt'), IDoff, fmt='%i')
