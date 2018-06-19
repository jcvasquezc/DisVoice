
# -*- coding: utf-8 -*-
"""
Created on Jul 21 2017

@author: J. C. Vasquez-Correa


Compute phonation features derived from the glottal source reconstruction from sustained vowels and continuous speech.

For continuous speech, the features are computed over voiced segments

Fourteen descriptors are computed:

1. Variability of time between consecutive glottal closure instants (GCI)
2. Average opening quotient (OQ) for consecutive glottal cycles-> rate of opening phase duration / duration of glottal cycle
3. Variability of opening quotient (OQ) for consecutive glottal cycles-> rate of opening phase duration /duration of glottal cycle
4. Average normalized amplitude quotient (NAQ) for consecutive glottal cycles-> ratio of the amplitude quotient and the duration of the glottal cycle
5. Variability of normalized amplitude quotient (NAQ) for consecutive glottal cycles-> ratio of the amplitude quotient and the duration of the glottal cycle
6. Average H1H2: Difference between the first two harmonics of the glottal flow signal
7. Variability H1H2: Difference between the first two harmonics of the glottal flow signal
8. Average of Harmonic richness factor (HRF): ratio of the sum of the harmonics amplitude and the amplitude of the fundamental frequency
9. Variability of HRF

Static or dynamic matrices can be computed:

Static matrix is formed with 36 features formed with (9 descriptors) x (4 functionals: mean, std, skewness, kurtosis)

Dynamic matrix is formed with the 9 descriptors computed for frames of 40 ms length.

Notes:

1. The fundamental frequency is computed using the RAPT algorithm.

2. When Kaldi output is set to "true" two files will be generated, the ".ark" with the data in binary format and the ".scp" Kaldi script file.

Script is called as follows

python glottal.py <file_or_folder_audio> <file_features.txt> [dynamic_or_static (default static)] [size frame analysis (s) (default 0.04)] [plots (true or false) (default false)] [kaldi output (true or false) (default false)]

examples:

python glottal.py "./001_a1_PCGITA.wav" "glottalfeaturesAst.txt" "static" "true" "false"
python glottal.py "./098_u1_PCGITA.wav" "glottalfeaturesUst.txt" "static" "true" "false"
python glottal.py "./001_a1_PCGITA.wav" "glottalfeaturesAdyn.txt" "dynamic" "true" "false"
python glottal.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Ah/" "glottalfeaturesAdynFolder.txt" "dynamic" "false" "false"
python glottal.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Ah/" "glottalfeaturesAstatFolder.txt" "static" "false" "false"
python glottal.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Ah/" "glottalfeaturesAdynFolder.ark" "dynamic" "false" "true"
"""

from __future__ import print_function


from scipy.io.wavfile import read
import os
import sys
sys.stdout.flush()
import numpy as np
import matplotlib.pyplot as plt
import math
import pysptk
import scipy.stats as st
import uuid
from peakdetect import peakdetect


sys.path.append('../')
from utils import Hz2semitones
sys.path.append('../kaldi-io')
from kaldi_io import write_mat, write_vec_flt
from GCI import SE_VQ_varF0, IAIF, get_vq_params
from scipy.integrate import cumtrapz


def plot_glottal(data_audio,fs,GCI, glottal_flow, glottal_sig, GCI_avg, GCI_std):
    plt.figure(1)
    plt.subplot(311)
    t=np.arange(0, float(len(data_audio))/fs, 1.0/fs)
    if len(t)>len(data_audio):
        t=t[:len(data_audio)]
    elif len(t)<len(data_audio):
        data_audio=data_audio[:len(t)]
    plt.plot(t, data_audio, 'k')
    plt.ylabel('Amplitude', fontsize=12)
    plt.xlim([0, t[-1]])
    plt.grid(True)

    plt.subplot(312)
    plt.plot(t[0:-1], glottal_sig, color='k', linewidth=2.0, label="Glottal flow signal")
    amGCI=[glottal_sig[int(k-2)] for k in GCI]

    GCI=GCI/fs
    plt.plot(GCI, amGCI, 'bo', alpha=0.5, markersize=8, label="GCI")

    plt.ylabel("Glottal flow", fontsize=12)
    plt.text(t[2],-0.8, "Avg. time consecutive GCI:"+str(np.round(GCI_avg*1000,2))+" ms")
    plt.text(t[2],-1.05, "Std. time consecutive GCI:"+str(np.round(GCI_std*1000,2))+" ms")
    plt.xlabel('Time (s)', fontsize=12)
    plt.xlim([0, t[-1]])
    plt.ylim([-1.1, 1.1])
    plt.grid(True)
    plt.legend()


    plt.subplot(313)
    plt.plot(t, glottal_flow, color='k', linewidth=2.0)
    plt.ylabel("Glotal flow derivative", fontsize=12)
    plt.xlabel('Time (s)', fontsize=12)
    plt.xlim([0, t[-1]])
    plt.grid(True)


    plt.show()

def glottal_features(audio, flag_plots, size_frame=0.2, size_step=0.1):

    fs, data_audio=read(audio)
    data_audio=data_audio-np.mean(data_audio)
    data_audio=data_audio/float(np.max(np.abs(data_audio)))
    size_frameS=size_frame*float(fs)
    size_stepS=size_step*float(fs)
    overlap=size_stepS/size_frameS
    nF=int((len(data_audio)/size_frameS/overlap))-1
    data_audiof=np.asarray(data_audio*(2**15), dtype=np.float32)
    f0=pysptk.sptk.rapt(data_audiof, fs, int(0.01*fs), min=20, max=500, voice_bias=-0.2, otype='f0')
    sizef0=int(size_frame/0.01)
    stepf0=int(size_step/0.01)
    startf0=0
    stopf0=sizef0
    avgGCIt=np.zeros(nF)
    varGCIt=np.zeros(nF)
    avgNAQt=np.zeros(nF)
    varNAQt=np.zeros(nF)
    avgQOQt=np.zeros(nF)
    varQOQt=np.zeros(nF)
    avgH1H2t=np.zeros(nF)
    varH1H2t=np.zeros(nF)
    avgHRFt=np.zeros(nF)
    varHRFt=np.zeros(nF)
    rmwin=[]
    for l in range(nF):
        data_frame=data_audio[int(l*size_stepS):int(l*size_stepS+size_frameS)]
        f0_frame=f0[startf0:stopf0]
        pf0framez=np.where(f0_frame!=0)[0]
        f0nzframe=f0_frame[pf0framez]
        if len(f0nzframe)<10:
            startf0=startf0+stepf0
            stopf0=stopf0+stepf0
            rmwin.append(l)
            print("frame "+str(l) +" from "+str(nF)+"-"*int(100*l/nF)+">"+str(int(100*(l+1)/nF))+"%", sep=' ', end='\r')

            continue
        GCI=SE_VQ_varF0(data_frame,fs, f0=f0_frame)
        g_iaif=IAIF(data_frame,fs,GCI)
        g_iaif=g_iaif-np.mean(g_iaif)
        g_iaif=g_iaif/max(abs(g_iaif))
        glottal=cumtrapz(g_iaif)
        glottal=glottal-np.mean(glottal)
        glottal=glottal/max(abs(glottal))
        startf0=startf0+stepf0
        stopf0=stopf0+stepf0

        gci_s=GCI[:]
        GCId=np.diff(gci_s)
        avgGCIt[l]=np.mean(GCId/fs)
        varGCIt[l]=np.std(GCId/fs)
        NAQ, QOQ, T1, T2, H1H2, HRF=get_vq_params(glottal, g_iaif, fs, GCI)
        avgNAQt[l]=np.mean(NAQ)
        varNAQt[l]=np.std(NAQ)
        avgQOQt[l]=np.mean(QOQ)
        varQOQt[l]=np.std(QOQ)
        avgH1H2t[l]=np.mean(H1H2)
        varH1H2t[l]=np.std(H1H2)
        avgHRFt[l]=np.mean(HRF)
        varHRFt[l]=np.std(HRF)
        print("frame "+str(l) +" from "+str(nF)+"-"*int(100*l/nF)+">"+str(int(100*(l+1)/nF))+"%", sep=' ', end='\r')
        if flag_plots:
            plot_glottal(data_frame,fs,GCI, g_iaif, glottal, avgGCIt[l], varGCIt[l])


    if len(rmwin)>0:
        varGCI=np.delete(varGCIt,rmwin)
        avgNAQ=np.delete(avgNAQt,rmwin)
        varNAQ=np.delete(varNAQt,rmwin)
        avgQOQ=np.delete(avgQOQt,rmwin)
        varQOQ=np.delete(varQOQt,rmwin)
        avgH1H2=np.delete(avgH1H2t,rmwin)
        varH1H2=np.delete(varH1H2t,rmwin)
        avgHRF=np.delete(avgHRFt,rmwin)
        varHRF=np.delete(varHRFt,rmwin)
        return varGCI, avgNAQ, varNAQ, avgQOQ, varQOQ, avgH1H2, varH1H2, avgHRF, varHRF
    else:
        return varGCIt, avgNAQt, varNAQt, avgQOQt, varQOQt, avgH1H2t, varH1H2t, avgHRFt, varHRFt


if __name__=="__main__":
    promtp=' <file_or_folder_audio> <file_features.txt> '
    promtp+='[dynamic_or_static (default static)] '
    promtp+='[plots (true or false) (default false)] '
    promtp+='[kaldi output (true or false) (default false)]'
    if len(sys.argv)==6:
        audio=sys.argv[1]
        file_features=sys.argv[2]
        if sys.argv[3]=="static" or sys.argv[3]=="dynamic":
            flag_static=sys.argv[3]
        else:
            print('python '+sys.argv[0]+promtp)
            sys.exit()
        if sys.argv[4]=="false" or sys.argv[4]=="False":
            flag_plots=False
        elif sys.argv[4]=="true" or sys.argv[4]=="True":
            flag_plots=True
        else:
            print('python '+sys.argv[0]+promtp)
            sys.exit()
        if sys.argv[5]=="true" or sys.argv[5]=="True":
            flag_kaldi=True
        elif sys.argv[5]=="false" or sys.argv[5]=="False":
            flag_kaldi=False
        else:
            print('python '+sys.argv[0]+promtp)
            sys.exit()
    elif len(sys.argv)==5:
        flag_kaldi=False
        audio=sys.argv[1]
        file_features=sys.argv[2]
        if sys.argv[3]=="static" or sys.argv[3]=="dynamic":
            flag_static=sys.argv[3]
        else:
            print('python '+sys.argv[0]+promtp)
            sys.exit()
        if sys.argv[4]=="false" or sys.argv[4]=="False":
            flag_plots=False
        elif sys.argv[4]=="true" or sys.argv[4]=="True":
            flag_plots=True
        else:
            print('python '+sys.argv[0]+promtp)
            sys.exit()
    elif len(sys.argv)==4:
        audio=sys.argv[1]
        file_features=sys.argv[2]
        if sys.argv[3]=="static" or sys.argv[3]=="dynamic":
            flag_static=sys.argv[3]
        else:
            print('python '+sys.argv[0]+promtp)
            sys.exit()
        flag_plots=False
        flag_kaldi=False
    elif len(sys.argv)==3:
        audio=sys.argv[1]
        file_features=sys.argv[2]
        flag_static="static"
        flag_plots=False
        flag_kaldi=False
    elif len(sys.argv)<3:
        print('python '+sys.argv[0]+promtp)
        sys.exit()


    if audio.find('.wav')!=-1 or audio.find('.WAV')!=-1:
        nfiles=1
        hf=['']
    else:
        hf=os.listdir(audio)
        hf.sort()
        nfiles=len(hf)
    if flag_kaldi:
        Features={} # Kaldi Output requires a dictionary
    else:
        Features=[]
        ID=[]
    # For every file in the audio folder
    for k in range(nfiles):
        audio_file=audio+hf[k]
        print("Processing audio "+str(k+1)+ " from " + str(nfiles)+ " " +audio_file)
        varGCIt, avgNAQt, varNAQt, avgQOQt, varQOQt, avgH1H2t, varH1H2t, avgHRFt, varHRFt  = glottal_features(audio_file, flag_plots)


        if flag_static=="static":
            Features_mean=[varGCIt.mean(0), avgNAQt.mean(0), varNAQt.mean(0), avgQOQt.mean(0), varQOQt.mean(0), avgH1H2t.mean(0), varH1H2t.mean(0), avgHRFt.mean(0), varHRFt.mean(0)]
            Features_std=[varGCIt.std(0), avgNAQt.std(0), varNAQt.std(0), avgQOQt.std(0), varQOQt.std(0), avgH1H2t.std(0), varH1H2t.std(0), avgHRFt.std(0), varHRFt.std(0)]
            Features_sk=[st.skew(varGCIt), st.skew(avgNAQt), st.skew(varNAQt), st.skew(avgQOQt), st.skew(varQOQt), st.skew(avgH1H2t), st.skew(varH1H2t), st.skew(avgHRFt), st.skew(varHRFt)]
            Features_ku=[st.kurtosis(varGCIt, fisher=False), st.kurtosis(avgNAQt, fisher=False), st.kurtosis(varNAQt, fisher=False), st.kurtosis(avgQOQt, fisher=False), st.kurtosis(varQOQt, fisher=False), st.kurtosis(avgH1H2t, fisher=False), st.kurtosis(varH1H2t, fisher=False), st.kurtosis(avgHRFt, fisher=False), st.kurtosis(varHRFt, fisher=False)]
            feat_vec=np.hstack(([Features_mean, Features_std, Features_sk, Features_ku]))
            if flag_kaldi:
                if feat_vec.size>0:
                    key=hf[k].replace('.wav', '')
                    Features[key]=feat_vec
                else:
                    print ("Problem with file: {}".format(key))
            else:
                Features.append(feat_vec)

        if flag_static=="dynamic":
            feat_mat=np.vstack((varGCIt, avgNAQt, varNAQt, avgQOQt, varQOQt, avgH1H2t, varH1H2t, avgHRFt, varHRFt)).T
            print(feat_mat.shape)
            IDs=np.ones(feat_mat.shape[0])*(k+1)
            if flag_kaldi:
                if feat_mat.size>0:
                    key=hf[k].replace('.wav', '')
                    Features[key]=feat_mat
                else:
                    print ("Problem with file: {}".format(key))
            else:
                Features.append(feat_mat)
                ID.append(IDs)
    # Once the features of all files have been extracted save them
    # TODO: Save them within the loop, waiting for them to finish will become an issue later
    if flag_static=="static":
        if flag_kaldi:
            temp_file='temp_static'+file_features[:-4]+'.ark'
            with open(temp_file,'wb') as f:
                for key in sorted(Features):
                    write_vec_flt(f, Features[key], key=key)
            ark_file=file_features.replace('.txt','')+'.ark'
            scp_file=file_features.replace('.txt','')+'.scp'
            os.system("copy-vector ark:"+temp_file+" ark,scp:"+ark_file+','+scp_file)
            os.remove(temp_file)
        else:
            Features=np.asarray(Features)
            np.savetxt(file_features, Features)

    if flag_static=="dynamic":
        if flag_kaldi:
            temp_file='temp_dynamic'+file_features[:-4]+'.ark'
            with open(temp_file,'wb') as f:
                for key in sorted(Features):
                    write_mat(f, Features[key], key=key)
            ark_file=file_features.replace('.txt','')+'.ark'
            scp_file=file_features.replace('.txt','')+'.scp'
            os.system("copy-matrix ark:"+temp_file+" ark,scp:"+ark_file+','+scp_file)
            os.remove(temp_file)
        else:
            Features=np.vstack(Features)
            np.savetxt(file_features, Features)
            ID=np.hstack(ID)
            np.savetxt(file_features.replace('.txt', 'ID.txt'), ID, fmt='%i')
