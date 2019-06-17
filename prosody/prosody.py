
# -*- coding: utf-8 -*-
"""
Created on Jul 21 2017, Modified Apr 10 2018.

@author: J. C. Vasquez-Correa, T. Arias-Vergara, J. S. Guerrero


Compute prosody features from continuous speech based on duration, fundamental frequency and energy.

Static or dynamic matrices can be computed:

Static matrix is formed with 103 features and include

Num     Feature                                                          Description
--------------------------------------------------------------------------------------------------------------------------
                                Features based on F0
---------------------------------------------------------------------------------------------------------------------------
1-6     F0-contour                                                       Avg., Std., Max., Min., Skewness, Kurtosis
7-12    Tilt of a linear estimation of F0 for each voiced segment        Avg., Std., Max., Min., Skewness, Kurtosis
13-18   MSE of a linear estimation of F0 for each voiced segment         Avg., Std., Max., Min., Skewness, Kurtosis
19-24   F0 on the first voiced segment                                   Avg., Std., Max., Min., Skewness, Kurtosis
25-30   F0 on the last voiced segment                                    Avg., Std., Max., Min., Skewness, Kurtosis
--------------------------------------------------------------------------------------------------------------------------
                                Features based on energy
---------------------------------------------------------------------------------------------------------------------------
31-34   energy-contour for voiced segments                               Avg., Std., Skewness, Kurtosis
35-38   Tilt of a linear estimation of energy contour for V segments     Avg., Std., Skewness, Kurtosis
39-42   MSE of a linear estimation of energy contour for V segment       Avg., Std., Skewness, Kurtosis
43-48   energy on the first voiced segment                               Avg., Std., Max., Min., Skewness, Kurtosis
49-54   energy on the last voiced segment                                Avg., Std., Max., Min., Skewness, Kurtosis
55-58   energy-contour for unvoiced segments                             Avg., Std., Skewness, Kurtosis
59-62   Tilt of a linear estimation of energy contour for U segments     Avg., Std., Skewness, Kurtosis
63-66   MSE of a linear estimation of energy contour for U segments      Avg., Std., Skewness, Kurtosis
67-72   energy on the first unvoiced segment                             Avg., Std., Max., Min., Skewness, Kurtosis
73-78   energy on the last unvoiced segment                              Avg., Std., Max., Min., Skewness, Kurtosis
--------------------------------------------------------------------------------------------------------------------------
                                Features based on duration
---------------------------------------------------------------------------------------------------------------------------
79      Voiced rate                                                      Number of voiced segments per second
80-85   Duration of Voiced                                               Avg., Std., Max., Min., Skewness, Kurtosis
86-91   Duration of Unvoiced                                             Avg., Std., Max., Min., Skewness, Kurtosis
92-97   Duration of Pauses                                               Avg., Std., Max., Min., Skewness, Kurtosis
98-103  Duration ratios                                                  Pause/(Voiced+Unvoiced), Pause/Unvoiced, Unvoiced/(Voiced+Unvoiced),
                                                                         Voiced/(Voiced+Unvoiced), Voiced/Puase, Unvoiced/Pause

Dynamic matrix is formed with 13 features computed for each voiced segment and contains

1. Duration of the voiced segment
2-7. Coefficients of 5-degree Lagrange polynomial to model F0 contour
8-13. Coefficients of 5-degree Lagrange polynomial to model energy contour

Dynamic prosody features are based on
Najim Dehak, "Modeling Prosodic Features With Joint Factor Analysis for Speaker Verification", 2007

Script is called as follows

python prosody.py <file_or_folder_audio> <file_features.txt> [dynamic_or_static (default static)] [plots (true or false) (default false)] [kaldi output (true or false) (default false)]

examples:

python prosody.py "./001_ddk1_PCGITA.wav" "featuresDDKst.txt" "static" "true"
python prosody.py "./001_ddk1_PCGITA.wav" "featuresDDKdyn.txt" "dynamic" "true"
python prosody.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Rainbow/" "featuresDDKdynFolder.txt" "dynamic" "false"
python prosody.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Rainbow/" "featuresDDKstatFolder.txt" "static" "false"
python prosody.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Rainbow/" "featuresDDKdynFolder.txt" "dynamic" "false" "true"

"""


from scipy.io.wavfile import read
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import pysptk
from prosody_functions import V_UV, E_cont, logEnergy, F0feat, energy_cont_segm, polyf0, energy_feat, dur_seg, duration_feat, E_cont
import scipy.stats as st
import uuid
from sklearn.metrics import mean_squared_error
import pandas as pd


path_app = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path_app+'/../')
sys.path.append(path_app+'/../praat')

from utils import Hz2semitones
import praat_functions

if "kaldi_io" in sys.modules:
    from kaldi_io import write_mat, write_vec_flt
    sys.path.append(path_app+'../kaldi-io')


def plot_pros(data_audio,fs,F0,segmentsV, segmentsU):
    plt.figure(figsize=(6,6))
    plt.subplot(211)
    ax1=plt.gca()
    t=np.arange(len(data_audio))/float(fs)
    ax1.plot(t, data_audio, 'k', label="speech signal", alpha=0.8)
    ax1.set_ylabel('Amplitude', fontsize=12)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_xlim([0, t[-1]])
    plt.grid(True)
    ax2 = ax1.twinx()
    fsp=len(F0)/t[-1]
    t2=np.arange(len(F0))/fsp
    ax2.plot(t2, F0, 'r', linewidth=2,label="F0")
    ax2.set_ylabel('Fundamental Frequency (Hz)', color='r', fontsize=12)
    ax2.tick_params('y', colors='r')

    plt.grid(True)

    plt.subplot(212)
    size_frameS=0.02*float(fs)
    size_stepS=0.01*float(fs)

    logE=energy_cont_segm([data_audio], fs,size_frameS, size_stepS)
    Esp=len(logE[0])/t[-1]
    t2=np.arange(len(logE[0]))/float(Esp)
    plt.plot(t2, logE[0], color='k', linewidth=2.0)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Energy (dB)', fontsize=14)
    plt.xlim([0, t[-1]])
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(6,3))
    F0rec=polyf0(F0, fs)
    t=np.arange(len(F0))*0.01
    plt.plot(t,F0, label="real F0")
    plt.plot(t,F0rec, label="estimated linear F0")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(6,3))
    Ev=energy_cont_segm(segmentsV, fs, size_frameS, size_stepS)
    Eu=energy_cont_segm(segmentsU, fs, size_frameS, size_stepS)

    plt.plot([np.mean(Ev[j]) for j in range(len(Ev))], label="Voiced energy")
    plt.plot([np.mean(Eu[j]) for j in range(len(Eu))], label="Unvoiced energy")

    plt.xlabel("Number of segments")
    plt.ylabel("Energy (dB)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def prosody_dynamic(audio, size_frame=0.03,size_step=0.01,minf0=60,maxf0=350, voice_bias=-0.2,energy_thr_percent=0.025,P=5, pitch_method='praat'):
    """
    Based on:
    Najim Dehak, "Modeling Prosodic Features With Joint Factor Analysis for Speaker Verification", 2007
    """
    fs, data_audio=read(audio)
    data_audio=data_audio-np.mean(data_audio)
    data_audio=data_audio/float(np.max(np.abs(data_audio)))
    size_frameS=size_frame*float(fs)
    size_stepS=size_step*float(fs)
    overlap=size_stepS/size_frameS
    nF=int((len(data_audio)/size_frameS/overlap))-1
    data_audiof=np.asarray(data_audio*(2**15), dtype=np.float32)
    if pitch_method == 'praat':
        name_audio=audio.split('/')
        temp_uuid='pros'+name_audio[-1][0:-4]
        temp_filename_vuv=path_app+'/../tempfiles/tempVUV'+temp_uuid+'.txt'
        temp_filename_f0=path_app+'/../tempfiles/tempF0'+temp_uuid+'.txt'
        praat_functions.praat_vuv(audio, temp_filename_f0, temp_filename_vuv, time_stepF0=size_step, minf0=minf0, maxf0=maxf0)
        F0,_=praat_functions.decodeF0(temp_filename_f0,len(data_audio)/float(fs),size_step)
        os.remove(temp_filename_vuv)
        os.remove(temp_filename_f0)
    elif pitch_method == 'rapt':
        F0=pysptk.sptk.rapt(data_audiof, fs, int(size_stepS), min=minf0, max=maxf0, voice_bias=voice_bias, otype='f0')

    #Find pitch contour of EACH voiced segment
    pitchON = np.where(F0!=0)[0]
    dchange = np.diff(pitchON)
    change = np.where(dchange>1)[0]
    iniV = pitchON[0]

    featvec = []
    iniVoiced = (pitchON[0]*size_stepS)+size_stepS#To compute energy
    seg_voiced=[]
    f0v=[]
    Ev=[]
    for indx in change:
        finV = pitchON[indx]+1
        finVoiced = (pitchON[indx]*size_stepS)+size_stepS#To compute energy
        VoicedSeg = data_audio[int(iniVoiced):int(finVoiced)]#To compute energy
        temp = F0[iniV:finV]
        tempvec = []
        if len(VoicedSeg)>int(size_frameS): #Take only segments greater than frame size
            seg_voiced.append(VoicedSeg)
            #Compute duration
            dur = len(VoicedSeg)/float(fs)
            tempvec.append(dur)
            #Pitch coefficients
            x = np.arange(0,len(temp))
            z = np.poly1d(np.polyfit(x,temp,P))
            f0v.append(temp)
            #fitCoeff.append(z.coeffs)
            tempvec.extend(z.coeffs)
            #Energy coefficients
            temp = E_cont(VoicedSeg,size_frameS,size_stepS,overlap)
            Ev.append(temp)
            x = np.arange(0,len(temp))
            z = np.poly1d(np.polyfit(x,temp,P))
            tempvec.extend(z.coeffs)
            featvec.append(tempvec)
        iniV= pitchON[indx+1]
        iniVoiced = (pitchON[indx+1]*size_stepS)+size_stepS#To compute energy

    #Add the last voiced segment
    finV = (pitchON[len(pitchON)-1])
    finVoiced = (pitchON[len(pitchON)-1]*size_stepS)+size_stepS#To compute energy
    VoicedSeg = data_audio[int(iniVoiced):int(finVoiced)]#To compute energy
    temp = F0[iniV:finV]
    tempvec = []
    if len(VoicedSeg)>int(size_frameS): #Take only segments greater than frame size
        #Compute duration
        dur = len(VoicedSeg)/float(fs)
        tempvec.append(dur)
        x = np.arange(0,len(temp))
        z = np.poly1d(np.polyfit(x,temp,P))
        tempvec.extend(z.coeffs)
        #Energy coefficients
        temp = E_cont(VoicedSeg,size_frameS,size_stepS,overlap)
        x = np.arange(0,len(temp))
        z = np.poly1d(np.polyfit(x,temp,P))
        tempvec.extend(z.coeffs)
        #Compute duration
        featvec.append(tempvec)



    if flag_plots:
        plot_pros(data_audio,fs,F0,seg_voiced,Ev,featvec,f0v)

    return np.asarray(featvec)



def prosody_static(audio, flag_plots, pitch_method='praat'):

    fs, data_audio=read(audio)
    data_audio=data_audio-np.mean(data_audio)
    data_audio=data_audio/float(np.max(np.abs(data_audio)))
    size_frameS=0.02*float(fs)
    size_stepS=0.01*float(fs)
    thr_len_pause=0.14*float(fs)
    thr_en_pause=10*np.log10(0.02)
    overlap=size_stepS/size_frameS
    nF=int((len(data_audio)/size_frameS/overlap))-1

    if pitch_method == 'praat':
        temp_uuid=audio.split('/')[-1][0:-4]
        temp_filename_f0=path_app+'/../tempfiles/tempF0'+temp_uuid+'.txt'
        temp_filename_vuv='../tempfiles/tempVUV'+temp_uuid+'.txt'
        praat_functions.praat_vuv(audio, temp_filename_f0, temp_filename_vuv, time_stepF0=0.01, minf0=60, maxf0=350)

        F0,_=praat_functions.decodeF0(temp_filename_f0,len(data_audio)/float(fs),0.01)
        os.remove(temp_filename_f0)

    elif pitch_method == 'rapt':
        data_audiof=np.asarray(data_audio*(2**15), dtype=np.float32)
        F0=pysptk.sptk.rapt(data_audiof, fs, int(size_stepS), min=60, max=350, voice_bias=-0.2, otype='f0')


    segmentsV=V_UV(F0, data_audio, fs, type_seg="Voiced", size_stepS=size_stepS)
    segmentsUP=V_UV(F0, data_audio, fs, type_seg="Unvoiced", size_stepS=size_stepS)

    segmentsP=[]
    segmentsU=[]
    for k in range(len(segmentsUP)):
        eu=logEnergy(segmentsUP[k])
        if (len(segmentsUP[k])>thr_len_pause):
            segmentsP.append(segmentsUP[k])
        else:
            segmentsU.append(segmentsUP[k])

    F0_features=F0feat(F0)
    energy_featuresV=energy_feat(segmentsV, fs, size_frameS, size_stepS)
    energy_featuresU=energy_feat(segmentsU, fs, size_frameS, size_stepS)

    duration_features=duration_feat(segmentsV, segmentsU, segmentsP, data_audio, fs)

    if flag_plots:

        plot_pros(data_audio,fs,F0,segmentsV, segmentsU)

    features=np.hstack((F0_features, energy_featuresV, energy_featuresU, duration_features))
    return features

if __name__=="__main__":
    prompt=' <file_or_folder_audio> <file_features.txt> '
    prompt+='[dynamic_or_static (default static)] '
    prompt+='[plots (true or false) (default false)]'
    prompt+='[kaldi output (true or false) (default false)]'
    if len(sys.argv)==6:
        audio=sys.argv[1]
        file_features=sys.argv[2]
        if sys.argv[3]=="static" or sys.argv[3]=="dynamic":
            flag_static=sys.argv[3]
        else:
            print('python '+sys.argv[0]+prompt)
            sys.exit()
        if sys.argv[4]=="false" or sys.argv[4]=="False":
            flag_plots=False
        elif sys.argv[4]=="true" or sys.argv[4]=="True":
            flag_plots=True
        else:
            print('python '+sys.argv[0]+' <file_or_folder_audio> <file_features.txt> [dynamic_or_static (default static)] [plots (true or false) (default false)]')
            sys.exit()
        if sys.argv[5]=="true" or sys.argv[5]=="True":
            flag_kaldi=True
        elif sys.argv[5]=="false" or sys.argv[5]=="False":
            flag_kaldi=False
        else:
            print('python '+sys.argv[0]+' <file_or_folder_audio> <file_features.txt> [dynamic_or_static (default static)] [plots (true or false) (default false)]')
            sys.exit()
    elif len(sys.argv)==5:
        audio=sys.argv[1]
        file_features=sys.argv[2]
        flag_static=sys.argv[3]
        flag_kaldi=False
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
        flag_kaldi=False
    elif len(sys.argv)==3:
        audio=sys.argv[1]
        file_features=sys.argv[2]
        flag_static="static"
        flag_plots=False
        flag_kaldi=False

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

    if flag_kaldi:
        Features={}

    else:
        Features=[]
        ID=[]
    for k in range(nfiles):
        audio_file=audio+hf[k]
        print("Processing audio "+str(k+1)+ " from " + str(nfiles)+ " " +audio_file)

        if flag_static=="static":
            
            feat_vec=prosody_static(audio_file, flag_plots, pitch_method='praat')
            feat_vec=np.asarray(feat_vec)

            p=np.where(np.isnan(feat_vec))[0]            
            if len(p)>0:
                feat_vec[p]=0
            if flag_kaldi:
                key=hf[k].replace('.wav', '')
                Features[key]=np.asarray(feat_vec)
            else:
                Features.append(feat_vec)

        if flag_static=="dynamic":
            profeats = prosody_dynamic(audio_file)
            if profeats.size == 0:
                continue
            if flag_kaldi:
                key=hf[k].replace('.wav', '')
                Features[key]=profeats
            else:
                Features.append(profeats)
                IDs=np.ones(profeats.shape[0])*(k+1)
                ID.append(IDs)

    if flag_static=="static":
        if flag_kaldi:
            temp_file='temp'+str(uuid.uuid4().get_hex().upper()[0:6])+'.ark'
            with open(temp_file,'wb') as f:
                for key in sorted(Features):
                    write_vec_flt(f, Features[key], key=key)
            ark_file=file_features.replace('.txt','')+'.ark'
            scp_file=file_features.replace('.txt','')+'.scp'
            os.system("copy-vector ark:"+temp_file+" ark,scp:"+ark_file+','+scp_file)
            os.remove(temp_file)
        else:
            
            Features=np.stack(Features, axis=0)
            if file_features.find('.txt')>=0:
                np.savetxt(file_features, Features)
            elif file_features.find('.csv')>=0:
                namefeatf0=["F0avg", "F0std", "F0max", "F0min", 
                    "F0skew", "F0kurt", "F0tiltavg", "F0mseavg", 
                    "F0tiltstd", "F0msestd", "F0tiltmax", "F0msemax", 
                    "F0tiltmin", "F0msemin","F0tiltskw", "F0mseskw", 
                    "F0tiltku", "F0mseku", "1F0mean", "1F0std", 
                    "1F0max", "1F0min", "1F0skw", "1F0ku", "lastF0avg", 
                    "lastF0std", "lastF0max", "lastF0min", "lastF0skw", "lastF0ku"]

                namefeatEv=["avgEvoiced", "stdEvoiced", "skwEvoiced", "kurtosisEvoiced", 
                            "avgtiltEvoiced", "stdtiltEvoiced", "skwtiltEvoiced", "kurtosistiltEvoiced", 
                            "avgmseEvoiced", "stdmseEvoiced", "skwmseEvoiced", "kurtosismseEvoiced", 
                            "avg1Evoiced", "std1Evoiced", "max1Evoiced", "min1Evoiced", "skw1Evoiced", 
                            "kurtosis1Evoiced", "avglastEvoiced", "stdlastEvoiced", "maxlastEvoiced", 
                            "minlastEvoiced", "skwlastEvoiced",  "kurtosislastEvoiced"]    


                namefeatEu=["avgEunvoiced", "stdEunvoiced", "skwEunvoiced", "kurtosisEunvoiced", 
                            "avgtiltEunvoiced", "stdtiltEunvoiced", "skwtiltEunvoiced", "kurtosistiltEunvoiced", 
                            "avgmseEunvoiced", "stdmseEunvoiced", "skwmseEunvoiced", "kurtosismseEunvoiced", 
                            "avg1Eunvoiced", "std1Eunvoiced", "max1Eunvoiced", "min1Eunvoiced", "skw1Eunvoiced", 
                            "kurtosis1Eunvoiced", "avglastEunvoiced", "stdlastEunvoiced", "maxlastEunvoiced", 
                            "minlastEunvoiced", "skwlastEunvoiced",  "kurtosislastEunvoiced"]  

                namefeatdur=["Vrate", "avgdurvoiced", "stddurvoiced", "skwdurvoiced", "kurtosisdurvoiced", "maxdurvoiced", "mindurvoiced", 
                            "avgdurunvoiced", "stddurunvoiced", "skwdurunvoiced", "kurtosisdurunvoiced", "maxdurunvoiced", "mindurunvoiced", 
                            "avgdurpause", "stddurpause", "skwdurpause", "kurtosisdurpause", "maxdurpause", "mindurpause", 
                            "PVU", "PU", "UVU", "VVU", "VP", "UP"]


                feat_names_all=namefeatf0+namefeatEv+namefeatEu+namefeatdur
                df={}
                df["ID"]=hf

                for k in range(Features.shape[1]):
                    df[feat_names_all[k]]=Features[:,k]

                df=pd.DataFrame(df) 
                df.to_csv(file_features)


    if flag_static=="dynamic":
        if flag_kaldi:
            temp_file='temp'+str(uuid.uuid4().get_hex().upper()[0:6])+'.ark'
            with open(temp_file,'wb') as f:
                for key in sorted(Features):
                    try:
                        write_mat(f, Features[key], key=key)
                    except Exception as e:
                        print ("Problem with key: {}. Shape of features is: {}".format(key,Features[key].shape))
                        raise
                        exit()
            ark_file=file_features.replace('.txt','')+'.ark'
            scp_file=file_features.replace('.txt','')+'.scp'
            os.system("copy-matrix ark:"+temp_file+" ark,scp:"+ark_file+','+scp_file)
            os.remove(temp_file)
        else:
            Features=np.vstack(Features)
            
            np.savetxt(file_features, Features)
            ID=np.hstack(ID)
            np.savetxt(file_features.replace('.txt', 'ID.txt'), ID, fmt='%i')
