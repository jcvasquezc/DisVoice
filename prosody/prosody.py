
# -*- coding: utf-8 -*-
"""
Created on Jul 21 2017, Modified Apr 10 2018.

@author: J. C. Vasquez-Correa, T. Arias-Vergara, J. S. Guerrero


Compute prosody features from continuous speech based on duration, fundamental frequency and energy.



Static or dynamic matrices can be computed:

Static matrix is formed with 13 features and include

1. Average fundamental frequency in voiced segments
2. Standard deviation of fundamental frequency in Hz
3. Variablity of fundamental frequency in semitones
4. Maximum of the fundamental frequency in Hz
5. Average energy in dB
6. Standard deviation of energy in dB
7. Maximum energy
8. Voiced rate (number of voiced segments per second)
9. Average duration of voiced segments
10. Standard deviation of duratin of voiced segments
11. Pause rate (number of pauses per second)
12. Average duration of pauses
13. Standard deviation of duration of pauses
14. Average tilt of fundamental frequency
15. Tilt regularity of fundamental frequency
16. Mean square error of the reconstructed F0 with a  1-degree polynomial
17. (Silence duration)/(Voiced+Unvoiced durations)
18. (Voiced duration)/(Unvoiced durations)
19. (Unvoiced duration)/(Voiced+Unvoiced durations)
20. (Voiced duration)/(Voiced+Unvoiced durations)
21. (Voiced duration)/(Silence durations)
22. (Unvoiced duration)/(Silence durations)
23. Unvoiced duration Regularity
24. Unvoiced energy Regularity
25. Voiced duration Regularity
26. Voiced energy Regularity
27. Pause duration Regularity

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
from prosody_functions import V_UV, E_cont, logEnergy
import scipy.stats as st
import uuid

sys.path.append('../')
from utils import Hz2semitones
sys.path.append('../kaldi-io')
from kaldi_io import write_mat, write_vec_flt
sys.path.append('../praat')
import praat_functions

def plot_pros(data_audio,fs,F0,seg_voiced,Ev,featvec,f0v):
    plt.figure(1)
    plt.subplot(211)
    t=np.arange(0, float(len(data_audio))/fs, 1.0/fs)
    if len(t)!=len(data_audio):
        t=np.arange(1.0/fs, float(len(data_audio))/fs, 1.0/fs)
    print(len(t), len(data_audio))
    plt.plot(t, data_audio, 'k')
    plt.ylabel('Amplitude')
    plt.xlabel('Time (s)')
    plt.xlim([0, t[-1]])
    plt.grid(True)
    plt.subplot(212)
    fsp=len(F0)/t[-1]
    print(fsp)
    t2=np.arange(0.0, t[-1], 1.0/fsp)
    print(len(t2), len(F0))
    if len(t2)>len(F0):
        t2=t2[:len(F0)]
    elif len(F0)>len(t2):
        F0=F0[:len(t2)]
    plt.plot(t2, F0, color='k', linewidth=2.0)
    plt.xlabel('Time (s)')
    plt.ylabel('F0 (Hz)')
    plt.ylim([0,np.max(F0)+10])
    plt.xlim([0, t[-1]])
    plt.grid(True)
    plt.show()

    plt.figure(2)

    for j in range(len(seg_voiced)):
        plt.subplot(3, len(seg_voiced), j+1)
        t=np.arange(0, float(len(seg_voiced[j]))/fs, 1.0/fs)
        plt.plot(t, seg_voiced[j], linewidth=2.0)

        plt.subplot(3, len(seg_voiced), j+1+len(seg_voiced))
        plt.plot(f0v[j], linewidth=2.0, label="real")
        cf0=featvec[j][1:7]
        p = np.poly1d(cf0)
        estf0 = p(np.arange(len(f0v[j])))
        plt.plot(estf0, linewidth=2.0, label="estimated")
        if j==0:
            plt.ylabel("F0 (Hz)")

        plt.subplot(3, len(seg_voiced), j+1+2*len(seg_voiced))
        plt.plot(Ev[j], linewidth=2.0, label="real")

        cf0=featvec[j][8:]
        p = np.poly1d(cf0)
        estEv = p(np.arange(len(Ev[j])))
        plt.plot(estEv, linewidth=2.0, label="estimated")
        if j==0:
            plt.ylabel("Energy")
    plt.legend()
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
        temp_filename_vuv='../tempfiles/tempVUV'+temp_uuid+'.txt'
        temp_filename_f0='../tempfiles/tempF0'+temp_uuid+'.txt'
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



def prosody_static(audio, flag_plots):

    fs, data_audio=read(audio)
    data_audio=data_audio-np.mean(data_audio)
    data_audio=data_audio/float(np.max(np.abs(data_audio)))
    size_frameS=0.02*float(fs)
    size_stepS=0.01*float(fs)
    thr_len_pause=0.14*float(fs)
    thr_en_pause=0.2
    overlap=size_stepS/size_frameS
    nF=int((len(data_audio)/size_frameS/overlap))-1
    data_audiof=np.asarray(data_audio*(2**15), dtype=np.float32)
    F0=pysptk.sptk.rapt(data_audiof, fs, int(size_stepS), min=60, max=350, voice_bias=-0.2, otype='f0')

    logE=[]
    for l in range(nF):
        data_frame=data_audio[int(l*size_stepS):int(l*size_stepS+size_frameS)]
        logE.append(logEnergy(data_frame))
    logE=np.asarray(logE)
    print(sum(np.isnan(logE)))
    segmentsV=V_UV(F0, data_audio, fs, type_seg="Voiced", size_stepS=size_stepS)
    segmentsU=V_UV(F0, data_audio, fs, type_seg="Unvoiced", size_stepS=size_stepS)

    Nvoiced=len(segmentsV)
    Nunvoiced=len(segmentsU)

    Vrate=fs*float(Nvoiced)/len(data_audio)

    avgdurv=1000*np.mean([len(segmentsV[k]) for k in range(Nvoiced)])/float(fs)
    stddurv=1000*np.std([len(segmentsV[k]) for k in range(Nvoiced)])/float(fs)

    silence=[]
    for k in range(Nunvoiced):
        eu=logEnergy(segmentsU[k])
        if (eu<thr_en_pause or len(segmentsU[k])>thr_len_pause):
            silence.append(segmentsU[k])

    Silrate=fs*float(len(silence))/len(data_audio)

    avgdurs=1000*np.mean([len(silence[k]) for k in range(len(silence))])/float(fs)
    stddurs=1000*np.std([len(silence[k]) for k in range(len(silence))])/float(fs)

    if flag_plots:
        plt.figure(1)
        plt.subplot(311)
        t=np.arange(0, float(len(data_audio))/fs, 1.0/fs)
        if len(t)!=len(data_audio):
            t=np.arange(1.0/fs, float(len(data_audio))/fs, 1.0/fs)
        print(len(t), len(data_audio))
        plt.plot(t, data_audio, 'k')
        plt.ylabel('Amplitude')
        plt.xlabel('Time (s)')
        plt.xlim([0, t[-1]])
        plt.grid(True)
        plt.subplot(312)
        fsp=len(F0)/t[-1]
        print(fsp)
        t2=np.arange(0.0, t[-1], 1.0/fsp)
        if len(t2)>len(F0):
            t2=t2[:len(F0)]
        elif len(F0)>len(t2):
            F0=F0[:len(t2)]
        plt.plot(t2, F0, color='k', linewidth=2.0)
        plt.xlabel('Time (s)')
        plt.ylabel('F0 (Hz)')
        plt.ylim([0,np.max(F0)+10])
        plt.xlim([0, t[-1]])
        plt.grid(True)
        plt.subplot(313)
        fse=len(logE)/t[-1]
        t3=np.arange(0.0, t[-1], 1.0/fse)
        if len(t3)>len(logE):
            t3=t3[:len(logE)]
        elif len(logE)>len(t3):
            logE=logE[:len(t3)]
        plt.plot(t3, logE, color='k', linewidth=2.0)
        plt.xlabel('Time (s)')
        plt.ylabel('Energy (dB)')
        #plt.ylim([0,np.max(logE)])
        plt.xlim([0, t[-1]])
        plt.grid(True)
        plt.show()

    F0std=np.std(F0[F0!=0])
    F0varsemi=Hz2semitones(F0std**2)

    return F0, logE, np.mean(F0[F0!=0]), np.std(F0[F0!=0]), np.max(F0), np.mean(logE), np.std(logE), np.max(logE), Vrate, avgdurv, stddurv, Silrate, avgdurs, stddurs, F0varsemi

def intonation_duration(audio,size_step=0.01,minf0=60,maxf0=350,stol=0.150, flag_plots=False):
    fs, data_audio=read(audio)
    temp_filename_f0='../tempfiles/pitchtemp.txt'
    temp_filename_vuv='../tempfiles/voicetemp.txt'

    praat_functions.praat_vuv(audio,temp_filename_f0,temp_filename_vuv,time_stepF0=size_step, minf0=minf0, maxf0=maxf0)
    pitch_z,ttotal = praat_functions.decodeF0(temp_filename_f0,len(data_audio)/fs,size_step)

    #Slopes
    slopes=[]
    #buffers for voiced and unvoiced segments
    vbuffer=[]
    ubuffer=[]
    #energy for total voiced and unvoiced segments
    venergy=[]
    uenergy=[]
    #arrays for time-storing
    voicedtimes=[]
    unvoicedtimes=[]
    silencetimes=[]
    #flag for starting point voiced time and unvoiced time
    startvoicedflag=True
    startUNvoicedflag=True
    F0_rec=np.zeros(len(pitch_z))
    for i in range(0,len(pitch_z)-1):
        #condition for voiced segment
        if pitch_z[i]>=minf0 and pitch_z[i]<=maxf0:
            vbuffer.append(pitch_z[i])
            #voiced segment starting time
            if (startvoicedflag):
                t_start_venergy=ttotal[i]
                startvoicedflag=False
                frameF0start=i

            if len(ubuffer)!=0:
                samples = len(ubuffer)

                t=float(samples*size_step)#unvoiced time based on F0 Fs and actual samples
                #silence condition
                if t>stol:
                    silencetimes.append(t)
                else:
                    unvoicedtimes.append(t)

                #clear the mess
                ubuffer=[]
                #final time for unvoiced
                t_end_uenergy=ttotal[i]
                startUNvoicedflag=True
                #calculate segments with obtained times
                n_start_unvoiced=fs*t_start_uenergy
                n_end_unvoiced=fs*t_end_uenergy
                #energy of real audio segment based on fs and timestamp from F0
                #store
                uenergy.append(logEnergy(data_audio[int(n_start_unvoiced):int(n_end_unvoiced)]))
        #start appending unvoiced segments
        else:
            if(len(vbuffer)!=0):
                #based on F0 Fs and in buffer length, actual time is calculated
                samples = len(vbuffer)
                t=float(samples*size_step)
                #pick up voiced times
                voicedtimes.append(t)
                #voiced segment slope process
                #temporal x axis vector for slope calculation
                xtemp_slope=[]
                tempslope=np.array(vbuffer)
                for j in range(0,len(vbuffer)):
                    xtemp_slope.append(j)
                #get slopes of voiced segments

                pol=np.polyfit(xtemp_slope, tempslope,1)
                if np.isnan(pol[0]):
                    print("#################################")
                    print("detected short voiced segment")
                    #print(xtemp_slope, tempslope)
                else:
                    slopes.append(pol[0])
                #slopes.append(np.average(np.diff(tempslope)) / np.average(np.diff(xtemp_slope)))


                #clear the mess

                vbuffer=[]

                #final time of voiced segment
                t_end_venergy=ttotal[i]
                frameF0end=i
                if np.isnan(pol[0]):
                    F0_rec[int(frameF0start):int(frameF0end)]=tempslope
                else:
                    F0_rec[int(frameF0start):int(frameF0end)]=pol[0]*np.asarray(xtemp_slope)+pol[1]

                tempslope=[]
                xtemp_slope=[]
                startvoicedflag=True
                #calculate how many segments are in voiced time on the original audio file, based on start-end time stamps
                n_start_voiced=fs*t_start_venergy
                n_end_voiced=fs*t_end_venergy
                #calculate energy and make venergy append the result
                venergy.append(logEnergy(data_audio[int(n_start_voiced):int(n_end_voiced)]))
            else:
                ubuffer.append(pitch_z[i])
                #initial time of unvoiced segment
                if (startUNvoicedflag):
                    t_start_uenergy=ttotal[i]
                    startUNvoicedflag=False



    voicedtimes=np.array(voicedtimes)
    unvoicedtimes=np.array(unvoicedtimes)

    silencetimes=np.array(silencetimes)
    #print(unvoicedtimes, silencetimes)
    uenergy=np.array(uenergy)
    venergy=np.array(venergy)
    """Measures"""
    """Intonation"""
    avgF0slopes=np.average(slopes)# 1. average F0 slope
    stdF0slopes=np.std(slopes)# 2. std F0 slope
    """Duration"""
    if((silencetimes.size>0)):
        SVU=(np.sum(silencetimes))/(np.sum(voicedtimes)+np.sum(unvoicedtimes))#  3.S/(V+U)
    else:
        SVU=0
    VU=(np.sum(voicedtimes))/np.sum(unvoicedtimes)#  4.V/U
    UVU=np.sum(unvoicedtimes)/(np.sum(voicedtimes)+np.sum(unvoicedtimes))#  5.U/(V+U)
    VVU=np.sum(voicedtimes)/(np.sum(voicedtimes)+np.sum(unvoicedtimes))#  6.V/V+U
    #si no hay silencios hay que prevenir dividir por cero
    if((silencetimes.size>0)):
        VS=np.sum(voicedtimes)/np.sum(silencetimes)# 7. V/S
        US=np.sum(unvoicedtimes)/np.sum(silencetimes)# 8. U/S
    else:
        VS=0
        US=0

    URD=np.std(unvoicedtimes)# 9. (std U)
    VRD=np.std(voicedtimes)# 10. (std V)

    URE=np.std(uenergy) # 11. (std Energy U) wtf
    VRE=np.std(venergy) # 12. (std Energy V)
    MSEF0=np.mean((np.asarray(pitch_z)-np.asarray(F0_rec))**2)
    if ((silencetimes.size>0)):# 13. (std S)
        PR=np.std(silencetimes)
    else:
        PR=0

    os.remove(temp_filename_f0)
    os.remove(temp_filename_vuv)


    if flag_plots:
        plt.figure(1)
        plt.plot(ttotal, pitch_z, label="F0 (Hz)", linewidth=2.0)
        plt.plot(ttotal,F0_rec, label="Linear regresion F0", linewidth=2.0)
        plt.text(min(ttotal), max(pitch_z)-5, "MSE="+str(np.round(MSEF0,3)))
        plt.text(min(ttotal), max(pitch_z)-10, "Avg. tilt="+str(np.round(avgF0slopes,3)))
        plt.text(min(ttotal), max(pitch_z)-15, "Std. tilt="+str(np.round(stdF0slopes,3)))
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.legend()

        plt.grid(True)
        plt.show()


    return avgF0slopes,stdF0slopes,MSEF0, SVU,VU,UVU,VVU,VS,US,URD,VRD,URE,VRE,PR



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
            F0, logE, mF0, sF0, mmF0, mlogE, slogE, mmlogE, Vrate, avgdurv, stddurv, Silrate, avgdurs, stddurs, F0varsemi=prosody_static(audio_file, flag_plots)
            avgF0slopes, stdF0slopes, MSEF0, SVU , VU ,UVU ,VVU ,VS ,US ,URD,VRD ,URE,VRE ,PR = intonation_duration(audio_file, flag_plots=flag_plots)
            feat_vec=[mF0, sF0, F0varsemi, mmF0, mlogE, slogE, mmlogE, Vrate, avgdurv, stddurv, Silrate, avgdurs, stddurs, avgF0slopes, stdF0slopes, MSEF0, SVU , VU ,UVU ,VVU ,VS ,US ,URD,VRD ,URE,VRE ,PR]
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
            Features=np.asarray(Features)
            print(Features.shape)
            np.savetxt(file_features, Features)

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
            print(Features.shape)
            np.savetxt(file_features, Features)
            ID=np.hstack(ID)
            print(ID.shape)
            np.savetxt(file_features.replace('.txt', 'ID.txt'), ID, fmt='%i')
