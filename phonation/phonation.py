
# -*- coding: utf-8 -*-
"""
Created on Jul 21 2017

@author: J. C. Vasquez-Correa


Compute phonation features from sustained vowels and continuous speech.

For continuous speech, the features are computed over voiced segments

Seven descriptors are computed:

1. First derivative of the fundamental Frequency
2. Second derivative of the fundamental Frequency
3. Jitter
4. Shimmer
5. Amplitude perturbation quotient
6. Pitch perturbation quotient
7. Logaritmic Energy

Static or dynamic matrices can be computed:

Static matrix is formed with 29 features formed with (seven descriptors) x (4 functionals: mean, std, skewness, kurtosis) + degree of Unvoiced

Dynamic matrix is formed with the seven descriptors computed for frames of 40 ms.

Notes:

1. In dynamic features the first 11 frames of each recording are not considered to be able to stack the APQ and PPQ descriptors with the remaining ones.
2. The fundamental frequency is computed using Praat. To use the RAPT algorithm change the "pitch method" variable in the function phonation_vowel.
3. When Kaldi output is set to "true" two files will be generated, the ".ark" with the data in binary format and the ".scp" Kaldi script file.

Script is called as follows

python phonation.py <file_or_folder_audio> <file_features.txt> [dynamic_or_static (default static)] [plots (true or false) (default false)] [kaldi output (true or false) (default false)]

examples:

python phonation.py "./001_a1_PCGITA.wav" "featuresAst.txt" "static" "true"
python phonation.py "./001_a1_PCGITA.wav" "featuresAdyn.txt" "dynamic" "true"
python phonation.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Ah/" "featuresAdynFolder.txt" "dynamic" "false"
python phonation.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Ah/" "featuresAstatFolder.txt" "static" "false"
python phonation.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Ah/" "featuresAdynFolder.txt" "dynamic" "false" "true"

python phonation.py "./001_ddk1_PCGITA.wav" "featuresDDKst.txt" "static" "true"
python phonation.py "./001_ddk1_PCGITA.wav" "featuresDDKdyn.txt" "dynamic" "true"
python phonation.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Rainbow/" "featuresDDKdynFolder.txt" "dynamic" "false"
python phonation.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Rainbow/" "featuresDDKstatFolder.txt" "static" "false"
python phonation.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Rainbow/" "featuresDDKstatFolder.txt" "dynamice" "false" "true"

"""


from scipy.io.wavfile import read
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import pysptk
from phonation_functions import jitter_env, logEnergy, shimmer_env, APQ, PPQ
import scipy.stats as st
import uuid

path_app = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path_app+'/../')
from utils import Hz2semitones
sys.path.append(path_app+'/../kaldi-io')
from kaldi_io import write_mat, write_vec_flt

sys.path.append(path_app+'/../praat')
import praat_functions

def plot_phon(data_audio,fs,F0,logE):
    plt.figure(1)
    plt.subplot(311)
    t=np.arange(0, float(len(data_audio))/fs, 1.0/fs)
    if len(t)>len(data_audio):
        t=t[:len(data_audio)]
    elif len(t)<len(data_audio):
        data_audio=data_audio[:len(t)]
    plt.plot(t, data_audio, 'k')
    plt.ylabel('Amplitude', fontsize=14)
    plt.xlabel('Time (s)', fontsize=14)
    plt.xlim([0, t[-1]])
    plt.grid(True)

    plt.subplot(312)
    fsp=int(len(F0)/t[-1])
    t2=np.arange(0.0, t[-1], 1.0/fsp)
    if len(t2)>len(F0):
        t2=t2[:len(F0)]
    elif len(F0)>len(t2):
        F0=F0[:len(t2)]

    plt.plot(t2, F0, color='k', linewidth=2.0)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Frequency (Hz)', fontsize=14)
    plt.ylim([0,np.max(F0)+10])
    plt.xlim([0, t[-1]])
    plt.grid(True)
    #plt.show()
    plt.subplot(313)
    Esp=int(len(logE)/t[-1])
    t2=np.arange(0.0, t[-1], 1.0/Esp)
    if len(t2)>len(logE):
        t2=t2[:len(logE)]
    elif len(logE)>len(t2):
        logE=logE[:len(t2)]

    plt.plot(t2, logE, color='k', linewidth=2.0)
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Energy (dB)', fontsize=14)
    plt.ylim([0,np.max(logE)+10])
    plt.xlim([0, t[-1]])
    plt.grid(True)
    plt.show()

def phonationVowels(audio, flag_plots, size_frame=0.04,size_step=0.02,minf0=60,maxf0=350, voice_bias=-0.2,energy_thr_percent=0.025, pitch_method='praat'):



    fs, data_audio=read(audio)
    data_audio=data_audio-np.mean(data_audio)
    data_audio=data_audio/float(np.max(np.abs(data_audio)))
    size_frameS=size_frame*float(fs)
    size_stepS=size_step*float(fs)
    overlap=size_stepS/size_frameS
    if pitch_method == 'praat':
        name_audio=audio.split('/')
        temp_uuid='phon'+name_audio[-1][0:-4]
        if not os.path.exists('../tempfiles/'):
            os.makedirs('../tempfiles/')
        temp_filename_vuv='../tempfiles/tempVUV'+temp_uuid+'.txt'
        temp_filename_f0='../tempfiles/tempF0'+temp_uuid+'.txt'
        praat_functions.praat_vuv(audio, temp_filename_f0, temp_filename_vuv, time_stepF0=size_step, minf0=minf0, maxf0=maxf0)
        F0,_=praat_functions.decodeF0(temp_filename_f0,len(data_audio)/float(fs),size_step)
        #os.remove(temp_filename_vuv)
        #os.remove(temp_filename_f0)
    elif pitch_method == 'rapt':
        data_audiof=np.asarray(data_audio*(2**15), dtype=np.float32)
        F0=pysptk.sptk.rapt(data_audiof, fs, int(size_stepS), min=minf0, max=maxf0, voice_bias=voice_bias, otype='f0')
    F0nz=F0[F0!=0]
    Jitter=jitter_env(F0nz, len(F0nz))

    nF=int((len(data_audio)/size_frameS/overlap))-1
    Amp=[]
    logE=[]
    apq=[]
    ppq=[]

    DF0=np.diff(F0nz, 1)
    DDF0=np.diff(DF0,1)

    F0z=F0[F0==0]
    totaldurU=len(F0z)

    thresholdE=10*logEnergy([energy_thr_percent])
    degreeU=100*float(totaldurU)/len(F0)
    lnz=0
    for l in range(nF):
        data_frame=data_audio[int(l*size_stepS):int(l*size_stepS+size_frameS)]
        energy=10*logEnergy(data_frame)
        if F0[l]!=0:
            Amp.append(np.max(np.abs(data_frame)))
            logE.append(10*logEnergy(data_frame))
            if lnz>=12: # TODO:
                amp_arr=np.asarray([Amp[j] for j in range(lnz-12, lnz)])
                #print(amp_arr)
                apq.append(APQ(amp_arr))
            if lnz>=6: # TODO:
                f0arr=np.asarray([F0nz[j] for j in range(lnz-6, lnz)])
                ppq.append(PPQ(1/f0arr))
            lnz=lnz+1
        print("frame "+str(l) +" from "+str(nF)+"-"*int(100*l/nF)+">"+str(int(100*(l+1)/nF))+"%", sep=' ', end='\r', flush=True)

    Shimmer=shimmer_env(Amp, len(Amp))
    apq=np.asarray(apq)
    ppq=np.asarray(ppq)
    logE=np.asarray(logE)
    F0semi=np.asarray([Hz2semitones(F0nz[l]) for l in range(len(F0nz))])

    if flag_plots:
        plot_phon(data_audio,fs,F0,logE)

    print("Jitter=", len(Jitter))
    print("Shimmer", len(Shimmer))
    print("APQ", len(apq))
    print("PPQ", len(ppq))
    print("DF0", len(DF0))
    print("DDF0", len(DDF0))
    print("Energy", len(logE))
    print("degree unvoiced",degreeU)

    return F0, DF0, DDF0, F0semi, Jitter, Shimmer, apq, ppq, logE, degreeU


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
        F0, DF0, DDF0, F0semi, Jitter, Shimmer, apq, ppq, logE, degreeU  = phonationVowels(audio_file, flag_plots)

        if flag_static=="static":
            Features_mean=[DF0.mean(0), DDF0.mean(0), Jitter.mean(0), Shimmer.mean(0), apq.mean(0), ppq.mean(0), logE.mean(0)]
            Features_std=[DF0.std(0), DDF0.std(0), Jitter.std(0), Shimmer.std(0), apq.std(0), ppq.std(0), logE.std(0)]
            Features_sk=[st.skew(DF0), st.skew(DDF0), st.skew(Jitter), st.skew(Shimmer), st.skew(apq), st.skew(ppq), st.skew(logE)]
            Features_ku=[st.kurtosis(DF0, fisher=False), st.kurtosis(DDF0, fisher=False), st.kurtosis(Jitter, fisher=False), st.kurtosis(Shimmer, fisher=False), st.kurtosis(apq, fisher=False), st.kurtosis(ppq, fisher=False), st.kurtosis(logE, fisher=False)]
            feat_vec=np.hstack(([degreeU], Features_mean, Features_std, Features_sk, Features_ku))
            if flag_kaldi:
                if feat_vec.size>0:
                    key=hf[k].replace('.wav', '')
                    Features[key]=feat_vec
                else:
                    print ("Problem with file: {}".format(key))
            else:
                Features.append(feat_vec)

        if flag_static=="dynamic":
            feat_mat=np.vstack((DF0[11:], DDF0[10:], Jitter[12:], Shimmer[12:], apq, ppq[6:], logE[12:])).T
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
