
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
The fundamental frequency is used to detect the transitions and it is computed using Praat by default. To use the RAPT algorithm change the "pitch method" variable in the function articulation_continuous.

The formant frequencies are computed using Praat.

When Kaldi output is set to "true" two files will be generated, the ".ark" with the data in binary format and the ".scp" Kaldi script file.

Script is called as follows

python articulation.py <file_or_folder_audio> <file_features.txt> [dynamic_or_static (default static)] [plots (true or false) (default false)] [kaldi output (true or false) (default false)]


examples:


python articulation.py "./001_ddk1_PCGITA.wav" "featuresDDKst.txt" "static" "true"
python articulation.py "./001_ddk1_PCGITA.wav" "featuresDDKdyn.txt" "dynamic" "true"
python articulation.py "./001_ddk1_PCGITA.wav" "featuresDDKdyn.txt" "dynamic" "true" "true"
python articulation.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Rainbow/" "featuresDDKdynFolder.txt" "dynamic" "false"
python articulation.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Rainbow/" "featuresDDKstatFolder.txt" "static" "false"
python articulation.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Rainbow/" "featuresDDKstatFolder.txt" "dynamic" "false" "true"

"""


from scipy.io.wavfile import read
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math
import pysptk
import scipy.stats as st
from articulation_functions import extractTrans, V_UV
import uuid
import pandas as pd


path_app = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path_app+'/../')

sys.path.append(path_app+'/../praat')

import praat_functions

def plot_art(data_audio,fs,F0,F1,F2,segmentsOn,segmentsOff):
    plt.figure(1)
    plt.subplot(311)
    t=np.arange(0, float(len(data_audio))/fs, 1.0/fs)
    if len(t)>len(data_audio):
        t=t[:len(data_audio)]
    elif len(t)<len(data_audio):
        data_audio=data_audio[:len(t)]
    plt.plot(t, data_audio, 'k')
    plt.ylabel('Amplitude', fontsize=14)
    plt.xlim([0, t[-1]])
    plt.grid(True)

    plt.subplot(312)
    t0=np.linspace(0.0,t[-1],len(F0))
    plt.plot(t0, F0, color='r', linewidth=2.0, label='F0')
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('Frequency (Hz)', fontsize=14)
    plt.ylim([0,np.max(F0)+10])
    plt.xlim([0, t0[-1]])
    plt.grid(True)
    plt.legend()

    plt.subplot(313)
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
    plt.tight_layout()    
    plt.show()
    plt.close()



    for j in range(len(segmentsOn)):
        f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1,3]}, sharex=True, figsize=(5,4))
        t=np.arange(len(segmentsOn[j]))/fs
        a0.plot(t[0:640], segmentsOn[j][0:640], color='k', label="unvoiced")
        a0.plot(t[640:], segmentsOn[j][640:], color='k', alpha=0.5, label="voiced")
        a0.grid()
        a0.legend(loc=2, ncol=2 )
        spec, freqs, t, im=plt.specgram(segmentsOn[j], NFFT=128, Fs=fs, window=mlab.window_hanning, noverlap=100, detrend=mlab.detrend_mean)
        a1.imshow(np.log10(np.abs(np.flipud(spec))), extent=[0, .08, 1, 4000], aspect='auto', cmap=plt.cm.jet,
                vmax=np.log10(np.abs(spec).max()), vmin=np.log10(np.abs(spec).min()), interpolation="bilinear")
        a1.set_ylabel('Frequency (Hz)', fontsize=12)
        a1.set_xlabel('Time (s)', fontsize=12)
        plt.tight_layout()
        plt.show()



    for j in range(len(segmentsOff)):
        f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1,3]}, sharex=True, figsize=(5,4))
        t=np.arange(len(segmentsOff[j]))/fs
        a0.plot(t[0:640], segmentsOff[j][0:640], color='k', label="voiced")
        a0.plot(t[640:], segmentsOff[j][640:], color='k', alpha=0.5, label="unvoiced")
        a0.grid()
        a0.legend(loc=1, ncol=2 )
        spec, freqs, t, im=plt.specgram(segmentsOff[j], NFFT=128, Fs=fs, window=mlab.window_hanning, noverlap=100, detrend=mlab.detrend_mean)
        a1.imshow(np.log10(np.abs(np.flipud(spec))), extent=[0, .08, 1, 4000], aspect='auto', cmap=plt.cm.jet,
                vmax=np.log10(np.abs(spec).max()), vmin=np.log10(np.abs(spec).min()), interpolation="bilinear")
        a1.set_ylabel('Frequency (Hz)', fontsize=12)
        a1.set_xlabel('Time (s)', fontsize=12)
        plt.tight_layout()
        plt.show()



def articulation_continuous(audio_filename, flag_plots,sizeframe=0.04,step=0.02,nB=22,nMFCC=12,minf0=60,maxf0=350, voice_bias=-0.5,len_thr_miliseconds=270.0, pitch_method='praat'):

    fs, data_audio=read(audio_filename)

    data_audio=data_audio-np.mean(data_audio)
    data_audio=data_audio/float(np.max(np.abs(data_audio)))
    size_frameS=sizeframe*float(fs)
    size_stepS=step*float(fs)
    overlap=size_stepS/size_frameS

    if pitch_method == 'praat':
        name_audio=audio_filename.split('/')
        temp_uuid='artic'+name_audio[-1][0:-4]
        temp_filename_vuv=path_app+'/../tempfiles/tempVUV'+temp_uuid+'.txt'
        temp_filename_f0=path_app+'/../tempfiles/tempF0'+temp_uuid+'.txt'
        praat_functions.praat_vuv(audio_filename, temp_filename_f0, temp_filename_vuv, time_stepF0=step, minf0=minf0, maxf0=maxf0)
        F0,_=praat_functions.decodeF0(temp_filename_f0,len(data_audio)/float(fs),step)
        segmentsFull,segmentsOn,segmentsOff=praat_functions.read_textgrid_trans(temp_filename_vuv,data_audio,fs,sizeframe)
        os.remove(temp_filename_vuv)
        os.remove(temp_filename_f0)
    elif pitch_method == 'rapt':
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
    name_audio=audio_filename.split('/')
    temp_uuid='artic'+name_audio[-1][0:-4]
    temp_filename=path_app+'/../tempfiles/tempFormants'+temp_uuid+'.txt'
    praat_functions.praat_formants(audio_filename, temp_filename,sizeframe,step, path_praat_script=path_app+"/../praat/")
    [F1, F2]=praat_functions.decodeFormants(temp_filename)
    os.remove(temp_filename)

    if len(F0)<len(F1):
        F0=np.hstack((F0, np.zeros(len(F1)-len(F0))))
    else:
        F1=np.hstack((F1, np.zeros(len(F0)-len(F1))))
        F2=np.hstack((F2, np.zeros(len(F0)-len(F2))))


    pos0=np.where(F0==0)[0]
    dpos0=np.hstack(([1],np.diff(pos0)))
    f0u=np.split(pos0, np.where(dpos0>1)[0])

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

    if len(F1nz)==0:
        F1nz=np.zeros((0,1))
    if len(F2nz)==0:
        F2nz=np.zeros((0,1))
    if len(DF1)==0:
        DF1=np.zeros((0,1))
    if len(DDF1)==0:
        DDF1=np.zeros((0,1))
    if len(DF2)==0:
        DF2=np.zeros((0,1))
    if len(DDF2)==0:
        DDF2=np.zeros((0,1))


    return BBEon, MFCCon, DMFCCon, DDMFCCon, BBEoff, MFCCoff, DMFCCoff, DDMFCCoff, F1nz, DF1, DDF1, F2nz, DF2, DDF2


if __name__=="__main__":
    prompt=' <file_or_folder_audio> <file_features.txt> '
    prompt+='[dynamic_or_static (default static)] '
    prompt+='[plots (true or false) (default false)] '
    prompt+='[kaldi output (true or false) (default false)] '
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
            print('python '+sys.argv[0]+prompt)
            sys.exit()
        if sys.argv[5]=="true" or sys.argv[5]=="True":
            flag_kaldi=True
        elif sys.argv[5]=="false" or sys.argv[5]=="False":
            flag_kaldi=False
        else:
            print('python '+sys.argv[0]+prompt)
            sys.exit()
    elif len(sys.argv)==5:
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
            print('python '+sys.argv[0]+prompt)
            sys.exit()
        flag_kaldi=False
    elif len(sys.argv)==4:
        audio=sys.argv[1]
        file_features=sys.argv[2]
        if sys.argv[3]=="static" or sys.argv[3]=="dynamic":
            flag_static=sys.argv[3]
        else:
            print('python '+sys.argv[0]+prompt)
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
        print('python '+sys.argv[0]+prompt)
        sys.exit()


    if audio.find('.wav')!=-1:
        nfiles=1
        hf=['']
    else:
        hf=os.listdir(audio)
        hf.sort()
        nfiles=len(hf)

    if flag_kaldi:
        Features={} # Kaldi Output requires a dictionary
        FeaturesOnset={}
        FeaturesOffset={}
    else:
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


            Feat=[BBEon, MFCCon, DMFCCon, DDMFCCon, BBEoff, MFCCoff, DMFCCoff, DDMFCCoff, F1, DF1, DDF1, F2, DF2, DDF2]

            Nfr=[Feat[n].shape[0] for n in range(len(Feat))]

            avgfeat=[]
            stdfeat=[]
            sk=[]
            ku=[]
            for n in range(len(Feat)):
                Nfr=len(Feat[n].shape)
                if Feat[n].shape[0]>1:
                    avgfeat.append(Feat[n].mean(0))
                    stdfeat.append(Feat[n].std(0))
                    sk.append(st.skew(Feat[n]))
                    ku.append(st.kurtosis(Feat[n], fisher=False))
                elif Feat[n].shape[0]==1:
                    avgfeat.append(Feat[n][0,:])
                    stdfeat.append(np.zeros(Feat[n].shape[1]))
                    sk.append(np.zeros(Feat[n].shape[1]))
                    ku.append(np.zeros(Feat[n].shape[1]))
                else:
                    print(Feat[n].shape)
                    avgfeat.append(np.zeros(Feat[n].shape[1]))
                    stdfeat.append(np.zeros(Feat[n].shape[1]))
                    sk.append(np.zeros(Feat[n].shape[1]))
                    ku.append(np.zeros(Feat[n].shape[1]))
                #print(len(avgfeat[-1]))


            Features_mean=np.hstack(avgfeat)
            Features_std=np.hstack(stdfeat)
            Features_sk=np.hstack(sk)
            Features_ku=np.hstack(ku)
            feat_vec=np.hstack((Features_mean, Features_std, Features_sk, Features_ku))
            if flag_kaldi:
                key=hf[k].replace('.wav', '')
                Features[key]=feat_vec
            else:
                Features.append(feat_vec)


        if flag_static=="dynamic":
            feat_onset=np.hstack((BBEon[2:,:], MFCCon[2:,:], DMFCCon[1:,:], DDMFCCon))
            print(feat_onset.shape)

            if feat_onset.shape[0]==0:
                feat_onset=np.zeros((1,58))

            if flag_kaldi:
                if feat_onset.shape[0] > 0:
                    key=hf[k].replace('.wav', '')
                    FeaturesOnset[key]=feat_onset
            else:
                IDson=np.ones(feat_onset.shape[0])*(k+1)
                FeaturesOnset.append(feat_onset)
                IDon.append(IDson)
            feat_offset=np.hstack((BBEoff[2:,:], MFCCoff[2:,:], DMFCCoff[1:,:], DDMFCCoff))
            if feat_offset.shape[0]==0:
                feat_offset=np.zeros((1,58))

            if flag_kaldi:
                if feat_offset.shape[0] > 0:
                    key=hf[k].replace('.wav', '')
                    FeaturesOffset[key]=feat_offset
            else:
                IDsoff=np.ones(feat_offset.shape[0])*(k+1)
                FeaturesOffset.append(feat_offset)
                IDoff.append(IDsoff)

    if flag_static=="static":
        if flag_kaldi:
            temp_file='temp_static_art'+file_features[:-4]+'.ark'
            with open(temp_file,'wb') as f:
                for key in sorted(Features):
                    write_vec_flt(f, Features[key], key=key)
            ark_file=file_features.replace('.txt','')+'.ark'
            scp_file=file_features.replace('.txt','')+'.scp'
            os.system("copy-vector ark:"+temp_file+" ark,scp:"+ark_file+','+scp_file)
            os.remove(temp_file)
        else:
            Features=np.asarray(Features)
            if file_features.find('.txt')>=0:

                print(file_features, Features.shape)
                np.savetxt(file_features, Features)
            elif file_features.find('.csv')>=0:
                feat_names=["BBEon_"+str(j) for j in range(1,23)]
                feat_names+=["MFCCon_"+str(j) for j in range(1,13)]
                feat_names+=["DMFCCon_"+str(j) for j in range(1,13)]
                feat_names+=["DDMFCCon_"+str(j) for j in range(1,13)]
                feat_names+=["BBEoff_"+str(j) for j in range(1,23)]
                feat_names+=["MFCCoff_"+str(j) for j in range(1,13)]
                feat_names+=["DMFCCoff_"+str(j) for j in range(1,13)]
                feat_names+=["DDMFCCoff_"+str(j) for j in range(1,13)]
                feat_names+=["F1", "DF1", "DDF1", "F2", "DF2", "DDF2"]

                statistics=["avg", "std", "skew", "kurt"]

                feat_names_all=[]
                for k in statistics:
                    for j in feat_names:
                        feat_names_all.append(k+"_"+j)


                df={}
                df["ID"]=hf

                for k in range(Features.shape[1]):
                    df[feat_names_all[k]]=Features[:,k]

                df=pd.DataFrame(df) 
                df.to_csv(file_features)


    if flag_static=="dynamic":
        if flag_kaldi:
            temp_file='temp_dynamic_art'+file_features[:-4]+'.ark'
            with open(temp_file,'wb') as f:
                for key in sorted(FeaturesOnset):
                    write_mat(f, FeaturesOnset[key], key=key)
            ark_file=file_features.replace('.txt','')+'_onset.ark'
            scp_file=file_features.replace('.txt','')+'_onset.scp'
            os.system("copy-matrix ark:"+temp_file+" ark,scp:"+ark_file+','+scp_file)
            # os.remove(temp_file)
            temp_file='temp_dynamic_art2'+file_features[:-4]+'.ark'
            with open(temp_file,'wb') as f:
                for key in sorted(FeaturesOffset):
                    write_mat(f, FeaturesOffset[key], key=key)
            ark_file=file_features.replace('.txt','')+'_offset.ark'
            scp_file=file_features.replace('.txt','')+'_offset.scp'
            os.system("copy-matrix ark:"+temp_file+" ark,scp:"+ark_file+','+scp_file)
            os.remove(temp_file)
        else:
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
