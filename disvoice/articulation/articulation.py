
# -*- coding: utf-8 -*-
"""
Created on Jul 21 2017

@author: J. C. Vasquez-Correa
"""

from scipy.io.wavfile import read
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import matplotlib.mlab as mlab
import pysptk
import pandas as pd
import torch
from tqdm import tqdm
PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(PATH, '..'))
sys.path.append(PATH)
import disvoice.praat.praat_functions as praat_functions
from disvoice.script_mananger import script_manager
from articulation_functions import extract_transitions, get_transition_segments

from utils import dynamic2statict_artic, save_dict_kaldimat, get_dict, fill_when_empty




class Articulation:
    """

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
    
    Dynamic matrix are formed with the 58 descriptors (22 BBEs, 12 MFCC, 12DMFCC, 12 DDMFCC ) computed for frames of 40 ms with a time-shift of 20 ms in onset transitions.

    The first two frames of each recording are not considered for dynamic analysis to be able to stack the derivatives of MFCCs


    Notes:
    1. The first two frames of each recording are not considered for dynamic analysis to be able to stack the derivatives of MFCCs
    2. The fundamental frequency is computed the PRAAT algorithm. To use the RAPT method,  change the "self.pitch method" variable in the class constructor.

    Script is called as follows

    >>> python articulation.py <file_or_folder_audio> <file_features> <static (true or false)> <plots (true or false)> <format (csv, txt, npy, kaldi, torch)>

    Examples command line:

    >>> python articulation.py "../audios/001_ddk1_PCGITA.wav" "articulation_featuresDDKst.txt" "true" "true" txt
    >>> python articulation.py "../audios/001_ddk1_PCGITA.wav" "articulation_featuresDDKst.csv" "true" "true" csv
    >>> python articulation.py "../audios/001_ddk1_PCGITA.wav" "articulation_featuresDDKst.pt" "true" "true" torch
    >>> python articulation.py "../audios/001_ddk1_PCGITA.wav" "articulation_featuresDDKdyn.txt" "false" "true" txt
    >>> python articulation.py "../audios/001_ddk1_PCGITA.wav" "articulation_featuresDDKdyn.csv" "false" "true" csv
    >>> python articulation.py "../audios/001_ddk1_PCGITA.wav" "articulation_featuresDDKdyn.pt" "false" "true" torch
    
    Examples directly in Python

    >>> articulation=Articulation()
    >>> file_audio="../audios/001_ddk1_PCGITA.wav"
    >>> features1=articulation.extract_features_file(file_audio, static=True, plots=True, fmt="npy")
    >>> features2=articulation.extract_features_file(file_audio, static=True, plots=True, fmt="dataframe")
    >>> features3=articulation.extract_features_file(file_audio, static=False, plots=True, fmt="torch")
    >>> articulation.extract_features_file(file_audio, static=False, plots=False, fmt="kaldi", kaldi_file="./test")

    """

    def __init__(self):
        self.pitch_method="rapt"
        self.sizeframe=0.04
        self.step=0.02
        self.nB=22
        self.nMFCC=12
        self.minf0=60
        self.maxf0=350
        self.voice_bias=-0.2
        self.len_thr_miliseconds=270.0
        self.head=["BBEon_"+str(j) for j in range(1,23)]
        self.head+=["MFCCon_"+str(j) for j in range(1,13)]
        self.head+=["DMFCCon_"+str(j) for j in range(1,13)]
        self.head+=["DDMFCCon_"+str(j) for j in range(1,13)]
        self.head+=["BBEoff_"+str(j) for j in range(1,23)]
        self.head+=["MFCCoff_"+str(j) for j in range(1,13)]
        self.head+=["DMFCCoff_"+str(j) for j in range(1,13)]
        self.head+=["DDMFCCoff_"+str(j) for j in range(1,13)]
        self.head+=["F1", "DF1", "DDF1", "F2", "DF2", "DDF2"]
        self.head_dyn=["BBEon_"+str(j) for j in range(1,23)]
        self.head_dyn+=["MFCCon_"+str(j) for j in range(1,13)]
        self.head_dyn+=["DMFCCon_"+str(j) for j in range(1,13)]
        self.head_dyn+=["DDMFCCon_"+str(j) for j in range(1,13)]
        self.head_st=[]
        for k in ["avg", "std", "skewness", "kurtosis"]:
            for h in self.head:
                self.head_st.append(k+" "+h)
        if not os.path.exists(PATH+'/../../tempfiles/'):
            os.makedirs(PATH+'/../../tempfiles/')

    def plot_art(self, data_audio,fs,F0,F1,F2,segmentsOn,segmentsOff):
        """Plots of the articulation features

        :param data_audio: speech signal.
        :param fs: sampling frequency
        :param F0: contour of the fundamental frequency
        :param F1: contour of the 1st formant
        :param F2: contour of the 2nd formant
        :param segmentsOn: list with the onset segments
        :param segmentsOff: list with the offset segments
        :returns: plots of the articulation features.
        """
        x_axis='Time (s)'
        y_axis='Frequency (Hz)'
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
        plt.xlabel(x_axis, fontsize=14)
        plt.ylabel(y_axis, fontsize=14)
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
        plt.xlabel(x_axis, fontsize=14)
        plt.ylabel(y_axis, fontsize=14)
        plt.ylim([0,np.max(F2)+10])
        plt.xlim([0, t2[-1]])
        plt.grid(True)
        plt.legend()
        plt.tight_layout()    
        plt.show()
        plt.close()
        F0on=[]
        F0off=[]
        for j in range(2, len(F0)):
            if F0[j-1]==0 and F0[j]!=0:
                F0on.append(F0[j-8:j+7])
            elif F0[j-1]!=0 and F0[j]==0:
                F0off.append(F0[j-8:j+7])

        for j in range(1,len(segmentsOn)-1):
            f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1,3]}, sharex=True, figsize=(5,4))
            t=np.arange(len(segmentsOn[j]))/fs
            a0.plot(t[0:640], segmentsOn[j][0:640], color='k', label="unvoiced")
            a0.plot(t[640:], segmentsOn[j][640:], color='k', alpha=0.5, label="voiced")
            a0.grid()
            a0.legend(loc=2, ncol=2 )
            a0.set_xlim([0,0.08])
            spec, freqs, t, im=plt.specgram(segmentsOn[j], NFFT=128, Fs=fs, window=mlab.window_hanning, noverlap=100, detrend=mlab.detrend_mean)
            a1.imshow(np.log10(np.abs(np.flipud(spec))), extent=[0, .08, 1, 4000], aspect='auto', cmap=plt.cm.viridis,
                    vmax=np.log10(np.abs(spec).max()), vmin=np.log10(np.abs(spec).min()), interpolation="bilinear")
            a1.set_ylabel(y_axis, fontsize=12)
            a1.set_xlabel(x_axis, fontsize=12)
            a1.set_xlim([0,0.08])
            a02 = a1.twinx()
            fsp=len(F0on[j])/0.086
            t2=np.arange(len(F0on[j]))/fsp
            a02.plot(t2, F0on[j], 'r', linewidth=2,label="F0")
            a02.set_ylabel(r'$F_0$ (Hz)', color='r', fontsize=12)
            a02.tick_params('y', colors='r')
            a02.set_xlim([0,0.08])
            plt.tight_layout()
            plt.show()

        for j in range(1,len(segmentsOff)-1):
            f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1,3]}, sharex=True, figsize=(5,4))
            t=np.arange(len(segmentsOff[j]))/fs
            a0.plot(t[0:640], segmentsOff[j][0:640], color='k', label="voiced")
            a0.plot(t[640:], segmentsOff[j][640:], color='k', alpha=0.5, label="unvoiced")
            a0.grid()
            a0.legend(loc=1, ncol=2 )
            
            spec, freqs, t, im=plt.specgram(segmentsOff[j], NFFT=128, Fs=fs, window=mlab.window_hanning, noverlap=100, detrend=mlab.detrend_mean)
            a1.imshow(np.log10(np.abs(np.flipud(spec))), extent=[0, .08, 1, 4000], aspect='auto', cmap=plt.cm.viridis,
                    vmax=np.log10(np.abs(spec).max()), vmin=np.log10(np.abs(spec).min()), interpolation="bilinear")
            a1.set_ylabel(y_axis, fontsize=12)
            a1.set_xlabel(x_axis, fontsize=12)
            a02 = a1.twinx()
            fsp=len(F0off[j])/0.086
            t2=np.arange(len(F0off[j]))/fsp
            a02.plot(t2, F0off[j], 'r', linewidth=2,label="F0")
            a02.set_ylabel(r'$F_0$ (Hz)', color='r', fontsize=12)
            a02.tick_params('y', colors='r')
            plt.tight_layout()
            plt.show()


    def extract_features_file(self, audio, static=True, plots=False, fmt="npy", kaldi_file=""):
        """Extract the articulation features from an audio file

        :param audio: .wav audio file.
        :param static: whether to compute and return statistic functionals over the feature matrix, or return the feature matrix computed over frames
        :param plots: timeshift to extract the features
        :param fmt: format to return the features (npy, dataframe, torch, kaldi)
        :param kaldi_file: file to store kaldi features, only valid when fmt=="kaldi"
        :returns: features computed from the audio file.

        >>> articulation=Articulation()
        >>> file_audio="../audios/001_ddk1_PCGITA.wav"
        >>> features1=articulation.extract_features_file(file_audio, static=True, plots=True, fmt="npy")
        >>> features2=articulation.extract_features_file(file_audio, static=True, plots=True, fmt="dataframe")
        >>> features3=articulation.extract_features_file(file_audio, static=False, plots=True, fmt="torch")
        >>> articulation.extract_features_file(file_audio, static=False, plots=False, fmt="kaldi", kaldi_file="./test")
        
        >>> path_audio="../audios/"
        >>> features1=articulation.extract_features_path(path_audio, static=True, plots=False, fmt="npy")
        >>> features2=articulation.extract_features_path(path_audio, static=True, plots=False, fmt="csv")
        >>> features3=articulation.extract_features_path(path_audio, static=False, plots=True, fmt="torch")
        >>> articulation.extract_features_path(path_audio, static=False, plots=False, fmt="kaldi", kaldi_file="./test.ark")

        """
        fs, data_audio=read(audio)

        if len(data_audio.shape)>1:
            data_audio = data_audio.mean(1)
        data_audio=data_audio-np.mean(data_audio)
        data_audio=data_audio/float(np.max(np.abs(data_audio)))
        size_frameS=self.sizeframe*float(fs)
        size_stepS=self.step*float(fs)

        if static and fmt=="kaldi":
            raise ValueError("Kaldi is only supported for dynamic features")

        F0, segmentsOn, segmentsOff = self.extract_transition_segments(audio, fs, data_audio, size_stepS)

        BBEon, MFCCon=extract_transitions(segmentsOn, fs, size_frameS, size_stepS, self.nB, self.nMFCC)
        BBEoff, MFCCoff=extract_transitions(segmentsOff, fs, size_frameS, size_stepS, self.nB, self.nMFCC)

        DMFCCon=np.asarray([np.diff(MFCCon[:,nf], n=1) for nf in range(MFCCon.shape[1])]).T
        DDMFCCon=np.asarray([np.diff(MFCCon[:,nf], n=2) for nf in range(MFCCon.shape[1])]).T

        DMFCCoff=np.asarray([np.diff(MFCCoff[:,nf], n=1) for nf in range(MFCCoff.shape[1])]).T
        DDMFCCoff=np.asarray([np.diff(MFCCoff[:,nf], n=2) for nf in range(MFCCoff.shape[1])]).T


        name_audio=audio.split('/')
        temp_uuid='artic'+name_audio[-1][0:-4]

        temp_filename=PATH+'/../../tempfiles/tempFormants'+temp_uuid+'.txt'
        praat_functions.praat_formants(audio, temp_filename,self.sizeframe,self.step)
        [F1, F2]=praat_functions.decodeFormants(temp_filename)
        os.remove(temp_filename)

        if len(F0)<len(F1):
            F0=np.hstack((F0, np.zeros(len(F1)-len(F0))))
            F1nz=np.zeros((0,1))
            F2nz=np.zeros((0,1))
            DF1=np.zeros((0,1))
            DDF1=np.zeros((0,1))
            DF2=np.zeros((0,1))
            DDF2=np.zeros((0,1))

        else:
            F1=np.hstack((F1, np.zeros(len(F0)-len(F1))))
            F2=np.hstack((F2, np.zeros(len(F0)-len(F2))))

            pos0=np.where(F0==0)[0]
            dpos0=np.hstack(([1],np.diff(pos0)))
            f0u=np.split(pos0, np.where(dpos0>1)[0])

            thr_sil=int(self.len_thr_miliseconds/self.step)

            len_segments=np.array([len(segment) for segment in f0u])
            index_silence=np.where(len_segments>=thr_sil)[0]
            F1[index_silence]=0
            F2[index_silence]=0

            F1nz=F1[F1!=0]
            F2nz=F2[F2!=0]
            DF1=np.diff(F1, n=1)
            DF2=np.diff(F2, n=1)
            DDF1=np.diff(F1, n=2)
            DDF2=np.diff(F2, n=2)


            if plots:
                self.plot_art(data_audio,fs,F0,F1,F2,segmentsOn,segmentsOff)

            F1nz=fill_when_empty(F1nz)
            F2nz=fill_when_empty(F2nz)
            DF1=fill_when_empty(DF1)
            DDF1=fill_when_empty(DDF1)
            DF2=fill_when_empty(DF2)
            DDF2=fill_when_empty(DDF2)

        if static:
            feat_mat=dynamic2statict_artic([BBEon, MFCCon, DMFCCon, DDMFCCon, BBEoff, MFCCoff, DMFCCoff, DDMFCCoff, F1nz, DF1, DDF1, F2nz, DF2, DDF2])
            feat_mat=np.expand_dims(feat_mat,0)
            head=self.head_st
        else:
            feat_mat=np.hstack((BBEon[2:,:], MFCCon[2:,:], DMFCCon[1:,:], DDMFCCon))
            head=self.head_dyn

        if fmt in("npy","txt"):
            return feat_mat
        elif fmt in("dataframe","csv"):
            df={}
            for e, k in enumerate(head):
                df[k]=feat_mat[:,e]
            return pd.DataFrame(df)
        elif fmt=="torch":
            feat_t=torch.from_numpy(feat_mat)
            return feat_t
        elif fmt=="kaldi":
            name_all=audio.split('/')
            dictX={name_all[-1]:feat_mat}
            save_dict_kaldimat(dictX, kaldi_file)

    def extract_transition_segments(self, audio, fs, data_audio, size_stepS):
        if self.pitch_method == 'praat':
            name_audio=audio.split('/')
            temp_uuid='articulation'+name_audio[-1][0:-4]
            temp_filename_vuv=PATH+'/../../tempfiles/tempVUV'+temp_uuid+'.txt'
            temp_filename_f0=PATH+'/../../tempfiles/tempF0'+temp_uuid+'.txt'
            praat_functions.praat_vuv(audio, temp_filename_f0, temp_filename_vuv, time_stepF0=self.step, minf0=self.minf0, maxf0=self.maxf0)
            F0,_=praat_functions.decodeF0(temp_filename_f0,len(data_audio)/float(fs),self.step)
            segmentsFull,segmentsOn,segmentsOff=praat_functions.read_textgrid_trans(temp_filename_vuv,data_audio,fs,self.sizeframe)
            os.remove(temp_filename_vuv)
            os.remove(temp_filename_f0)
        elif self.pitch_method == 'rapt':
            data_audiof=np.asarray(data_audio*(2**15), dtype=np.float32)
            F0=pysptk.sptk.rapt(data_audiof, fs, int(size_stepS), min=self.minf0, max=self.maxf0, voice_bias=self.voice_bias, otype='f0')

            segmentsOn=get_transition_segments(F0, data_audio, fs, 'onset')
            segmentsOff=get_transition_segments(F0, data_audio, fs, 'offset')
        return F0,segmentsOn,segmentsOff


    def extract_features_path(self, path_audio, static=True, plots=False, fmt="npy", kaldi_file=""):
        """Extract the articulation features for audios inside a path
        
        :param path_audio: directory with (.wav) audio files inside, sampled at 16 kHz
        :param static: whether to compute and return statistic functionals over the feature matrix, or return the feature matrix computed over frames
        :param plots: timeshift to extract the features
        :param fmt: format to return the features (npy, dataframe, torch, kaldi)
        :param kaldi_file: file to store kaldifeatures, only valid when fmt=="kaldi"
        :returns: features computed from the audio file.

        >>> articulation=Articulation()
        >>> path_audio="../audios/"
        >>> features1=articulation.extract_features_path(path_audio, static=True, plots=False, fmt="npy")
        >>> features2=articulation.extract_features_path(path_audio, static=True, plots=False, fmt="csv")
        >>> features3=articulation.extract_features_path(path_audio, static=False, plots=True, fmt="torch")
        >>> articulation.extract_features_path(path_audio, static=False, plots=False, fmt="kaldi", kaldi_file="./test.ark")
        """

        hf=os.listdir(path_audio)
        hf.sort()

        pbar=tqdm(range(len(hf)))
        ids=[]

        if static and fmt=="kaldi":
            raise ValueError("Kaldi is only supported for dynamic features")        

        Features=[]
        for j in pbar:
            pbar.set_description("Processing %s" % hf[j])
            audio_file=path_audio+hf[j]
            feat=self.extract_features_file(audio_file, static=static, plots=plots, fmt="npy")
            Features.append(feat)
            if static:
                ids.append(hf[j])
            else:
                ids.append(np.repeat(hf[j], feat.shape[0]))
        
        Features=np.vstack(Features)
        ids=np.hstack(ids)
        if fmt in("npy","txt"):
            return Features
        if fmt in("dataframe","csv"):
            df={}
            if static:
                head=self.head_st
            else:
                head=self.head_dyn
            for e, k in enumerate(head):
                df[k]=Features[:,e]

            df["id"]=ids
            return pd.DataFrame(df)
        if fmt=="torch":
            return torch.from_numpy(Features)
        if fmt=="kaldi":
            dictX=get_dict(Features, ids)
            save_dict_kaldimat(dictX, kaldi_file)


if __name__=="__main__":

    if len(sys.argv)!=6:
        print("python articulation.py <file_or_folder_audio> <file_features> <static (true, false)> <plots (true,  false)> <format (csv, txt, npy, kaldi, torch)>")
        sys.exit()

    articulation=Articulation()
    script_manager(sys.argv, articulation)