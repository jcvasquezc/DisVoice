
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

import pysptk

import pandas as pd
PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(PATH, '..'))
sys.path.append(PATH)
from phonation_functions import jitter_env, get_log_energy, shimmer_env, APQ, PPQ

from utils import dynamic2statict, save_dict_kaldimat,get_dict
import praat.praat_functions as praat_functions
from script_mananger import script_manager
import torch
from tqdm import tqdm


class Phonation:
    """
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
    2. The fundamental frequency is computed the RAPT algorithm. To use the PRAAT method,  change the "self.pitch method" variable in the class constructor.

    Script is called as follows

    >>> python phonation.py <file_or_folder_audio> <file_features> <static (true or false)> <plots (true or false)> <format (csv, txt, npy, kaldi, torch)>

    Examples command line:

    >>> python phonation.py "../audios/001_a1_PCGITA.wav" "phonationfeaturesAst.txt" "true" "true" "txt"
    >>> python phonation.py "../audios/098_u1_PCGITA.wav" "phonationfeaturesUst.csv" "true" "true" "csv"
    >>> python phonation.py "../audios/098_u1_PCGITA.wav" "phonationfeaturesUdyn.pt" "false" "true" "torch"

    >>> python phonation.py "../audios/" "phonationfeaturesst.txt" "true" "false" "txt"
    >>> python phonation.py "../audios/" "phonationfeaturesst.csv" "true" "false" "csv"
    >>> python phonation.py "../audios/" "phonationfeaturesdyn.pt" "false" "false" "torch"

    Examples directly in Python

    >>> from disvoice.phonation import Phonation
    >>> phonation=Phonation()
    >>> file_audio="../audios/001_a1_PCGITA.wav"
    >>> features=phonation.extract_features_file(file_audio, static, plots=True, fmt="numpy")
    >>> features2=phonation.extract_features_file(file_audio, static, plots=True, fmt="dataframe")
    >>> features3=phonation.extract_features_file(file_audio, dynamic, plots=True, fmt="torch")
    
    >>> path_audios="../audios/"
    >>> features1=phonation.extract_features_path(path_audios, static, plots=False, fmt="numpy")
    >>> features2=phonation.extract_features_path(path_audios, static, plots=False, fmt="torch")
    >>> features3=phonation.extract_features_path(path_audios, static, plots=False, fmt="dataframe")

    """
    def __init__(self):
        self.pitch_method="rapt"
        self.size_frame=0.04
        self.size_step=0.02
        self.minf0=60
        self.maxf0=350
        self.voice_bias=-0.2
        self.energy_thr_percent=0.025
        self.head_dyn=["DF0", "DDF0", "Jitter", "Shimmer", "apq", "ppq", "logE"]
        
        if not os.path.exists(PATH+'/../tempfiles/'):
            os.makedirs(PATH+'/../tempfiles/')
        self.head_st=[]
        for k in ["avg", "std", "skewness", "kurtosis"]:
            for h in self.head_dyn:
                self.head_st.append(k+" "+h)


    def plot_phon(self, data_audio,fs,F0,logE):
        """Plots of the phonation features

        :param data_audio: speech signal.
        :param fs: sampling frequency
        :param F0: contour of the fundamental frequency
        :param logE: contour of the log-energy
        :returns: plots of the phonation features.
        """
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
        ax2.plot(t2, F0, 'r', linewidth=2,label=r"F_0")
        ax2.set_ylabel(r'$F_0$ (Hz)', color='r', fontsize=12)
        ax2.tick_params('y', colors='r')

        plt.grid(True)

        plt.subplot(212)
        Esp=len(logE)/t[-1]
        t2=np.arange(len(logE))/float(Esp)
        plt.plot(t2, logE, color='k', linewidth=2.0)
        plt.xlabel('Time (s)', fontsize=14)
        plt.ylabel('Energy (dB)', fontsize=14)
        plt.xlim([0, t[-1]])
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def extract_features_file(self, audio, static=True, plots=False, fmt="npy", kaldi_file=""):
        """Extract the phonation features from an audio file
        
        :param audio: .wav audio file.
        :param static: whether to compute and return statistic functionals over the feature matrix, or return the feature matrix computed over frames
        :param plots: timeshift to extract the features
        :param fmt: format to return the features (npy, dataframe, torch, kaldi)
        :param kaldi_file: file to store kaldi features, only valid when fmt=="kaldi"
        :returns: features computed from the audio file.

        >>> phonation=Phonation()
        >>> file_audio="../audios/001_a1_PCGITA.wav"
        >>> features1=phonation.extract_features_file(file_audio, static=True, plots=True, fmt="npy")
        >>> features2=phonation.extract_features_file(file_audio, static=True, plots=True, fmt="dataframe")
        >>> features3=phonation.extract_features_file(file_audio, static=False, plots=True, fmt="torch")
        >>> phonation.extract_features_file(file_audio, static=False, plots=False, fmt="kaldi", kaldi_file="./test")
        """
        if static and fmt=="kaldi":
            raise ValueError("Kaldi is only supported for dynamic features")

        if audio.find('.wav') == -1 and audio.find('.WAV') == -1:
            raise ValueError(audio+" is not a valid wav file")

        fs, data_audio=read(audio)

        if len(data_audio.shape)>1:
            data_audio = data_audio.mean(1)
        data_audio=data_audio-np.mean(data_audio)
        data_audio=data_audio/float(np.max(np.abs(data_audio)))
        size_frameS=self.size_frame*float(fs)
        size_stepS=self.size_step*float(fs)
        overlap=size_stepS/size_frameS
        F0 = self.get_F0(audio, fs, data_audio, size_stepS)
        F0nz=F0[F0!=0]
        Jitter=jitter_env(F0nz, len(F0nz))
        nF=int((len(data_audio)/size_frameS/overlap))-1
        Amp=[]
        logE=[]
        apq=[]
        ppq=[]
        DF0=np.diff(F0nz, 1)
        DDF0=np.diff(DF0,1)
        lnz=0
        for l in range(nF):
            if F0[l]==0:
                continue
            data_frame=data_audio[int(l*size_stepS):int(l*size_stepS+size_frameS)]
            energy=10*get_log_energy(data_frame)
            Amp.append(np.max(np.abs(data_frame)))
            logE.append(energy)
            if lnz>=12:
                amp_arr=np.asarray([Amp[j] for j in range(lnz-12, lnz)])
                apq.append(APQ(amp_arr))
            if lnz>=6:
                f0arr=np.asarray([F0nz[j] for j in range(lnz-6, lnz)])
                ppq.append(PPQ(1./f0arr))
            lnz=lnz+1
        
        Shimmer=shimmer_env(Amp, len(Amp))
        apq=np.asarray(apq)
        ppq=np.asarray(ppq)
        logE=np.asarray(logE)

        if len(apq)==0:
            apq=Shimmer

        if plots:
            self.plot_phon(data_audio,fs,F0,logE)

        if static:
            feat=dynamic2statict([DF0, DDF0, Jitter, Shimmer, apq, ppq, logE])
            feat=np.expand_dims(feat, axis=0)
            head = self.head_st
        else:
            if len(Shimmer)==len(apq):
                feat=np.vstack((DF0[5:], DDF0[4:], Jitter[6:], Shimmer[6:], apq[6:], ppq, logE[6:])).T
            else:
                feat=np.vstack((DF0[11:], DDF0[10:], Jitter[12:], Shimmer[12:], apq, ppq[6:], logE[12:])).T
            head=self.head_dyn

        if fmt in("npy","txt"):
            return feat

        elif fmt in("dataframe","csv"):
            df = {}
            for e, k in enumerate(head):
                df[k] = feat[:, e]
            return pd.DataFrame(df)
        elif fmt=="torch":
            return torch.from_numpy(feat)
        elif fmt=="kaldi":
            name_all=audio.split('/')
            dictX={name_all[-1]:feat}
            save_dict_kaldimat(dictX, kaldi_file)
        else:
            raise ValueError(fmt+" is not supported")

    def get_F0(self, audio, fs, data_audio, size_stepS):
        if self.pitch_method == 'praat':
            name_audio=audio.split('/')
            temp_uuid='phon'+name_audio[-1][0:-4]
            temp_filename_vuv=PATH+'/../tempfiles/tempVUV'+temp_uuid+'.txt'
            temp_filename_f0=PATH+'/../tempfiles/tempF0'+temp_uuid+'.txt'
            praat_functions.praat_vuv(audio, temp_filename_f0, temp_filename_vuv, time_stepF0=self.size_step, minf0=self.minf0, maxf0=self.maxf0)
            F0,_=praat_functions.decodeF0(temp_filename_f0,len(data_audio)/float(fs),self.size_step)
            os.remove(temp_filename_vuv)
            os.remove(temp_filename_f0)
        elif self.pitch_method == 'rapt':
            data_audiof=np.asarray(data_audio*(2**15), dtype=np.float32)
            F0=pysptk.sptk.rapt(data_audiof, fs, int(size_stepS), min=self.minf0, max=self.maxf0, voice_bias=self.voice_bias, otype='f0')
        return F0

    def extract_features_path(self, path_audio, static=True, plots=False, fmt="npy", kaldi_file=""):
        """Extract the phonation features for audios inside a path
        
        :param path_audio: directory with (.wav) audio files inside, sampled at 16 kHz
        :param static: whether to compute and return statistic functionals over the feature matrix, or return the feature matrix computed over frames
        :param plots: timeshift to extract the features
        :param fmt: format to return the features (npy, dataframe, torch, kaldi)
        :param kaldi_file: file to store kaldifeatures, only valid when fmt=="kaldi"
        :returns: features computed from the audio file.

        >>> phonation=Phonation()
        >>> path_audio="../audios/"
        >>> features1=phonation.extract_features_path(path_audio, static=True, plots=False, fmt="npy")
        >>> features2=phonation.extract_features_path(path_audio, static=True, plots=False, fmt="csv")
        >>> features3=phonation.extract_features_path(path_audio, static=False, plots=True, fmt="torch")
        >>> phonation.extract_features_path(path_audio, static=False, plots=False, fmt="kaldi", kaldi_file="./test.ark")
        """
        hf=os.listdir(path_audio)
        hf.sort()

        pbar=tqdm(range(len(hf)))
        ids=[]

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
        return self.save_features(Features, ids, fmt, static, kaldi_file)



    def save_features(self, Features, ids, fmt, static, kaldi_file):
        if static:
            head = self.head_st
        else:
            head = self.head_dyn
        
        if fmt in ("npy", "txt"):
            return Features
        if fmt in ("dataframe", "csv"):
            df = {}
            for e, k in enumerate(head):
                df[k] = Features[:, e]
            df["id"] = ids
            return pd.DataFrame(df)
        if fmt == "torch":
            return torch.from_numpy(Features)
        if fmt == "kaldi":
            dictX = get_dict(Features, ids)
            save_dict_kaldimat(dictX, kaldi_file)

if __name__=="__main__":

    if len(sys.argv)!=6:
        print("python phonation.py <file_or_folder_audio> <file_features> <static (true, false)> <plots (true,  false)> <format (csv, txt, npy, kaldi, torch)>")
        sys.exit()

    phonation=Phonation()
    script_manager(sys.argv, phonation)