
# -*- coding: utf-8 -*-
"""
Created on Jul 21 2017, Modified Apr 10 2018.

@author: J. C. Vasquez-Correa, T. Arias-Vergara, J. S. Guerrero
"""



import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import pandas as pd
import pysptk
from matplotlib import cm
from scipy.io.wavfile import read
import os
import sys
PATH = os.path.dirname(os.path.realpath(__file__))

sys.path.append(PATH+'/../')
sys.path.append(PATH)

plt.rcParams["font.family"] = "Times New Roman"
from prosody_functions import V_UV, F0feat, energy_cont_segm, polyf0, energy_feat, dur_seg, duration_feat, get_energy_segment

from script_mananger import script_manager
from utils import save_dict_kaldimat, get_dict
import praat.praat_functions as praat_functions
class Prosody:
    """
    Compute prosody features from continuous speech based on duration, fundamental frequency and energy.
    Static or dynamic matrices can be computed:
    Static matrix is formed with 103 features and include

    1-6     F0-contour:                                                       Avg., Std., Max., Min., Skewness, Kurtosis

    7-12    Tilt of a linear estimation of F0 for each voiced segment:        Avg., Std., Max., Min., Skewness, Kurtosis

    13-18   MSE of a linear estimation of F0 for each voiced segment:         Avg., Std., Max., Min., Skewness, Kurtosis

    19-24   F0 on the first voiced segment:                                   Avg., Std., Max., Min., Skewness, Kurtosis

    25-30   F0 on the last voiced segment:                                    Avg., Std., Max., Min., Skewness, Kurtosis

    31-34   energy-contour for voiced segments:                               Avg., Std., Skewness, Kurtosis

    35-38   Tilt of a linear estimation of energy contour for V segments:     Avg., Std., Skewness, Kurtosis

    39-42   MSE of a linear estimation of energy contour for V segment:       Avg., Std., Skewness, Kurtosis

    43-48   energy on the first voiced segment:                               Avg., Std., Max., Min., Skewness, Kurtosis

    49-54   energy on the last voiced segment:                                Avg., Std., Max., Min., Skewness, Kurtosis

    55-58   energy-contour for unvoiced segments:                             Avg., Std., Skewness, Kurtosis

    59-62   Tilt of a linear estimation of energy contour for U segments:     Avg., Std., Skewness, Kurtosis

    63-66   MSE of a linear estimation of energy contour for U segments:      Avg., Std., Skewness, Kurtosis

    67-72   energy on the first unvoiced segment:                             Avg., Std., Max., Min., Skewness, Kurtosis

    73-78   energy on the last unvoiced segment:                              Avg., Std., Max., Min., Skewness, Kurtosis

    79      Voiced rate:                                                      Number of voiced segments per second

    80-85   Duration of Voiced:                                               Avg., Std., Max., Min., Skewness, Kurtosis

    86-91   Duration of Unvoiced:                                             Avg., Std., Max., Min., Skewness, Kurtosis

    92-97   Duration of Pauses:                                               Avg., Std., Max., Min., Skewness, Kurtosis

    98-103  Duration ratios:                                                 Pause/(Voiced+Unvoiced), Pause/Unvoiced, Unvoiced/(Voiced+Unvoiced),Voiced/(Voiced+Unvoiced), Voiced/Puase, Unvoiced/Pause

    Dynamic matrix is formed with 13 features computed for each voiced segment and contains


    1-6. Coefficients of 5-degree Lagrange polynomial to model F0 contour

    7-12. Coefficients of 5-degree Lagrange polynomial to model energy contour

    13. Duration of the voiced segment

    Dynamic prosody features are based on
    Najim Dehak, "Modeling Prosodic Features With Joint Factor Analysis for Speaker Verification", 2007

    Script is called as follows

    >>> python prosody.py <file_or_folder_audio> <file_features> <static (true or false)> <plots (true or false)> <format (csv, txt, npy, kaldi, torch)>

    Examples command line:

    >>> python Prosody.py "../audios/001_ddk1_PCGITA.wav" "prosodyfeaturesAst.txt" "true" "true" "txt"
    >>> python Prosody.py "../audios/001_ddk1_PCGITA.wav" "prosodyfeaturesUst.csv" "true" "true" "csv"
    >>> python prosody.py "../audios/001_ddk1_PCGITA.wav" "prosodyfeaturesUdyn.pt" "false" "true" "torch"

    >>> python Prosody.py "../audios/" "prosodyfeaturesst.txt" "true" "false" "txt"
    >>> python Prosody.py "../audios/" "prosodyfeaturesst.csv" "true" "false" "csv"
    >>> python Prosody.py "../audios/" "prosodyfeaturesdyn.pt" "false" "false" "torch"
    >>> python Prosody.py "../audios/" "prosodyfeaturesdyn.csv" "false" "false" "csv"

    Examples directly in Python

    >>> prosody=Prosody()
    >>> file_audio="../audios/001_ddk1_PCGITA.wav"
    >>> features1=prosody.extract_features_file(file_audio, static=True, plots=True, fmt="npy")
    >>> features2=prosody.extract_features_file(file_audio, static=True, plots=True, fmt="dataframe")
    >>> features3=prosody.extract_features_file(file_audio, static=False, plots=True, fmt="torch")
    >>> prosody.extract_features_file(file_audio, static=False, plots=False, fmt="kaldi", kaldi_file="./test")

    >>> path_audio="../audios/"
    >>> features1=prosody.extract_features_path(path_audio, static=True, plots=False, fmt="npy")
    >>> features2=prosody.extract_features_path(path_audio, static=True, plots=False, fmt="csv")
    >>> features3=prosody.extract_features_path(path_audio, static=False, plots=True, fmt="torch")
    >>> prosody.extract_features_path(path_audio, static=False, plots=False, fmt="kaldi", kaldi_file="./test.ark")

    """

    def __init__(self):
        self.pitch_method = "rapt"
        self.size_frame = 0.02
        self.step = 0.01
        self.thr_len = 0.14
        self.minf0 = 60
        self.maxf0 = 350
        self.voice_bias = -0.2
        self.P = 5
        self.namefeatf0 = ["F0avg", "F0std", "F0max", "F0min",
                           "F0skew", "F0kurt", "F0tiltavg", "F0mseavg",
                           "F0tiltstd", "F0msestd", "F0tiltmax", "F0msemax",
                           "F0tiltmin", "F0msemin", "F0tiltskw", "F0mseskw",
                           "F0tiltku", "F0mseku", "1F0mean", "1F0std",
                           "1F0max", "1F0min", "1F0skw", "1F0ku", "lastF0avg",
                           "lastF0std", "lastF0max", "lastF0min", "lastF0skw", "lastF0ku"]
        self.namefeatEv = ["avgEvoiced", "stdEvoiced", "skwEvoiced", "kurtosisEvoiced",
                           "avgtiltEvoiced", "stdtiltEvoiced", "skwtiltEvoiced", "kurtosistiltEvoiced",
                           "avgmseEvoiced", "stdmseEvoiced", "skwmseEvoiced", "kurtosismseEvoiced",
                           "avg1Evoiced", "std1Evoiced", "max1Evoiced", "min1Evoiced", "skw1Evoiced",
                           "kurtosis1Evoiced", "avglastEvoiced", "stdlastEvoiced", "maxlastEvoiced",
                           "minlastEvoiced", "skwlastEvoiced",  "kurtosislastEvoiced"]
        self.namefeatEu = ["avgEunvoiced", "stdEunvoiced", "skwEunvoiced", "kurtosisEunvoiced",
                           "avgtiltEunvoiced", "stdtiltEunvoiced", "skwtiltEunvoiced", "kurtosistiltEunvoiced",
                           "avgmseEunvoiced", "stdmseEunvoiced", "skwmseEunvoiced", "kurtosismseEunvoiced",
                           "avg1Eunvoiced", "std1Eunvoiced", "max1Eunvoiced", "min1Eunvoiced", "skw1Eunvoiced",
                           "kurtosis1Eunvoiced", "avglastEunvoiced", "stdlastEunvoiced", "maxlastEunvoiced",
                           "minlastEunvoiced", "skwlastEunvoiced",  "kurtosislastEunvoiced"]

        self.namefeatdur = ["Vrate", "avgdurvoiced", "stddurvoiced", "skwdurvoiced", "kurtosisdurvoiced", "maxdurvoiced", "mindurvoiced",
                            "avgdurunvoiced", "stddurunvoiced", "skwdurunvoiced", "kurtosisdurunvoiced", "maxdurunvoiced", "mindurunvoiced",
                            "avgdurpause", "stddurpause", "skwdurpause", "kurtosisdurpause", "maxdurpause", "mindurpause",
                            "PVU", "PU", "UVU", "VVU", "VP", "UP"]
        self.head_st = self.namefeatf0+self.namefeatEv+self.namefeatEu+self.namefeatdur

        self.namef0d = ["f0coef"+str(i) for i in range(6)]
        self.nameEd = ["Ecoef"+str(i) for i in range(6)]
        self.head_dyn = self.namef0d+self.nameEd+["Voiced duration"]

    def plot_pros(self, data_audio, fs, F0, segmentsV, segmentsU, F0_features):
        """Plots of the prosody features

        :param data_audio: speech signal.
        :param fs: sampling frequency
        :param F0: contour of the fundamental frequency
        :param segmentsV: list with the voiced segments
        :param segmentsU: list with the unvoiced segments
        :param F0_features: vector with f0-based features
        :returns: plots of the prosody features.
        """
        plt.figure(figsize=(6, 6))
        plt.subplot(211)
        ax1 = plt.gca()
        t = np.arange(len(data_audio))/float(fs)
        colors = cm.get_cmap('Accent', 5)
        ax1.plot(t, data_audio, 'k', label="speech signal",
                 alpha=0.5, color=colors.colors[4])
        ax1.set_ylabel('Amplitude', fontsize=12)
        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_xlim([0, t[-1]])
        ax2 = ax1.twinx()
        fsp = len(F0)/t[-1]
        t2 = np.arange(len(F0))/fsp
        ax2.plot(
            t2, F0, color=colors.colors[0], linewidth=2, label=r"Real $F_0$", alpha=0.5)
        ax2.set_ylabel(r'$F_0$ (Hz)', color=colors.colors[0], fontsize=12)
        ax2.tick_params('y', colors=colors.colors[0])

        p0 = np.where(F0 != 0)[0]
        f0avg = np.nanmean(np.where(F0 != 0, F0, np.nan))
        f0std = np.std(F0[p0])

        ax2.plot([t2[0], t2[-1]], [f0avg, f0avg],
                 color=colors.colors[2], label=r"Avg. $F_0$")
        ax2.fill_between([t2[0], t2[-1]], y1=[f0avg+f0std, f0avg+f0std], y2=[f0avg-f0std,
                         f0avg-f0std], color=colors.colors[2], alpha=0.2, label=r"Avg. $F_0\pm$ SD.")
        F0rec = polyf0(F0)
        ax2.plot(t2, F0rec, label=r"estimated $F_0$",
                 c=colors.colors[1], linewidth=2.0)
        plt.text(t2[2], np.max(F0)-5, r"$F_0$ SD.=" +
                 str(np.round(f0std, 1))+" Hz")
        plt.text(t2[2], np.max(F0)-20, r"$F_0$ tilt.=" +
                 str(np.round(F0_features[6], 1))+" Hz")

        plt.legend(ncol=2, loc=8)

        plt.subplot(212)
        size_frameS = 0.02*float(fs)
        size_stepS = 0.01*float(fs)

        logE = energy_cont_segm([data_audio], size_frameS, size_stepS)
        Esp = len(logE[0])/t[-1]
        t2 = np.arange(len(logE[0]))/float(Esp)
        plt.plot(t2, logE[0], color='k', linewidth=2.0)
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Energy (dB)', fontsize=12)
        plt.xlim([0, t[-1]])
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6, 3))
        Ev = energy_cont_segm(segmentsV, size_frameS, size_stepS)
        Eu = energy_cont_segm(segmentsU, size_frameS, size_stepS)

        plt.plot([np.mean(Ev[j])
                 for j in range(len(Ev))], label="Voiced energy")
        plt.plot([np.mean(Eu[j])
                 for j in range(len(Eu))], label="Unvoiced energy")

        plt.xlabel("Number of segments")
        plt.ylabel("Energy (dB)")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    def extract_static_features(self, audio, plots, fmt):
        features = self.prosody_static(audio, plots)
        if fmt in ("npy", "txt"):
            return features
        elif fmt in ("dataframe", "csv"):
            df = {}
            for e, k in enumerate(self.head_st):
                df[k] = [features[e]]
            return pd.DataFrame(df)
        elif fmt == "torch":
            feat_t = torch.from_numpy(features)
            return feat_t
        elif fmt == "kaldi":
            raise ValueError("Kaldi is only supported for dynamic features")
        raise ValueError("format" + fmt+" is not supported")

    def extract_dynamic_features(self, audio, fmt, kaldi_file=""):
        features = self.prosody_dynamic(audio)
        if fmt in ("npy", "txt"):
            return features
        if fmt in ("dataframe", "csv"):
            df = {}
            for e, k in enumerate(self.head_dyn):
                df[k] = features[:, e]
            return pd.DataFrame(df)
        if fmt == "torch":
            feat_t = torch.from_numpy(features)
            return feat_t
        if fmt == "kaldi":
            name_all = audio.split('/')
            dictX = {name_all[-1]: features}
            save_dict_kaldimat(dictX, kaldi_file)
        else:
            raise ValueError("format" + fmt+" is not supported")

    def extract_features_file(self, audio, static=True, plots=False, fmt="npy", kaldi_file=""):
        """Extract the prosody features from an audio file

        :param audio: .wav audio file.
        :param static: whether to compute and return statistic functionals over the feature matrix, or return the feature matrix computed over frames
        :param plots: timeshift to extract the features
        :param fmt: format to return the features (npy, dataframe, torch, kaldi)
        :param kaldi_file: file to store kaldi features, only valid when fmt=="kaldi"
        :returns: features computed from the audio file.

        >>> prosody=Prosody()
        >>> file_audio="../audios/001_ddk1_PCGITA.wav"
        >>> features1=prosody.extract_features_file(file_audio, static=True, plots=True, fmt="npy")
        >>> features2=prosody.extract_features_file(file_audio, static=True, plots=True, fmt="dataframe")
        >>> features3=prosody.extract_features_file(file_audio, static=False, plots=True, fmt="torch")
        >>> prosody.extract_features_file(file_audio, static=False, plots=False, fmt="kaldi", kaldi_file="./test")
        """
        if static:
            return self.extract_static_features(audio, plots, fmt)

        else:
            return self.extract_dynamic_features(audio, fmt, kaldi_file)

    def prosody_static(self, audio, plots):
        """Extract the static prosody features from an audio file

        :param audio: .wav audio file.
        :param plots: timeshift to extract the features
        :returns: array with the 103 prosody features

        >>> prosody=Prosody()
        >>> file_audio="../audios/001_ddk1_PCGITA.wav"
        >>> features=prosody.prosody_static(file_audio, plots=True)

        """
        fs, data_audio = read(audio)

        if len(data_audio.shape)>1:
            data_audio = data_audio.mean(1)
        data_audio = data_audio-np.mean(data_audio)
        data_audio = data_audio/float(np.max(np.abs(data_audio)))
        size_frameS = self.size_frame*float(fs)
        size_stepS = self.step*float(fs)
        thr_len_pause = self.thr_len*float(fs)

        if self.pitch_method == 'praat':
            name_audio = audio.split('/')
            temp_uuid = 'prosody'+name_audio[-1][0:-4]
            if not os.path.exists(PATH+'/../tempfiles/'):
                os.makedirs(PATH+'/../tempfiles/')
            temp_filename_f0 = PATH+'/../tempfiles/tempF0'+temp_uuid+'.txt'
            temp_filename_vuv = PATH+'/../tempfiles/tempVUV'+temp_uuid+'.txt'
            praat_functions.praat_vuv(audio, temp_filename_f0, temp_filename_vuv,
                                      time_stepF0=self.step, minf0=self.minf0, maxf0=self.maxf0)

            F0, _ = praat_functions.decodeF0(
                temp_filename_f0, len(data_audio)/float(fs), self.step)
            os.remove(temp_filename_f0)
            os.remove(temp_filename_vuv)
        elif self.pitch_method == 'rapt':
            data_audiof = np.asarray(data_audio*(2**15), dtype=np.float32)
            F0 = pysptk.sptk.rapt(data_audiof, fs, int(
                size_stepS), min=self.minf0, max=self.maxf0, voice_bias=self.voice_bias, otype='f0')

        segmentsV = V_UV(F0, data_audio, type_seg="Voiced",
                         size_stepS=size_stepS)
        segmentsUP = V_UV(F0, data_audio, type_seg="Unvoiced",
                          size_stepS=size_stepS)

        segmentsP = []
        segmentsU = []
        for k in range(len(segmentsUP)):
            if (len(segmentsUP[k]) > thr_len_pause):
                segmentsP.append(segmentsUP[k])
            else:
                segmentsU.append(segmentsUP[k])

        F0_features = F0feat(F0)
        energy_featuresV = energy_feat(segmentsV, fs, size_frameS, size_stepS)
        energy_featuresU = energy_feat(segmentsU, fs, size_frameS, size_stepS)
        duration_features = duration_feat(
            segmentsV, segmentsU, segmentsP, data_audio, fs)

        if plots:
            self.plot_pros(data_audio, fs, F0, segmentsV,
                           segmentsU, F0_features)

        features = np.hstack(
            (F0_features, energy_featuresV, energy_featuresU, duration_features))

        return features

    def prosody_dynamic(self, audio):
        """Extract the dynamic prosody features from an audio file

        :param audio: .wav audio file.
        :returns: array (N,13) with the prosody features extracted from an audio file.  N= number of voiced segments

        >>> prosody=Prosody()
        >>> file_audio="../audios/001_ddk1_PCGITA.wav"
        >>> features=prosody.prosody_dynamic(file_audio)

        """
        fs, data_audio = read(audio)

        if len(data_audio.shape)>1:
            data_audio = data_audio.mean(1)
        data_audio = data_audio-np.mean(data_audio)
        data_audio = data_audio/float(np.max(np.abs(data_audio)))
        size_frameS = self.size_frame*float(fs)
        size_stepS = self.step*float(fs)
        overlap = size_stepS/size_frameS

        if self.pitch_method == 'praat':
            name_audio = audio.split('/')
            temp_uuid = 'prosody'+name_audio[-1][0:-4]
            if not os.path.exists(PATH+'/../tempfiles/'):
                os.makedirs(PATH+'/../tempfiles/')
            temp_filename_f0 = PATH+'/../tempfiles/tempF0'+temp_uuid+'.txt'
            temp_filename_vuv = PATH+'/../tempfiles/tempVUV'+temp_uuid+'.txt'
            praat_functions.praat_vuv(audio, temp_filename_f0, temp_filename_vuv,
                                      time_stepF0=self.step, minf0=self.minf0, maxf0=self.maxf0)

            F0, _ = praat_functions.decodeF0(
                temp_filename_f0, len(data_audio)/float(fs), self.step)
            os.remove(temp_filename_f0)
            os.remove(temp_filename_vuv)
        elif self.pitch_method == 'rapt':
            data_audiof = np.asarray(data_audio*(2**15), dtype=np.float32)
            F0 = pysptk.sptk.rapt(data_audiof, fs, int(
                size_stepS), min=self.minf0, max=self.maxf0, voice_bias=self.voice_bias, otype='f0')

        pitchON = np.where(F0 != 0)[0]
        dchange = np.diff(pitchON)
        change = np.where(dchange > 1)[0]
        iniV = pitchON[0]

        featvec = []
        iniVoiced = (pitchON[0]*size_stepS)+size_stepS
        seg_voiced = []
        f0v = []
        Ev = []
        for indx in change:
            finV = pitchON[indx]+1
            finVoiced = (pitchON[indx]*size_stepS)+size_stepS
            VoicedSeg = data_audio[int(iniVoiced):int(finVoiced)]
            temp = F0[iniV:finV]
            tempvec = []
            if len(VoicedSeg) > int(size_frameS):
                seg_voiced.append(VoicedSeg)
                dur = len(VoicedSeg)/float(fs)
                x = np.arange(0,len(temp))
                z = np.poly1d(np.polyfit(x,temp,self.P))
                f0v.append(temp)
                tempvec.extend(z.coeffs)
                temp=get_energy_segment(size_frameS, size_stepS, VoicedSeg, overlap)
                Ev.append(temp)
                x = np.arange(0, len(temp))
                z = np.poly1d(np.polyfit(x, temp, self.P))
                tempvec.extend(z.coeffs)
                tempvec.append(dur)
                featvec.append(tempvec)
            iniV = pitchON[indx+1]
            iniVoiced = (pitchON[indx+1]*size_stepS)+size_stepS

        # Add the last voiced segment
        finV = (pitchON[len(pitchON)-1])
        finVoiced = (pitchON[len(pitchON)-1]*size_stepS)+size_stepS
        VoicedSeg = data_audio[int(iniVoiced):int(finVoiced)]
        temp = F0[iniV:finV]
        tempvec = []

        if len(VoicedSeg) > int(size_frameS):
            # Compute duration
            dur = len(VoicedSeg)/float(fs)
            
            x = np.arange(0, len(temp))
            z = np.poly1d(np.polyfit(x, temp, self.P))
            tempvec.extend(z.coeffs)
            # Energy coefficients
            temp=get_energy_segment(size_frameS, size_stepS, VoicedSeg, overlap)
            x = np.arange(0, len(temp))
            z = np.poly1d(np.polyfit(x, temp, self.P))
            tempvec.extend(z.coeffs)
            tempvec.append(dur)
            # Compute duration
            featvec.append(tempvec)

        return np.asarray(featvec)

    def extract_features_path(self, path_audio, static=True, plots=False, fmt="npy", kaldi_file=""):
        """Extract the prosody features for audios inside a path

        :param path_audio: directory with (.wav) audio files inside, sampled at 16 kHz
        :param static: whether to compute and return statistic functionals over the feature matrix, or return the feature matrix computed over frames
        :param plots: timeshift to extract the features
        :param fmt: format to return the features (npy, dataframe, torch, kaldi)
        :param kaldi_file: file to store kaldifeatures, only valid when fmt=="kaldi"
        :returns: features computed from the audio file.

        >>> prosody=Prosody()
        >>> path_audio="../audios/"
        >>> features1=prosody.extract_features_path(path_audio, static=True, plots=False, fmt="npy")
        >>> features2=prosody.extract_features_path(path_audio, static=True, plots=False, fmt="csv")
        >>> features3=prosody.extract_features_path(path_audio, static=False, plots=True, fmt="torch")
        >>> prosody.extract_features_path(path_audio, static=False, plots=False, fmt="kaldi", kaldi_file="./test.ark")
        """

        hf = os.listdir(path_audio)
        hf.sort()

        pbar = tqdm(range(len(hf)))
        ids = []

        Features = []
        for j in pbar:
            pbar.set_description("Processing %s" % hf[j])
            audio_file = path_audio+hf[j]
            feat = self.extract_features_file(
                audio_file, static=static, plots=plots, fmt="npy")
            Features.append(feat)
            if static:
                ids.append(hf[j])
            else:
                ids.append(np.repeat(hf[j], feat.shape[0]))

        Features = np.vstack(Features)
        ids = np.hstack(ids)

        return self.save_features(Features, ids, fmt, static, kaldi_file)

    def save_features(self, Features, ids, fmt, static, kaldi_file):

        if fmt in ("npy", "txt"):
            return Features
        elif fmt in ("dataframe", "csv"):
            df = {}
            if static:
                head = self.head_st
            else:
                head = self.head_dyn
            for e, k in enumerate(head):
                df[k] = Features[:, e]

            df["id"] = ids
            return pd.DataFrame(df)
        elif fmt == "torch":
            return torch.from_numpy(Features)
        elif fmt == "kaldi":
            if static:
                raise ValueError(
                    "Kaldi is only supported for dynamic features")
            dictX = get_dict(Features, ids)
            save_dict_kaldimat(dictX, kaldi_file)
        else:
            raise ValueError(fmt+" is not supported")


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("python prosody.py <file_or_folder_audio> <file_features> <static (true, false)> <plots (true,  false)> <format (csv, txt, npy, kaldi, torch)>")
        sys.exit()

    prosody = Prosody()
    script_manager(sys.argv, prosody)
