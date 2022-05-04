import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysptk
import torch
from scipy.integrate import cumtrapz
from scipy.io.wavfile import read
from tqdm import tqdm

PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(PATH, '..'))
sys.path.append(PATH)
from script_mananger import script_manager
from utils import dynamic2static, get_dict, save_dict_kaldimat
plt.rcParams["font.family"] = "Times New Roman"
from GCI import iaif, se_vq_varf0, get_vq_params


class Glottal:
    """
    Compute features based on the glottal source reconstruction from sustained vowels and continuous speech.

    For continuous speech, the features are computed over voiced segments

    Nine descriptors are computed:

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

    Dynamic matrix is formed with the 9 descriptors computed for frames of 200 ms length with a time-shift of 50 ms.

    Notes:

    1. The fundamental frequency is computed using the RAPT algorithm.

    >>> python glottal.py <file_or_folder_audio> <file_features> <dynamic_or_static> <plots (true,  false)> <format (csv, txt, npy, kaldi, torch)>

    Examples command line:

    >>> python glottal.py "../audios/001_a1_PCGITA.wav" "glottalfeaturesAst.txt" "static" "true" "txt"
    >>> python glottal.py "../audios/098_u1_PCGITA.wav" "glottalfeaturesUst.csv" "static" "true" "csv"
    >>> python glottal.py "../audios/098_u1_PCGITA.wav" "glottalfeaturesUst.ark" "dynamic" "true" "kaldi"
    >>> python glottal.py "../audios/098_u1_PCGITA.wav" "glottalfeaturesUst.pt" "dynamic" "true" "torch"


    Examples directly in Python

    >>> from disvoice.glottal import Glottal
    >>> glottal=Glottal()
    >>> file_audio="../audios/001_a1_PCGITA.wav"
    >>> features=glottal.extract_features_file(file_audio, static, plots=True, fmt="numpy")
    >>> features2=glottal.extract_features_file(file_audio, static, plots=True, fmt="dataframe")
    >>> features3=glottal.extract_features_file(file_audio, dynamic, plots=True, fmt="torch")

    >>> path_audios="../audios/"
    >>> features1=glottal.extract_features_path(path_audios, static, plots=False, fmt="numpy")
    >>> features2=glottal.extract_features_path(path_audios, static, plots=False, fmt="torch")
    >>> features3=glottal.extract_features_path(path_audios, static, plots=False, fmt="dataframe")
    """

    def __init__(self):
        self.size_frame = 0.2
        self.size_step = 0.05
        self.head_dyn = ["var GCI", "avg NAQ", "std NAQ", "avg QOQ",
                         "std QOQ", "avg H1H2", "std H1H2", "avg HRF", "std HRF"]
        self.head_st = []
        for k in ["global avg", "global std", "global skewness", "global kurtosis"]:
            for h in self.head_dyn:
                self.head_st.append(k+" "+h)

    def plot_glottal(self, data_audio, fs, GCI, glottal_flow, glottal_sig):
        """Plots of the glottal features

        :param data_audio: speech signal.
        :param fs: sampling frequency
        :param GCI: glottal closure instants
        :param glottal_flow: glottal flow
        :param glottal_sig: reconstructed glottal signal
        :returns: plots of the glottal features.
        """

        fig, ax = plt.subplots(3, sharex=True)
        t = np.arange(0, float(len(data_audio))/fs, 1.0/fs)
        if len(t) > len(data_audio):
            t = t[:len(data_audio)]
        elif len(t) < len(data_audio):
            data_audio = data_audio[:len(t)]
        ax[0].plot(t, data_audio, 'k')
        ax[0].set_ylabel('Amplitude', fontsize=12)
        ax[0].set_xlim([0, t[-1]])
        ax[0].grid(True)

        ax[1].plot(t, glottal_sig, color='k', linewidth=2.0,
                   label="Glottal flow signal")
        amGCI = [glottal_sig[int(k-2)] for k in GCI]

        GCI = GCI/fs
        ax[1].plot(GCI, amGCI, 'bo', alpha=0.5, markersize=8, label="GCI")
        GCId = np.diff(GCI)
        ax[1].set_ylabel("Glottal flow", fontsize=12)
        ax[1].text(t[2], -0.8, "Avg. time consecutive GCI:" +
                   str(np.round(np.mean(GCId)*1000, 2))+" ms")
        ax[1].text(t[2], -1.05, "Std. time consecutive GCI:" +
                   str(np.round(np.std(GCId)*1000, 2))+" ms")
        ax[1].set_xlabel('Time (s)', fontsize=12)
        ax[1].set_xlim([0, t[-1]])
        ax[1].set_ylim([-1.1, 1.1])
        ax[1].grid(True)

        ax[1].legend(ncol=2, loc=2)

        ax[2].plot(t, glottal_flow, color='k', linewidth=2.0)
        ax[2].set_ylabel("Glotal flow derivative", fontsize=12)
        ax[2].set_xlabel('Time (s)', fontsize=12)
        ax[2].set_xlim([0, t[-1]])
        ax[2].grid(True)

        plt.show()

    def extract_glottal_signal(self, x, fs):
        """Extract the glottal flow and the glottal flow derivative signals

        :param x: data from the speech signal.
        :param fs: sampling frequency
        :returns: glottal signal
        :returns: derivative  of the glottal signal
        :returns: glottal closure instants

        >>> from scipy.io.wavfile import read
        >>> glottal=Glottal()
        >>> file_audio="../audios/001_a1_PCGITA.wav"
        >>> fs, data_audio=read(audio)
        >>> glottal, g_iaif, GCIs=glottal.extract_glottal_signal(data_audio, fs)

        """
        winlen = int(0.025*fs)
        winshift = int(0.005*fs)
        x = x-np.mean(x)
        x = x/float(np.max(np.abs(x)))
        GCIs = se_vq_varf0(x, fs)
        g_iaif = np.zeros(len(x))
        glottal = np.zeros(len(x))

        if GCIs is None:
            sys.warn("not enought voiced segments were found to compute GCI")
            return glottal, g_iaif, GCIs

        start = 0
        stop = int(start+winlen)
        win = np.hanning(winlen)

        while stop <= len(x):

            x_frame = x[start:stop]
            pGCIt = np.where((GCIs > start) & (GCIs < stop))[0]
            GCIt = GCIs[pGCIt]-start

            g_iaif_f = iaif(x_frame, fs, GCIt)
            glottal_f = cumtrapz(g_iaif_f, dx=1/fs)
            glottal_f = np.hstack((glottal[start], glottal_f))
            g_iaif[start:stop] = g_iaif[start:stop]+g_iaif_f*win
            glottal[start:stop] = glottal[start:stop]+glottal_f*win
            start = start+winshift
            stop = start+winlen
        g_iaif = g_iaif-np.mean(g_iaif)
        g_iaif = g_iaif/max(abs(g_iaif))

        glottal = glottal-np.mean(glottal)
        glottal = glottal/max(abs(glottal))
        glottal = glottal-np.mean(glottal)
        glottal = glottal/max(abs(glottal))

        return glottal, g_iaif, GCIs

    def extract_features_file(self, audio, static=True, plots=False, fmt="npy", kaldi_file=""):
        """Extract the glottal features from an audio file

        :param audio: .wav audio file.
        :param static: whether to compute and return statistic functionals over the feature matrix, or return the feature matrix computed over frames
        :param plots: timeshift to extract the features
        :param fmt: format to return the features (npy, dataframe, torch, kaldi)
        :param kaldi_file: file to store kaldi features, only valid when fmt=="kaldi"
        :returns: features computed from the audio file.

        >>> glottal=Glottal()
        >>> file_audio="../audios/001_a1_PCGITA.wav"
        >>> features1=glottal.extract_features_file(file_audio, static=True, plots=True, fmt="npy")
        >>> features2=glottal.extract_features_file(file_audio, static=True, plots=True, fmt="dataframe")
        >>> features3=glottal.extract_features_file(file_audio, static=False, plots=True, fmt="torch")
        >>> glottal.extract_features_file(file_audio, static=False, plots=False, fmt="kaldi", kaldi_file="./test.ark")
        """

        if static and fmt=="kaldi":
            raise ValueError("Kaldi is only supported for dynamic features")

        if audio.find('.wav') == -1 and audio.find('.WAV') == -1:
            raise ValueError(audio+" is not a valid wav file")
        fs, data_audio = read(audio)

        if len(data_audio.shape)>1:
            data_audio = data_audio.mean(1)

        data_audio = data_audio-np.mean(data_audio)
        data_audio = data_audio/float(np.max(np.abs(data_audio)))
        size_frameS = self.size_frame*float(fs)
        size_stepS = self.size_step*float(fs)
        overlap = size_stepS/size_frameS
        nF = int((len(data_audio)/size_frameS/overlap))-1
        data_audiof = np.asarray(data_audio*(2**15), dtype=np.float32)
        f0 = pysptk.sptk.rapt(data_audiof, fs, int(
            0.01*fs), min=20, max=500, voice_bias=-0.2, otype='f0')
        sizef0 = int(self.size_frame/0.01)
        stepf0 = int(self.size_step/0.01)
        startf0 = 0
        stopf0 = sizef0

        glottal, g_iaif, GCI = self.extract_glottal_signal(data_audio, fs)

        if plots:
            self.plot_glottal(data_audio, fs, GCI, g_iaif, glottal)

        avgGCIt = np.zeros(nF)
        varGCIt = np.zeros(nF)
        avgNAQt = np.zeros(nF)
        varNAQt = np.zeros(nF)
        avgQOQt = np.zeros(nF)
        varQOQt = np.zeros(nF)
        avgH1H2t = np.zeros(nF)
        varH1H2t = np.zeros(nF)
        avgHRFt = np.zeros(nF)
        varHRFt = np.zeros(nF)
        rmwin = []
        for l in range(nF):
            init = int(l*size_stepS)
            endi = int(l*size_stepS+size_frameS)
            gframe = glottal[init:endi]
            dgframe = glottal[init:endi]
            pGCIt = np.where((GCI > init) & (GCI < endi))[0]
            gci_s = GCI[pGCIt]-init
            f0_frame = f0[startf0:stopf0]
            pf0framez = np.where(f0_frame != 0)[0]
            f0nzframe = f0_frame[pf0framez]
            if len(f0nzframe) < 5:
                startf0 = startf0+stepf0
                stopf0 = stopf0+stepf0
                rmwin.append(l)
                continue

            startf0 = startf0+stepf0
            stopf0 = stopf0+stepf0
            GCId = np.diff(gci_s)
            avgGCIt[l] = np.mean(GCId/fs)
            varGCIt[l] = np.std(GCId/fs)
            NAQ, QOQ, T1, T2, H1H2, HRF = get_vq_params(
                gframe, dgframe, fs, gci_s)
            avgNAQt[l] = np.mean(NAQ)
            varNAQt[l] = np.std(NAQ)
            avgQOQt[l] = np.mean(QOQ)
            varQOQt[l] = np.std(QOQ)
            avgH1H2t[l] = np.mean(H1H2)
            varH1H2t[l] = np.std(H1H2)
            avgHRFt[l] = np.mean(HRF)
            varHRFt[l] = np.std(HRF)

        if static and len(rmwin) > 0:
            varGCIt = np.delete(varGCIt, rmwin)
            avgNAQt = np.delete(avgNAQt, rmwin)
            varNAQt = np.delete(varNAQt, rmwin)
            avgQOQt = np.delete(avgQOQt, rmwin)
            varQOQt = np.delete(varQOQt, rmwin)
            avgH1H2t = np.delete(avgH1H2t, rmwin)
            varH1H2t = np.delete(varH1H2t, rmwin)
            avgHRFt = np.delete(avgHRFt, rmwin)
            varHRFt = np.delete(varHRFt, rmwin)

        feat = np.stack((varGCIt, avgNAQt, varNAQt, avgQOQt,
                        varQOQt, avgH1H2t, varH1H2t, avgHRFt, varHRFt), axis=1)
        if static:
            feat=dynamic2static(feat)
            feat=np.expand_dims(feat, axis=0)
            head = self.head_st
        else:
            head=self.head_dyn

        if fmt in ("npy", "txt"):
            return feat

        elif fmt in ("dataframe", "csv"):
            df = {}
            for e, k in enumerate(head):
                df[k] = feat[:, e]
            return pd.DataFrame(df)
        elif fmt == "torch":
            return torch.from_numpy(feat)
        elif fmt == "kaldi":
            name_all = audio.split('/')
            dictX = {name_all[-1]: feat}
            save_dict_kaldimat(dictX, kaldi_file)

    def extract_features_path(self, path_audio, static=True, plots=False, fmt="npy", kaldi_file=""):
        """Extract the glottal features for audios inside a path

        :param path_audio: directory with (.wav) audio files inside, sampled at 16 kHz
        :param static: whether to compute and return statistic functionals over the feature matrix, or return the feature matrix computed over frames
        :param plots: timeshift to extract the features
        :param fmt: format to return the features (npy, dataframe, torch, kaldi)
        :param kaldi_file: file to store kaldifeatures, only valid when fmt=="kaldi"
        :returns: features computed from the audio file.

        >>> glottal=Glottal()
        >>> path_audio="../audios/"
        >>> features1=glottal.extract_features_path(path_audio, static=True, plots=False, fmt="npy")
        >>> features2=glottal.extract_features_path(path_audio, static=True, plots=False, fmt="csv")
        >>> features3=glottal.extract_features_path(path_audio, static=False, plots=True, fmt="torch")
        >>> glottal.extract_features_path(path_audio, static=False, plots=False, fmt="kaldi", kaldi_file="./test.ark")
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


if __name__ == "__main__":

    if len(sys.argv) != 6:
        print("python glottal.py <file_or_folder_audio> <file_features> <static (true, false)> <plots (true,  false)> <format (csv, txt, npy, kaldi, torch)>")
        sys.exit()

    glottal_o = Glottal()
    script_manager(sys.argv, glottal_o)
