
from scipy.io.wavfile import read
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import math
import pysptk
import scipy.stats as st
import uuid
try:
    from .peakdetect import peakdetect
    from .GCI import SE_VQ_varF0, IAIF, get_vq_params

except:
    from peakdetect import peakdetect
    from GCI import SE_VQ_varF0, IAIF, get_vq_params

PATH=os.path.dirname(os.path.abspath(__file__))
sys.path.append('../')
from utils import dynamic2static, save_dict_kaldimat, get_dict
from kaldi_io import write_mat, write_vec_flt
from scipy.integrate import cumtrapz
from tqdm import tqdm
import pandas as pd
import torch
from script_mananger import script_manager


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

    Dynamic matrix is formed with the 9 descriptors computed for frames of 200 ms length with a time-shift of 100 ms.

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
        self.size_frame=0.2
        self.size_step=0.1
        self.head=["var GCI", "avg NAQ", "std NAQ", "avg QOQ", "std QOQ", "avg H1H2", "std H1H2", "avg HRF", "std HRF"]



    def plot_glottal(self, data_audio,fs,GCI, glottal_flow, glottal_sig, GCI_avg, GCI_std):
        """Plots of the glottal features

        :param data_audio: speech signal.
        :param fs: sampling frequency
        :param GCI: glottal closure instants
        :param glottal_flow: glottal flow
        :param glottal_sig: reconstructed glottal signal
        :param GCI_avg: average of the glottal closure instants
        :param GCI_std: standard deviation of the glottal closure instants
        :returns: plots of the glottal features.
        """
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
        plt.legend(ncol=2, loc=2)

        plt.subplot(313)
        plt.plot(t, glottal_flow, color='k', linewidth=2.0)
        plt.ylabel("Glotal flow derivative", fontsize=12)
        plt.xlabel('Time (s)', fontsize=12)
        plt.xlim([0, t[-1]])
        plt.grid(True)

        plt.show()

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
        if audio.find('.wav')==-1 and audio.find('.WAV')==-1:
            raise ValueError(audio+" is not a valid wav file")
        fs, data_audio=read(audio)
        data_audio=data_audio-np.mean(data_audio)
        data_audio=data_audio/float(np.max(np.abs(data_audio)))
        size_frameS=self.size_frame*float(fs)
        size_stepS=self.size_step*float(fs)
        overlap=size_stepS/size_frameS
        nF=int((len(data_audio)/size_frameS/overlap))-1
        data_audiof=np.asarray(data_audio*(2**15), dtype=np.float32)
        f0=pysptk.sptk.rapt(data_audiof, fs, int(0.01*fs), min=20, max=500, voice_bias=-0.2, otype='f0')
        sizef0=int(self.size_frame/0.01)
        stepf0=int(self.size_step/0.01)
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
                continue
                
            GCI=SE_VQ_varF0(data_frame,fs, f0=f0_frame)
            if GCI is None:
                print("------------- warning -------------------, not enought voiced segments were found to compute GCI")
            else:
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
                if plots:
                    self.plot_glottal(data_frame,fs,GCI, g_iaif, glottal, avgGCIt[l], varGCIt[l])

        if len(rmwin)>0:
            varGCIt=np.delete(varGCIt,rmwin)
            avgNAQt=np.delete(avgNAQt,rmwin)
            varNAQt=np.delete(varNAQt,rmwin)
            avgQOQt=np.delete(avgQOQt,rmwin)
            varQOQt=np.delete(varQOQt,rmwin)
            avgH1H2t=np.delete(avgH1H2t,rmwin)
            varH1H2t=np.delete(varH1H2t,rmwin)
            avgHRFt=np.delete(avgHRFt,rmwin)
            varHRFt=np.delete(varHRFt,rmwin)
        
        feat=np.stack((varGCIt, avgNAQt, varNAQt, avgQOQt, varQOQt, avgH1H2t, varH1H2t, avgHRFt, varHRFt), axis=1)

        if fmt=="npy" or fmt=="txt":
            if static:
                return dynamic2static(feat)
            else:
                return feat

        elif fmt=="dataframe" or fmt=="csv":
            if static:
                feat_st=dynamic2static(feat)
                head_st=[]
                df={}
                for k in ["global avg", "global std", "global skewness", "global kurtosis"]:
                    for h in self.head:
                        head_st.append(k+" "+h)
                for e, k in enumerate(head_st):
                    df[k]=[feat_st[e]]
                            
                return pd.DataFrame(df)
            else:
                df={}
                for e, k in enumerate(self.head):
                    df[k]=feat[:,e]
                return pd.DataFrame(df)
        elif fmt=="torch":
            if static:
                feat_s=dynamic2static(feat)
                feat_t=torch.from_numpy(feat_s)
                return feat_t
            else:
                return torch.from_numpy(feat)
        elif fmt=="kaldi":
            if static:
                raise ValueError("Kaldi is only supported for dynamic features")
            else:
                name_all=audio.split('/')
                dictX={name_all[-1]:feat}
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
        if fmt=="npy" or fmt=="txt":
            return Features
        elif fmt=="dataframe" or fmt=="csv":
            if static:
                head_st=[]
                df={}
                for k in ["global avg", "global std", "global skewness", "global kurtosis"]:
                    for h in self.head:
                        head_st.append(k+" "+h)
                for e, k in enumerate(head_st):
                    df[k]=Features[:,e]
            else:
                df={}
                for e, k in enumerate(self.head):
                    df[k]=Features[:,e]
            df["id"]=ids
            return pd.DataFrame(df)
        elif fmt=="torch":
            return torch.from_numpy(Features)
        elif fmt=="kaldi":
            if static:
                raise ValueError("Kaldi is only supported for dynamic features")
            else:
                dictX=get_dict(Features, ids)
                save_dict_kaldimat(dictX, kaldi_file)


if __name__=="__main__":

    if len(sys.argv)!=6:
        print("python glottal.py <file_or_folder_audio> <file_features> <static (true, false)> <plots (true,  false)> <format (csv, txt, npy, kaldi, torch)>")
        sys.exit()

    glottal=Glottal()
    script_manager(sys.argv, glottal)
