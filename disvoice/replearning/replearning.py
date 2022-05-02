import os
import sys

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"

PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(PATH, '..'))
sys.path.append(PATH)
from utils import save_dict_kaldimat, get_dict
from AEspeech import AEspeech
from script_mananger import script_manager
import torch
from tqdm import tqdm

class RepLearning:
    """
    Feature extraction from speech signals based on representation learning strategies using convolutional and recurrent autoencoders

    Two types of features are computed

    1. 256 features extracted from the bottleneck layer of the autoencoders
    2. 128 features based on the MSE between the decoded and input spectrograms of the autoencoder in different frequency regions


    Additionally, static (for all utterance) or dynamic (for each 500 ms speech segments) features can be computed:
    - The static feature vector is formed with 1024 features and contains (384 descriptors) x (4 functionals: mean, std, skewness, kurtosis)
    - The dynamic feature matrix is formed with the 384 descriptors computed for speech segments with 500ms length and 250ms time-shift
    - You can choose between features computed from a convolutional or recurrent autoencoder

    Script is called as follows

    >>> python replearning.py <file_or_folder_audio> <file_features> <static (true or false)> <plots (true or false)> <format (csv, txt, npy, kaldi, torch)> <model (CAE, RAE)>

    Examples command line:

    >>> python replearning.py "../audios/001_ddk1_PCGITA.wav" "replearningfeaturesDDKst.txt" "true" "true" "txt" "CAE"
    >>> python replearning.py "../audios/001_ddk1_PCGITA.wav" "replearningfeaturesDDKdyn.pt" "false" "true" "torch" "CAE"

    >>> python replearning.py "../audios/" "replearningfeaturesst.txt" "true" "false" "txt" "CAE"
    >>> python replearning.py "../audios/" "replearningfeaturesst.csv" "true" "false" "csv" "CAE"
    >>> python replearning.py "../audios/" "replearningfeaturesdyn.pt" "false" "false" "torch" "CAE"

    Examples directly in Python

    >>> from replearning import RepLearning
    >>> replearning=RepLearning('CAE')
    >>> file_audio="../audios/001_a1_PCGITA.wav"
    >>> features1=replearning.extract_features_file(file_audio, static=True, plots=True, fmt="npy")
    >>> features2=replearning.extract_features_file(file_audio, static=True, plots=True, fmt="dataframe")
    >>> features3=replearning.extract_features_file(file_audio, static=False, plots=True, fmt="torch")
    >>> replearning.extract_features_file(file_audio, static=False, plots=False, fmt="kaldi", kaldi_file="./test")
    """

    def __init__(self, model):
        self.size_bottleneck=256
        self.AEspeech=AEspeech(model, self.size_bottleneck)
        self.statistics=["mean", "std", "skewness", "kurtosis"]

        feat_names_bottle=["bottleneck_"+str(k) for k in range(self.size_bottleneck)]
        feat_names_error=["error_"+str(k) for k in range(128)]
        feat_names_bottle_all=[]
        feat_names_error_all=[]

        for k in self.statistics:
            for j in feat_names_bottle:
                feat_names_bottle_all.append(k+"_"+j)
            for j in feat_names_error:
                feat_names_error_all.append(k+"_"+j)

        self.head_st=feat_names_bottle_all+feat_names_error_all
        self.head_dyn=np.hstack(feat_names_bottle+feat_names_error)


    def extract_features_file(self, audio, static=True, plots=False, fmt="npy", kaldi_file=""):
        """
        Extract the representation learning features from an audio file

        :param audio: .wav audio file.
        :param static: whether to compute and return statistic functionals over the feature matrix, or return the feature matrix computed over frames
        :param plots: timeshift to extract the features
        :param fmt: format to return the features (npy, dataframe, torch, kaldi)
        :param kaldi_file: file to store kaldi features, only valid when fmt=="kaldi"
        :returns: features computed from the audio file.

        >>> replearning=RepLearning('CAE')
        >>> file_audio="../audios/001_ddk1_PCGITA.wav"
        >>> features1=replearning.extract_features_file(file_audio, static=True, plots=True, fmt="npy")
        >>> features2=replearning.extract_features_file(file_audio, static=True, plots=True, fmt="dataframe")
        >>> features3=replearning.extract_features_file(file_audio, static=False, plots=True, fmt="torch")
        >>> replearning.extract_features_file(file_audio, static=False, plots=False, fmt="kaldi", kaldi_file="./test")

        >>> replearning=RepLearning('CAE')
        >>> path_audio="../audios/"
        >>> features1=replearning.extract_features_path(path_audio, static=True, plots=False, fmt="npy")
        >>> features2=replearning.extract_features_path(path_audio, static=True, plots=False, fmt="csv")
        >>> features3=replearning.extract_features_path(path_audio, static=False, plots=True, fmt="torch")
        >>> replearning.extract_features_path(path_audio, static=False, plots=False, fmt="kaldi", kaldi_file="./test.ark")

        """
        if static and fmt=="kaldi":
            raise ValueError("Kaldi is only supported for dynamic features")
        hb=self.AEspeech.compute_bottleneck_features(audio)
        err=self.AEspeech.compute_rec_error_features(audio)
        if plots:
            self.AEspeech.plot_spectrograms(audio)
        if static:
            bottle_feat=np.hstack((np.mean(hb, 0), np.std(hb, 0), st.skew(hb, 0), st.kurtosis(hb, 0)))
            error_feat=np.hstack((np.mean(err, 0), np.std(err, 0), st.skew(err, 0), st.kurtosis(err, 0)))
            feat=np.hstack((bottle_feat, error_feat))
            feat=np.expand_dims(feat, axis=0)
            head=self.head_st
        else:
            feat=np.concatenate((hb, err), axis=1)
            head=self.head_dyn
        if fmt in("npy","txt"):
            return feat
        elif fmt in("dataframe","csv"):
            dff={}
            for e, key in enumerate(head):
                dff[key]=feat[:,e]
            dff=pd.DataFrame(dff)
            return dff
        elif fmt=="torch":
            return torch.from_numpy(feat)
        elif fmt=="kaldi":
            name_all=audio.split('/')
            dictX={name_all[-1]:feat}
            save_dict_kaldimat(dictX, kaldi_file)
        else:
            raise ValueError(fmt+" is not supported")

    def extract_features_path(self, path_audio, static=True, plots=False, fmt="npy", kaldi_file=""):
        """
        Extract the representation learning features for audios inside a path
        
        :param path_audio: directory with (.wav) audio files inside, sampled at 16 kHz
        :param static: whether to compute and return statistic functionals over the feature matrix, or return the feature matrix computed over frames
        :param plots: timeshift to extract the features
        :param fmt: format to return the features (npy, dataframe, torch, kaldi)
        :param kaldi_file: file to store kaldifeatures, only valid when fmt=="kaldi"
        :returns: features computed from the audio file.

        >>> replearning=RepLearning('CAE')
        >>> path_audio="../audios/"
        >>> features1=phonological.replearning(path_audio, static=True, plots=False, fmt="npy")
        >>> features2=phonological.replearning(path_audio, static=True, plots=False, fmt="csv")
        >>> features3=phonological.replearning(path_audio, static=False, plots=True, fmt="torch")
        >>> replearning.extract_features_path(path_audio, static=False, plots=False, fmt="kaldi", kaldi_file="./test.ark")
        """
        if static and fmt=="kaldi":
            raise ValueError("Kaldi is only supported for dynamic features")
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

        if static:
            head=self.head_st
        else:
            head=self.head_dyn

        if fmt in("npy","txt"):
            return Features
        if fmt in("dataframe","csv"):
            df={}
            for e, k in enumerate(head):
                df[k]=Features[:,e]
            df["id"]=ids
            return pd.DataFrame(df)
        if fmt=="torch":
            return torch.from_numpy(Features)
        if fmt=="kaldi":
            dictX=get_dict(Features, ids)
            save_dict_kaldimat(dictX, kaldi_file)
        else:
            raise ValueError(fmt+" is not supported")


if __name__=="__main__":

    if len(sys.argv)!=7:
        print("python replearning.py <file_or_folder_audio> <file_features> <static (true, false)> <plots (true,  false)> <format (csv, txt, npy, kaldi, torch)> <model (CAE,RAE)>")
        sys.exit()

    replearning=RepLearning(sys.argv[-1])
    script_manager(sys.argv, replearning)
