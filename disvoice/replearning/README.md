## Features based on Representation learning strategies using Autoencoders


```sh
replearning.py
```
Compute Features based on Representation learning strategies using convolutional and recurrent Autoencoders

Two types of features are computed

1. 256 features extracted from the bottleneck layer of the autoencoders
2. 128 features based on the MSE between the decoded and input spectrograms of the autoencoder in different frequency regions


Additionally, static (for all utterance) or dynamic (for each 500 ms speech segments) features can be computed:

- The static feature vector is formed with 1536 features and contains (384 descriptors) x (4 functionals: mean, std, skewness, kurtosis)

- The dynamic feature matrix is formed with the 384 descriptors computed for speech segments with 500ms length and 250ms time-shift

- You can choose between features computed from a convolutional or recurrent autoencoder


### Notes:

1. Detailed information is found in 

Vasquez-Correa, J. C., et al. (2020). Parallel Representation Learning for the Classification of Pathological Speech: Studies on Parkinson’s Disease and Cleft Lip and Palate. Speech Communication, 122, 56-67.

2. Additional trained models can be found in 

https://github.com/jcvasquezc/AEspeech


#### Running

Script is called as follows

```sh
python replearning.py <file_or_folder_audio> <file_features> <static (true or false)> <plots (true or false)> <format (csv, txt, npy, kaldi, torch)> <model (CAE, RAE)>
```

#### Examples:

Extract features in the command line


```sh
python replearning.py "../audios/001_ddk1_PCGITA.wav" "replearningfeaturesDDKst.txt" "true" "true" "txt" "CAE"
python replearning.py "../audios/001_ddk1_PCGITA.wav" "replearningfeaturesDDKdyn.pt" "false" "true" "torch" "CAE"

python replearning.py "../audios/" "replearningfeaturesst.txt" "true" "false" "txt" "CAE"
python replearning.py "../audios/" "replearningfeaturesst.csv" "true" "false" "csv" "CAE"
python replearning.py "../audios/" "replearningfeaturesdyn.pt" "false" "false" "torch" "CAE"

KALDI_ROOT=/home/camilo/Camilo/codes/kaldi-master2
export PATH=$PATH:$KALDI_ROOT/src/featbin/
python replearning.py "../audios/001_ddk1_PCGITA.wav" "replearningfeaturesDDKdyn" "false" "false" "kaldi" "CAE"

python replearning.py "../audios/" "replearningfeaturesdyn" "false" "false" "kaldi" "CAE"
```

Extract features directly in Python


```python
from replearning import RepLearning
replearning=RepLearning('CAE')
file_audio="../audios/001_a1_PCGITA.wav"
features1=replearning.extract_features_file(file_audio, static=True, plots=True, fmt="npy")
features2=replearning.extract_features_file(file_audio, static=True, plots=True, fmt="dataframe")
features3=replearning.extract_features_file(file_audio, static=False, plots=True, fmt="torch")
replearning.extract_features_file(file_audio, static=False, plots=False, fmt="kaldi", kaldi_file="./test")
```

[Jupyter notebook](https://github.com/jcvasquezc/DisVoice/blob/master/notebooks_examples/replearning_features.ipynb)

#### Results:


Input and decoded spectrograms from the autoencoders

![Image](https://github.com/jcvasquezc/DisVoice/blob/master/images/replearning_continuous.png?raw=true)


Reconstruction error in different frequency bands

![Image](https://github.com/jcvasquezc/DisVoice/blob/master/images/replearning_error.png?raw=true)



#### References

[1] Vasquez-Correa, J. C., et al. (2020). Parallel Representation Learning for the Classification of Pathological Speech: Studies on Parkinson’s Disease and Cleft Lip and Palate. Speech Communication, 122, 56-67.

