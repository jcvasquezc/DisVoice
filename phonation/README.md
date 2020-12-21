## Phonation features


### Perturbation features

```sh
phonation.py
```
Compute phonation-based features from sustained vowels and continuous speech utterances.

For continuous speech, the features are computed over voiced segments.

Seven descriptors are computed:

1. First derivative of the fundamental Frequency
2. Second derivative of the fundamental Frequency
3. Jitter
4. Shimmer
5. Amplitude perturbation quotient
6. Pitch perturbation quotient
7. Logaritmic Energy

Additionally, static (for all utterance) or dynamic (frame by frame) features can be computed:

- The static feature vector is formed with 28 features and contains (seven descriptors) x (4 functionals: mean, std, skewness, kurtosis)

- The dynamic feature matrix is formed with the seven descriptors computed for frames of 40 ms with a time-shift of 20 ms.

### Notes:

1. For the dynamic features, the first 11 frames of each recording are not considered to be able to stack the APQ and PPQ descriptors with the remaining ones.

2. The fundamental frequency is computed the RAPT algorithm. To use the PRAAT method,  change the "self.pitch method" variable in the class constructor.

#### Running

Script is called as follows

```sh
python phonation.py <file_or_folder_audio> <file_features> <static (true or false)> <plots (true or false)> <format (csv, txt, npy, kaldi, torch)>
```

#### Examples:

Extract features in the command line


```sh
python phonation.py "../audios/001_a1_PCGITA.wav" "phonationfeaturesAst.txt" "true" "true" "txt"
python phonation.py "../audios/098_u1_PCGITA.wav" "phonationfeaturesUst.csv" "true" "true" "csv"
python phonation.py "../audios/098_u1_PCGITA.wav" "phonationfeaturesUdyn.pt" "false" "true" "torch"

python phonation.py "../audios/" "phonationfeaturesst.txt" "true" "false" "txt"
python phonation.py "../audios/" "phonationfeaturesst.csv" "true" "false" "csv"
python phonation.py "../audios/" "phonationfeaturesdyn.pt" "false" "false" "torch"

KALDI_ROOT=/home/camilo/Camilo/codes/kaldi-master2
export PATH=$PATH:$KALDI_ROOT/src/featbin/
python phonation.py "../audios/098_u1_PCGITA.wav" "phonationfeaturesUdyn" "false" "false" "kaldi"

python phonation.py "../audios/" "phonationfeaturesdyn" "false" "false" "kaldi"
```

Extract features directly in Python


```python
from phonation import Phonation
phonation=Phonation()
file_audio="../audios/001_a1_PCGITA.wav"
features1=phonation.extract_features_file(file_audio, static=True, plots=True, fmt="npy")
features2=phonation.extract_features_file(file_audio, static=True, plots=True, fmt="dataframe")
features3=phonation.extract_features_file(file_audio, static=False, plots=True, fmt="torch")
phonation.extract_features_file(file_audio, static=False, plots=False, fmt="kaldi", kaldi_file="./test")
```

[Jupyter notebook](https://github.com/jcvasquezc/DisVoice/blob/master/notebooks_examples/phonation_features.ipynb)

#### Results:

Phonation analysis from a sustained vowel
!![Image](https://github.com/jcvasquezc/DisVoice/blob/master/images/phonation_vowel.png?raw=true)

Phonation analysis from continuous speech
!![Image](https://github.com/jcvasquezc/DisVoice/blob/master/images/phonation_continuous.png?raw=true)


#### References

[[1]](https://link.springer.com/article/10.1007%2Fs12559-017-9497-x) T. Arias-Vergara, J. C. Vásquez-Correa, J. R. Orozco-Arroyave, Parkinson’s Disease and Aging: Analysis of Their Effect in Phonation and Articulation of Speech, Cognitive computation, (2017).

