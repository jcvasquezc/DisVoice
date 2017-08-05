## Phonation features

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

Additionally, static (for all utterance) or dynamic (frame ny frame) features can be computed:

- The static feature vector is formed with 29 features and contains (seven descriptors) x (4 functionals: mean, std, skewness, kurtosis) + degree of Unvoiced

- The dynamic feature matrix is formed with the seven descriptors computed for frames of 40 ms with a time-shift of 20 ms.

### Notes:

1. For the dynamic features, the first 11 frames of each recording are not considered to be able to stack the APQ and PPQ descriptors with the remaining ones.

2. The fundamental frequency (F0) is computed using the RAPT algorithm

Script is called as follows

```sh
python phonation.py <file_or_folder_audio> <file_features.txt> [dynamic_or_static (default static)] [plots (true or false) (default false)]
```

#### Examples:
```sh
python phonation.py "./001_a1_PCGITA.wav" "featuresAst.txt" "static" "true"
python phonation.py "./001_a1_PCGITA.wav" "featuresAdyn.txt" "dynamic" "true"
python phonation.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Ah/" "featuresAdynFolder.txt" "dynamic" "false"
python phonation.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Ah/" "featuresAstatFolder.txt" "static" "false"

python phonation.py "./001_ddk1_PCGITA.wav" "featuresDDKst.txt" "static" "true"
python phonation.py "./001_ddk1_PCGITA.wav" "featuresDDKdyn.txt" "dynamic" "true"
python phonation.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Rainbow/" "featuresDDKdynFolder.txt" "dynamic" "false"
python phonation.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Rainbow/" "featuresDDKstatFolder.txt" "static" "false"
```
#### Results:

Phonation analysis from a sustained vowel
![Image](images/phonation_vowel.png)

Phonation analysis from continuous speech
![Image](images/phonation_continuous.png)
![Image](images/phonation_vowel.png)


#### References

[[1]](https://link.springer.com/article/10.1007%2Fs12559-017-9497-x) T. Arias-Vergara, J. C. Vásquez-Correa, J. R. Orozco-Arroyave, Parkinson’s Disease and Aging: Analysis of Their Effect in Phonation and Articulation of Speech, Cognitive computation, (2017).
