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

Additionally, static (for all utterance) or dynamic (frame ny frame) features can be computed:

- The static feature vector is formed with 29 features and contains (seven descriptors) x (4 functionals: mean, std, skewness, kurtosis) + degree of Unvoiced

- The dynamic feature matrix is formed with the seven descriptors computed for frames of 40 ms with a time-shift of 20 ms.

### Notes:

1. For the dynamic features, the first 11 frames of each recording are not considered to be able to stack the APQ and PPQ descriptors with the remaining ones.

2. The fundamental frequency (F0) is computed using the RAPT algorithm

3. When Kaldi output is set to "true" two files will be generated, the ".ark" with the data in binary format and the ".scp" Kaldi script file

Script is called as follows

```sh
python phonation.py <file_or_folder_audio> <file_features.txt> [dynamic_or_static (default static)] [plots (true or false) (default false)] [kaldi output (true or false) (default false)]
```

#### Examples:
```sh
python phonation.py "./001_a1_PCGITA.wav" "featuresAst.txt" "static" "true"
python phonation.py "./001_a1_PCGITA.wav" "featuresAdyn.txt" "dynamic" "true"
python phonation.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Ah/" "featuresAdynFolder.txt" "dynamic" "false"
python phonation.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Ah/" "featuresAstatFolder.txt" "static" "false"
python phonation.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Ah/" "featuresAdynFolder.txt" "dynamic" "false" "true"

python phonation.py "./001_ddk1_PCGITA.wav" "featuresDDKst.txt" "static" "true"
python phonation.py "./001_ddk1_PCGITA.wav" "featuresDDKdyn.txt" "dynamic" "true"
python phonation.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Rainbow/" "featuresDDKdynFolder.txt" "dynamic" "false"
python phonation.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Rainbow/" "featuresDDKstatFolder.txt" "static" "false"
python phonation.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Rainbow/" "featuresDDKstatFolder.txt" "dynamice" "false" "true"
```
#### Results:

Phonation analysis from a sustained vowel
!![Image](https://github.com/jcvasquezc/DisVoice/blob/master/images/phonation_vowel.png?raw=true)

Phonation analysis from continuous speech
!![Image](https://github.com/jcvasquezc/DisVoice/blob/master/images/phonation_continuous.png?raw=true)


#### References

[[1]](https://link.springer.com/article/10.1007%2Fs12559-017-9497-x) T. Arias-Vergara, J. C. Vásquez-Correa, J. R. Orozco-Arroyave, Parkinson’s Disease and Aging: Analysis of Their Effect in Phonation and Articulation of Speech, Cognitive computation, (2017).


### new

### Glottal source features

```sh
glottal.py
```

Compute phonation features derived from the glottal source reconstruction from sustained vowels.

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

Dynamic matrix is formed with the 9 descriptors computed for frames of 200 ms length.

Notes:

1. The fundamental frequency is computed using the RAPT algorithm.

2. When Kaldi output is set to "true" two files will be generated, the ".ark" with the data in binary format and the ".scp" Kaldi script file.

Script is called as follows

```sh
python glottal.py <file_or_folder_audio> <file_features.txt> [dynamic_or_static (default static)] [plots (true or false) (default false)] [kaldi output (true or false) (default false)]
```

#### Examples:
```sh
python glottal.py "./001_a1_PCGITA.wav" "glottalfeaturesAst.txt" "static" "true" "false"
python glottal.py "./098_u1_PCGITA.wav" "glottalfeaturesUst.txt" "static" "true" "false"
python glottal.py "./001_a1_PCGITA.wav" "glottalfeaturesAdyn.txt" "dynamic" "true" "false"
python glottal.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Ah/" "glottalfeaturesAdynFolder.txt" "dynamic" "false" "false"
python glottal.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Ah/" "glottalfeaturesAstatFolder.txt" "static" "false" "false"
python glottal.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Ah/" "glottalfeaturesAdynFolder.ark" "dynamic" "false" "true"
```

#### Results:

Glottal analysis from a sustained vowel
!![Image](https://github.com/jcvasquezc/DisVoice/blob/master/images/glottal_vowel.png?raw=true)


#### References

[[1]](https://link.springer.com/chapter/10.1007/978-3-319-45510-5_46) Belalcázar-Bolaños, E. A., Orozco-Arroyave, J. R., Vargas-Bonilla, J. F., Haderlein, T., & Nöth, E. (2016, September). Glottal Flow Patterns Analyses for Parkinson’s Disease Detection: Acoustic and Nonlinear Approaches. In International Conference on Text, Speech, and Dialogue (pp. 400-407). Springer.
