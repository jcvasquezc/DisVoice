## Articulation features

```sh
articulation.py
```
Compute articulation features from continuous speech.

122 descriptors are computed:

- 1 - 22. Bark band energies in onset transitions (22 BBE).
- 23 - 34. Mel frequency cepstral coefficients in onset transitions (12 MFCC onset)
- 35 - 46. First derivative of the MFCCs in onset transitions (12 DMFCC onset)
- 47 - 58. Second derivative of the MFCCs in onset transitions (12 DDMFCC onset)
- 59 - 80. Bark band energies in offset transitions (22 BBE).
- 81 - 92. MFCCC in offset transitions (12 MFCC offset)
- 93 - 104. First derivative of the MFCCs in offset transitions (12 DMFCC offset)
- 105 - 116. Second derivative of the MFCCs in offset transitions (12 DMFCC offset)
- 117 First formant Frequency
- 118 First Derivative of the first formant frequency
- 119 Second Derivative of the first formant frequency
- 120 Second formant Frequency
- 121 First derivative of the Second formant Frequency
- 122 Second derivative of the Second formant Frequency

In addition, static (for all utterance) or dynamic (at-frame level) features can be computed:

- The static feature vector is formed with 488 features (122 descriptors) x (4 functionals: mean, std, skewness, kurtosis)

- For the dynamic analysis, two matrices are created both for onset and offset based features
-- Dynamic matrices are formed with the 58 descriptors (22 BBEs, 12 MFCC, 12DMFCC, 12 DDMFCC ) computed for frames of 40 ms.

The first two frames of each recording are not considered for dynamic analysis to be able to stack the derivatives of MFCCs

#### Notes:
- The fundamental frequency is used to detect the transitions and it is computed using Praat by default. To use the RAPT algorithm change the "pitch method" variable in the function articulation_continuous.
- The formant frequencies are computed using Praat
- When Kaldi output is set to "true" two files will be generated, the ".ark" with the data in binary format and the ".scp" Kaldi script file

#### Running
```sh
python articulation.py <file_or_folder_audio> <file_features.txt> [dynamic_or_static (default static)] [plots (true or false) (default false)]
```

#### Examples:

```sh
python articulation.py "./001_ddk1_PCGITA.wav" "featuresDDKst.txt" "static" "true"
python articulation.py "./001_ddk1_PCGITA.wav" "featuresDDKdyn.txt" "dynamic" "true"
python articulation.py "./001_ddk1_PCGITA.wav" "featuresDDKdyn.txt" "dynamic" "true" "true"
python articulation.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Rainbow/" "featuresDDKdynFolder.txt" "dynamic" "false"
python articulation.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Rainbow/" "featuresDDKstatFolder.txt" "static" "false"
python articulation.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Rainbow/" "featuresDDKstatFolder.txt" "dynamic" "false" "true"
```
#### Results:

Articulation analysis from continuous speech
![Image](https://github.com/jcvasquezc/DisVoice/blob/master/images/articulation_continuousFormants.png?raw=True)

#### References

[[1]](http://ieeexplore.ieee.org/abstract/document/7472927/) J. R. Orozco-Arroyave, J. C. Vásquez-Correa, et, al. Towards an automatic monitoring of the neurological state of Parkinson's patients from speech. In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), (2016)
