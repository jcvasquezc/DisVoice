## Prosody features

```sh
prosody.py
```

Compute prosody features from continuous speech based on duration, fundamental frequency and energy.

Static or dynamic features can be computed:

The static feaature vector is formed with 13 features and include

1. Average fundamental frequency in voiced segments
2. Standard deviation of fundamental frequency in Hz
3. Variablity of fundamental frequency in semitones
4. Maximum of the fundamental frequency in Hz
5. Average energy in dB
6. Standard deviation of energy in dB
7. Maximum energy
8. Voiced rate (number of voiced segments per second)
9. Average duration of voiced segments
10. Standard deviation of duration of voiced segments
11. Pause rate (number of pauses per second)
12. Average duration of pauses
13. Standard deviation of duration of pauses
 <br /> NEW MEASURES <br />
14. Average tilt of fundamental frequency
15. Tilt regularity of fundamental frequency
16. Mean square error of the reconstructed F0 with a  1-degree polynomial
17. (Silence duration)/(Voiced+Unvoiced durations)
18. (Voiced duration)/(Unvoiced durations)
19. (Unvoiced duration)/(Voiced+Unvoiced durations)
20. (Voiced duration)/(Voiced+Unvoiced durations)
21. (Voiced duration)/(Silence durations)
22. (Unvoiced duration)/(Silence durations)
23. Unvoiced duration Regularity
24. Unvoiced energy Regularity
25. Voiced duration Regularity
26. Voiced energy Regularity
27. Pause duration Regularity
28. Maximum duration of voiced segments
29. Maximum duration of unvoiced segments
30. Minimum duration of voiced segments
31. Minimum duration of unvoiced segments
32. rate (# of voiced segments) / (# of unvoiced segments)
33. Average tilt of energy contour
34. Regression coefficient between the energy contour and a linear regression
35. Mean square error of the reconstructed energy contour with a  1-degree polynomial
34. Regression coefficient between the F0 contour and a linear regression
37. Average Delta energy within consecutive voiced segments
38. Standard deviation of Delta energy within consecutive voiced segments


The dynamic feature matrix is formed with 13 features computed for each voiced segment and contains:

- 1 Duration of the voiced segment
- 2-7. Coefficients of 5-degree Lagrange polynomial to model F0 contour
- 8-13. Coefficients of 5-degree Lagrange polynomial to model energy contour

Dynamic prosody features are based on
Najim Dehak, "Modeling Prosodic Features With Joint Factor Analysis for Speaker Verification", 2007

### Notes:

1. The fundamental frequency is computed using Praat. To use the RAPT algorithm change the "pitch method" variable in the function phonation_vowel.

2. When Kaldi output is set to "true" two files will be generated, the ".ark" with the data in binary format and the ".scp" Kaldi script file

### Runing
Script is called as follows
```sh
python prosody.py <file_or_folder_audio> <file_features.txt> [dynamic_or_static (default static)] [plots (true or false) (default false)] [kaldi output (true or false) (default false)]
```

### Examples:
```sh
python prosody.py "./001_ddk1_PCGITA.wav" "featuresDDKst.txt" "static" "true"
python prosody.py "./001_ddk1_PCGITA.wav" "featuresDDKdyn.txt" "dynamic" "true"
python prosody.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Rainbow/" "featuresDDKdynFolder.txt" "dynamic" "false"
python prosody.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Rainbow/" "featuresDDKstatFolder.txt" "static" "false"
python prosody.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Rainbow/" "featuresDDKdynFolder.txt" "dynamic" "false" "true"

```

#### Results:

Prosody analysis from continuous speech static
![Image](https://github.com/jcvasquezc/DisVoice/blob/master/images/prosody1.png?Raw=true)

![Image](https://github.com/jcvasquezc/DisVoice/blob/master/images/prosody3.png?Raw=true)


Prosody analysis from continuous speech dynamic
![Image](https://github.com/jcvasquezc/DisVoice/blob/master/images/prosody2.png?raw=true)

#### References

[[1]](http://ieeexplore.ieee.org/abstract/document/4291597/). N., Dehak, P. Dumouchel, and P. Kenny. "Modeling prosodic features with joint factor analysis for speaker verification." IEEE Transactions on Audio, Speech, and Language Processing 15.7 (2007): 2095-2103.

[[2]](http://www.sciencedirect.com/science/article/pii/S105120041730146X). J. R. Orozco-Arroyave, J. C. Vásquez-Correa et al. "NeuroSpeech: An open-source software for Parkinson's speech analysis." Digital Signal Processing (2017).
