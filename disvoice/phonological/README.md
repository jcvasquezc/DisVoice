## Phonological features


### Phonological features

```sh
phonological.py
```
Compute phonological features from continuous speech files.

18 descriptors are computed, bases on 18 different phonological classes from the phonet toolkit 
https://phonet.readthedocs.io/en/latest/?badge=latest

It computes the phonological log-likelihood ratio features from phonet

Static or dynamic matrices can be computed:

Static matrix is formed with 108 features formed with (18 descriptors) x (6 functionals: mean, std, skewness, kurtosis, max, min)

Dynamic matrix is formed with the 18 descriptors computed for frames of 25 ms with a time-shift of 10 ms.


#### Running

Script is called as follows

```sh
python phonological.py <file_or_folder_audio> <file_features> <static (true or false)> <plots (true or false)> <format (csv, txt, npy, kaldi, torch)>
```

#### Examples:

Extract features in the command line


```sh
python phonological.py "../audios/001_ddk1_PCGITA.wav" "phonologicalfeaturesAst.txt" "true" "true" "txt"
python phonological.py "../audios/001_ddk1_PCGITA.wav" "phonologicalfeaturesUdyn.pt" "false" "true" "torch"

python phonological.py "../audios/" "phonologicalfeaturesst.txt" "true" "false" "txt"
python phonological.py "../audios/" "phonologicalfeaturesst.csv" "true" "false" "csv"
python phonological.py "../audios/" "phonologicalfeaturesdyn.pt" "false" "false" "torch"
python phonological.py "../audios/" "phonologicalfeaturesdyn.csv" "false" "false" "csv"

KALDI_ROOT=/home/camilo/Camilo/codes/kaldi-master2
export PATH=$PATH:$KALDI_ROOT/src/featbin/
python phonological.py "../audios/001_ddk1_PCGITA.wav" "phonologicalfeaturesddk1dyn" "false" "false" "kaldi"

python phonological.py "../audios/" "phonologicalfeaturesdyn" "false" "false" "kaldi"
```

Extract features directly in Python


```python
phonological=Phonological()
file_audio="../audios/001_ddk1_PCGITA.wav"
features1=phonological.extract_features_file(file_audio, static=True, plots=True, fmt="npy")
features2=phonological.extract_features_file(file_audio, static=True, plots=True, fmt="dataframe")
features3=phonological.extract_features_file(file_audio, static=False, plots=True, fmt="torch")
phonological.extract_features_file(file_audio, static=False, plots=False, fmt="kaldi", kaldi_file="./test")

path_audio="../audios/"
features1=phonological.extract_features_path(path_audio, static=True, plots=False, fmt="npy")
features2=phonological.extract_features_path(path_audio, static=True, plots=False, fmt="csv")
features3=phonological.extract_features_path(path_audio, static=False, plots=True, fmt="torch")
phonological.extract_features_path(path_audio, static=False, plots=False, fmt="kaldi", kaldi_file="./test.ark")
```

[Jupyter notebook](https://github.com/jcvasquezc/DisVoice/blob/master/notebooks_examples/phonological_features.ipynb)

#### Results:



Phonological analysis
!![Image](https://github.com/jcvasquezc/DisVoice/blob/master/images/phonological1.png?raw=true)
!![Image](https://github.com/jcvasquezc/DisVoice/blob/master/images/phonological2.png?raw=true)


#### References

[[1]](https://gita.udea.edu.co/uploads/1405-Phonet.pdf) Vásquez-Correa, J. C., Klumpp, P., Orozco-Arroyave, J. R., & Nöth, E. (2019). Phonet: A Tool Based on Gated Recurrent Neural Networks to Extract Phonological Posteriors from Speech. In INTERSPEECH (pp. 549-553).

[2] Diez, M., Varona, A., Penagarikano, M., Rodriguez-Fuentes, L. J., & Bordel, G. (2014). On the projection of PLLRs for unbounded feature distributions in spoken language recognition. IEEE Signal Processing Letters, 21(9), 1073-1077.