# DisVoice

[![Documentation Status](https://readthedocs.org/projects/disvoice/badge/?version=latest)](https://disvoice.readthedocs.io/en/latest/?badge=latest)

![Image](https://github.com/jcvasquezc/DisVoice/blob/master/docs/logos/disvoice_logo.png?raw=true)

DisVoice is a python framework designed to compute features from speech files. Disvoice computes glottal, phonation, articulation, prosody, phonological, and features representation learnig strategies using autoencders. The features can be computed both from sustained vowels and continuous speech utterances with the aim to recognize praliguistic aspects from speech.

The features can be used in classifiers to recognize emotions, or communication capabilities of patients with different speech disorders including diseases with functional origin such as larinx cancer or nodules; craneo-facial based disorders such as hipernasality developed by cleft-lip and palate; or neurodegenerative disorders such as Parkinson's or Hungtinton's diseases.

The features are also suitable to evaluate mood problems like depression based on speech patterns.

For additional details about each feature type, and how to use DisVoice, please check

- [glottal](https://github.com/jcvasquezc/DisVoice/tree/master/glottal)
- [phonation](https://github.com/jcvasquezc/DisVoice/tree/master/phonation)
- [articulaton](https://github.com/jcvasquezc/DisVoice/tree/master/articulation)
- [prosody](https://github.com/jcvasquezc/DisVoice/tree/master/prosody) 
- [phonological](https://github.com/jcvasquezc/DisVoice/tree/master/phonological)
- [Representation learning](https://github.com/jcvasquezc/DisVoice/tree/master/replearning)


### Install

Praat should be installed first, and the executable file should be added as an environmental variable. 

For linux

```
apt-get install praat
pip install disvoice
```

or

```python setup.py install```

For Windows

Donwload the latest version of Praat from https://www.fon.hum.uva.nl/praat/download_win.html

and add the path file to the environment variables

Then

```
pip install disvoice
```

or

```python setup.py install```



Kaldi must be installed beforehand for Kaldi output  

## Reference

If you use Disvoice for research purposes, please cite the following papers, depending on the features you use:

## Glottal features

[1] Belalcázar-Bolaños, E. A., Orozco-Arroyave, J. R., Vargas-Bonilla, J. F., Haderlein, T., & Nöth, E. (2016, September). Glottal Flow Patterns Analyses for Parkinson’s Disease Detection: Acoustic and Nonlinear Approaches. In International Conference on Text, Speech, and Dialogue (pp. 400-407). Springer.


## Phonation features

[1] T. Arias-Vergara, J. C. Vásquez-Correa, J. R. Orozco-Arroyave, Parkinson's Disease and Aging: Analysis of Their Effect in Phonation and Articulation of Speech, Cognitive computation, (2017).

[2] Vásquez-Correa, J. C., et al. "Towards an automatic evaluation of the dysarthria level of patients with Parkinson's disease." Journal of communication disorders 76 (2018): 21-36.

## Articulation features

[1] Vásquez-Correa, J. C., et al. "Towards an automatic evaluation of the dysarthria level of patients with Parkinson's disease." Journal of communication disorders 76 (2018): 21-36.

[2]. J. R. Orozco-Arroyave, J. C. Vásquez-Correa et al. "NeuroSpeech: An open-source software for Parkinson's speech analysis." Digital Signal Processing (2017).

## Prosody features

[1]. N., Dehak, P. Dumouchel, and P. Kenny. "Modeling prosodic features with joint factor analysis for speaker verification." IEEE Transactions on Audio, Speech, and Language Processing 15.7 (2007): 2095-2103.

[2] Vásquez-Correa, J. C., et al. "Towards an automatic evaluation of the dysarthria level of patients with Parkinson's disease." Journal of communication disorders 76 (2018): 21-36.

## Phonological features

[1] Vásquez-Correa, J. C., et al (2019). Phonet: a Tool Based on Gated Recurrent Neural Networks to Extract Phonological Posteriors from Speech. Proc. Interspeech 2019, 549-553.

## Representaton learning-based features

[1] Vasquez-Correa, J. C., et al. (2020). Parallel Representation Learning for the Classification of Pathological Speech: Studies on Parkinson’s Disease and Cleft Lip and Palate. Speech Communication, 122, 56-67.


License
----

MIT
