import os, sys
PATH=os.path.dirname(os.path.realpath(__file__))

PATH_DISVOICE=os.path.dirname(os.path.realpath(__file__))+"/disvoice/"
sys.path.append(PATH_DISVOICE)

import disvoice.phonological.phonological as phonological


def test_extract_phonological1():
    feature_extractor=phonological.Phonological()
    file_audio=PATH+"/../audios/098_u1_PCGITA.wav"
    features1=feature_extractor.extract_features_file(file_audio, static=True, plots=True, fmt="npy")
    print(features1.shape)

def test_extract_phonological2():
    feature_extractor=phonological.Phonological()
    path_audio=PATH+"/../audios/"
    features2=feature_extractor.extract_features_path(path_audio, static=True, plots=False, fmt="csv")

    print(features2.head())


def test_extract_phonological3():
    feature_extractor=phonological.Phonological()
    file_audio=PATH+"/../audios/098_u1_PCGITA.wav"
    features3=feature_extractor.extract_features_file(file_audio, static=False, plots=True, fmt="torch")
    print(features3.size())

if __name__ == "__main__":
    test_extract_phonological1()
    test_extract_phonological2()
    test_extract_phonological3()