import os, sys
PATH=os.path.dirname(os.path.realpath(__file__))

PATH_DISVOICE=os.path.dirname(os.path.realpath(__file__))+"/disvoice/"
sys.path.append(PATH_DISVOICE)

import disvoice.articulation.articulation as articulation


def test_extract_articulation1():
    feature_extractor=articulation.Articulation()
    file_audio=PATH+"/../audios/001_ddk1_PCGITA.wav"
    features1=feature_extractor.extract_features_file(file_audio, static=True, plots=True, fmt="npy")
    print(features1.shape)

def test_extract_articulation2():
    feature_extractor=articulation.Articulation()
    path_audio=PATH+"/../audios/"
    features2=feature_extractor.extract_features_path(path_audio, static=True, plots=False, fmt="csv")

    print(features2.head())


def test_extract_articulation3():
    feature_extractor=articulation.Articulation()
    file_audio=PATH+"/../audios/001_ddk1_PCGITA.wav"
    features3=feature_extractor.extract_features_file(file_audio, static=False, plots=True, fmt="torch")
    print(features3.size())

if __name__ == "__main__":
    test_extract_articulation1()
    test_extract_articulation2()
    test_extract_articulation3()