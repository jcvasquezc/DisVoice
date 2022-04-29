

python Glottal.py "../audios/001_a1_PCGITA.wav" "glottalfeaturesAst.txt" "true" "true" "txt"
python Glottal.py "../audios/098_u1_PCGITA.wav" "glottalfeaturesUst.csv" "true" "true" "csv"
python Glottal.py "../audios/098_u1_PCGITA.wav" "glottalfeaturesUdyn.pt" "false" "true" "torch"

python Glottal.py "../audios/" "glottalfeaturesst.txt" "true" "false" "txt"
python Glottal.py "../audios/" "glottalfeaturesst.csv" "true" "false" "csv"
python Glottal.py "../audios/" "glottalfeaturesdyn.pt" "false" "false" "torch"

KALDI_ROOT=/home/camilo/Camilo/codes/kaldi-master2
export PATH=$PATH:$KALDI_ROOT/src/featbin/
python Glottal.py "../audios/098_u1_PCGITA.wav" "glottalfeaturesUdyn" "false" "false" "kaldi"

python Glottal.py "../audios/" "glottalfeaturesdyn" "false" "false" "kaldi"
