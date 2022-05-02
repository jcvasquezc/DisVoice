

python glottal.py "../../audios/001_a1_PCGITA.wav" "glottalfeaturesAst.txt" "true" "true" "txt"
python glottal.py "../../audios/098_u1_PCGITA.wav" "glottalfeaturesUst.csv" "true" "true" "csv"
python glottal.py "../../audios/098_u1_PCGITA.wav" "glottalfeaturesUdyn.pt" "false" "true" "torch"

python glottal.py "../../audios/" "glottalfeaturesst.txt" "true" "false" "txt"
python glottal.py "../../audios/" "glottalfeaturesst.csv" "true" "false" "csv"
python glottal.py "../../audios/" "glottalfeaturesdyn.pt" "false" "false" "torch"

KALDI_ROOT=/home/camilo/Camilo/codes/kaldi-master2
export PATH=$PATH:$KALDI_ROOT/src/featbin/
python glottal.py "../../audios/098_u1_PCGITA.wav" "glottalfeaturesUdyn" "false" "false" "kaldi"

python glottal.py "../../audios/" "glottalfeaturesdyn" "false" "false" "kaldi"
