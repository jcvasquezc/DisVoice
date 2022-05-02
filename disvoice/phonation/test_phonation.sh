

python phonation.py "../../audios/001_a1_PCGITA.wav" "phonationfeaturesAst.txt" "true" "true" "txt"
python phonation.py "../../audios/098_u1_PCGITA.wav" "phonationfeaturesUst.csv" "true" "true" "csv"
python phonation.py "../../audios/098_u1_PCGITA.wav" "phonationfeaturesUdyn.pt" "false" "true" "torch"

python phonation.py "../../audios/" "phonationfeaturesst.txt" "true" "false" "txt"
python phonation.py "../../audios/" "phonationfeaturesst.csv" "true" "false" "csv"
python phonation.py "../../audios/" "phonationfeaturesdyn.pt" "false" "false" "torch"

KALDI_ROOT=/home/camilo/Camilo/codes/kaldi-master2
export PATH=$PATH:$KALDI_ROOT/src/featbin/
python phonation.py "../../audios/098_u1_PCGITA.wav" "phonationfeaturesUdyn" "false" "false" "kaldi"

python phonation.py "../../audios/" "phonationfeaturesdyn" "false" "false" "kaldi"