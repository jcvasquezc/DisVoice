

python articulation.py "../../audios/001_ddk1_PCGITA.wav" "articulationfeaturesDDKst.txt" "true" "true" "txt"
python articulation.py "../../audios/001_ddk1_PCGITA.wav" "articulationfeaturesDDKst.csv" "true" "true" "csv"
python articulation.py "../../audios/001_ddk1_PCGITA.wav" "articulationfeaturesDDKdyn.pt" "false" "true" "torch"

python articulation.py "../../audios/" "articulationfeaturesst.txt" "true" "false" "txt"
python articulation.py "../../audios/" "articulationfeaturesst.csv" "true" "false" "csv"
python articulation.py "../../audios/" "articulationfeaturesdyn.pt" "false" "false" "torch"
python articulation.py "../../audios/" "articulationfeaturesdyn.csv" "false" "false" "csv"

KALDI_ROOT=/home/camilo/Camilo/codes/kaldi-master2
export PATH=$PATH:$KALDI_ROOT/src/featbin/
python articulation.py "../../audios/001_ddk1_PCGITA.wav" "articulationfeaturesDDKdyn" "false" "false" "kaldi"

python articulation.py "../../audios/" "articulationfeaturesdyn" "false" "false" "kaldi"