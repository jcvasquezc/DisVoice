python phonological.py "../../audios/001_ddk1_PCGITA.wav" "phonologicalfeaturesDDKst.txt" "true" "true" "txt"
python phonological.py "../../audios/001_ddk1_PCGITA.wav" "phonologicalfeaturesDDKst.csv" "true" "true" "csv"
python phonological.py "../../audios/001_ddk1_PCGITA.wav" "phonologicalfeaturesDDKdyn.pt" "false" "true" "torch"

python phonological.py "../../audios/" "phonologicalfeaturesst.txt" "true" "false" "txt"
python phonological.py "../../audios/" "phonologicalfeaturesst.csv" "true" "false" "csv"
python phonological.py "../../audios/" "phonologicalfeaturesdyn.pt" "false" "false" "torch"
python phonological.py "../../audios/" "phonologicalfeaturesdyn.csv" "false" "false" "csv"

KALDI_ROOT=/home/camilo/Camilo/codes/kaldi-master2
export PATH=$PATH:$KALDI_ROOT/src/featbin/
python phonological.py "../../audios/001_ddk1_PCGITA.wav" "phonologicalfeaturesDDKdyn" "false" "false" "kaldi"

python phonological.py "../../audios/" "phonologicalfeaturesdyn" "false" "false" "kaldi"