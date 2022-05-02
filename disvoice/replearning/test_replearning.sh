

python replearning.py "../../audios/001_ddk1_PCGITA.wav" "replearningfeaturesDDKst.txt" "true" "true" "txt" "CAE"
python replearning.py "../../audios/001_ddk1_PCGITA.wav" "replearningfeaturesDDKdyn.pt" "false" "true" "torch" "CAE"

python replearning.py "../../audios/" "replearningfeaturesst.txt" "true" "false" "txt" "CAE"
python replearning.py "../../audios/" "replearningfeaturesst.csv" "true" "false" "csv" "CAE"
python replearning.py "../../audios/" "replearningfeaturesdyn.pt" "false" "false" "torch" "CAE"

KALDI_ROOT=/home/camilo/Camilo/codes/kaldi-master2
export PATH=$PATH:$KALDI_ROOT/src/featbin/
python replearning.py "../../audios/001_ddk1_PCGITA.wav" "replearningfeaturesDDKdyn" "false" "false" "kaldi" "CAE"

python replearning.py "../../audios/" "replearningfeaturesdyn" "false" "false" "kaldi" "CAE"