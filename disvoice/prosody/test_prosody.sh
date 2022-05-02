python prosody.py "../../audios/001_ddk1_PCGITA.wav" "prosodyfeaturesDDKst.txt" "true" "true" "txt"
python prosody.py "../../audios/001_ddk1_PCGITA.wav" "prosodyfeaturesDDKst.csv" "true" "true" "csv"
python prosody.py "../../audios/001_ddk1_PCGITA.wav" "prosodyfeaturesDDKdyn.pt" "false" "true" "torch"

python prosody.py "../../audios/" "prosodyfeaturesst.txt" "true" "false" "txt"
python prosody.py "../../audios/" "prosodyfeaturesst.csv" "true" "false" "csv"
python prosody.py "../../audios/" "prosodyfeaturesdyn.pt" "false" "false" "torch"
python prosody.py "../../audios/" "prosodyfeaturesdyn.csv" "false" "false" "csv"

KALDI_ROOT=/home/camilo/Camilo/codes/kaldi-master2
export PATH=$PATH:$KALDI_ROOT/src/featbin/
python prosody.py "../../audios/001_ddk1_PCGITA.wav" "prosodyfeaturesDDKdyn" "false" "false" "kaldi"

python prosody.py "../../audios/" "prosodyfeaturesdyn" "false" "false" "kaldi"
