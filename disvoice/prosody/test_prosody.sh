python Prosody.py "../audios/001_ddk1_PCGITA.wav" "prosodyfeaturesDDKst.txt" "true" "true" "txt"
python Prosody.py "../audios/001_ddk1_PCGITA.wav" "prosodyfeaturesDDKst.csv" "true" "true" "csv"
python Prosody.py "../audios/001_ddk1_PCGITA.wav" "prosodyfeaturesDDKdyn.pt" "false" "true" "torch"

python Prosody.py "../audios/" "prosodyfeaturesst.txt" "true" "false" "txt"
python Prosody.py "../audios/" "prosodyfeaturesst.csv" "true" "false" "csv"
python Prosody.py "../audios/" "prosodyfeaturesdyn.pt" "false" "false" "torch"
python Prosody.py "../audios/" "prosodyfeaturesdyn.csv" "false" "false" "csv"

KALDI_ROOT=/home/camilo/Camilo/codes/kaldi-master2
export PATH=$PATH:$KALDI_ROOT/src/featbin/
python Prosody.py "../audios/001_ddk1_PCGITA.wav" "prosodyfeaturesDDKdyn" "false" "false" "kaldi"

python Prosody.py "../audios/" "prosodyfeaturesdyn" "false" "false" "kaldi"
