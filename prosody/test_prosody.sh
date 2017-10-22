dir2=$(pwd)
python prosody.py  $dir2"/001_ddk1_PCGITA.wav" "featuresDDKst.txt" "static" "true"
python prosody.py  $dir2"/001_ddk1_PCGITA.wav" "featuresDDKdyn.txt" "dynamic" "true"
#python prosody.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Rainbow/" "featuresDDKdynFolder.txt" "dynamic" "false"
#python prosody.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Rainbow/" "featuresDDKstatFolder.txt" "static" "false"
