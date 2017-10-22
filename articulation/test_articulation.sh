dir2=$(pwd)
python articulation.py $dir2"/001_ddk1_PCGITA.wav" "featuresDDKst.txt" "static" "true"
python articulation.py $dir2"/001_ddk1_PCGITA.wav" "featuresDDKdyn.txt" "dynamic" "true"
#python articulation.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Rainbow/" "featuresDDKdynFolder.txt" "dynamic" "false"
#python articulation.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Rainbow/" "featuresDDKstatFolder.txt" "static" "false"
