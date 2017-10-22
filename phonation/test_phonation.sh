
dir2=$(pwd)
python phonation.py $dir2"/001_a1_PCGITA.wav" "featuresAst.txt" "static" "true"
python phonation.py $dir2"/001_a1_PCGITA.wav" "featuresAdyn.txt" "dynamic" "true"
#python phonation.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Ah/" "featuresAdynFolder.txt" "dynamic" "false"
#python phonation.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Ah/" "featuresAstatFolder.txt" "static" "false"

python phonation.py $dir2"/001_ddk1_PCGITA.wav" "featuresDDKst.txt" "static" "true"
python phonation.py $dir2"/001_ddk1_PCGITA.wav" "featuresDDKdyn.txt" "dynamic" "true"
#python phonation.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Rainbow/" "featuresDDKdynFolder.txt" "dynamic" "false"
#python phonation.py "/home/camilo/Camilo/data/BDKayElemetrics/Norm/Rainbow/" "featuresDDKstatFolder.txt" "static" "false"
