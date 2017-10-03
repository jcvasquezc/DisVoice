pip install pysptk
sudo apt-get install praat
git submodule init
git submodule update
echo For Kaldi output Kaldi must be installed beforehand and the path at kaldi-io/kaldiio.py:
echo line 14: os.environ['KALDI_ROOT']='/mnt/matylda5/iveselyk/Tools/kaldi-trunk'
echo should be changed to with path to the proper Kaldi root directory.
