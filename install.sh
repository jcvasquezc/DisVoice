pip install pysptk

sudo apt-get install praat

git submodule init
git submodule update
echo For Kaldi output Kaldi must be installed beforehand and the path at kaldi-io/kaldiio.py:
echo line 14: os.environ['KALDI_ROOT']='/mnt/matylda5/iveselyk/Tools/kaldi-trunk'
echo should be changed to with path to the proper Kaldi root directory.


# Optional for GUI
pip install dash==0.21.1  # The core dash backend
pip install dash-renderer==0.12.1  # The dash front-end
pip install dash-html-components==0.10.1  # HTML components
pip install dash-core-components==0.23.0  # Supercharged components
pip install plotly --upgrade  # Plotly graphing library used in examples
pip install sounddevice
