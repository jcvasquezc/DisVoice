.. disvoice documentation master file, created by
   sphinx-quickstart on Sat Mar  9 04:39:38 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Disvoice's documentation!
-------------------------------------

.. image:: ../logos/disvoice_logo.png
  :width: 400
  :alt: Disvoice logo


DisVoice is a python framework designed to compute features from speech files. Disvoice computes glottal, phonation, articulation, prosody, phonological, and features representation learnig strategies using autoencders. The features can be computed both from sustained vowels and continuous speech utterances with the aim to recognize praliguistic aspects from speech.

The features can be used in classifiers to recognize emotions, or communication capabilities of patients with different speech disorders including diseases with functional origin such as larinx cancer or nodules; craneo-facial based disorders such as hipernasality developed by cleft-lip and palate; or neurodegenerative disorders such as Parkinson's or Hungtinton's diseases.

The features are also suitable to evaluate mood problems like depression based on speech patterns.

For additional details about each feature type, and how to use DisVoice, please check




.. toctree::
   :maxdepth: 3

   Glottal
   Phonation
   Articulation
   Prosody
   Phonological
   RepLearning
   help
   reference

Installation
-------------------------------------

From the source file::

    git clone https://github.com/jcvasquezc/disvoice
    cd disvoice
    bash install.sh


Indices and tables
-------------------------------------
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



Help
-------------------------------------
If you have trouble with Disvoice, please write to Camilo Vasquez at: juan.vasquez@fau.de