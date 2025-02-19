
import os
import numpy as np
import sys
import parselmouth
PATH = os.path.dirname(os.path.realpath(__file__))


sys.path.append(PATH)

def multi_find(s, r):
	"""
	Internal function used to decode the Formants file generated by Praat.
	"""
	s_len = len(s)
	r_len = len(r)
	_complete = []
	for i in range(s_len):
		# search for r in s until not enough characters are left
		if s[i:i + r_len] == r:
			_complete.append(i)
		else:
			i = i + 1
	return _complete

def praat_vuv(audio_filaname, resultsp, resultst, time_stepF0=0, minf0=75, maxf0=600, maxVUVPeriod=0.02, averageVUVPeriod=0.01):
	"""
	runs vuv_praat script to obtain pitch and voicing decisions for a wav file.
	It writes the results into two text files, one for the pitch and another
	for the voicing decisions. These results can then be read using the function
	read_textgrid_trans and decodeF0

	:param audio_filaname: Full path to the wav file
	:param resultsp: Full path to the resulting file with the pitch
	:param resultst: Full path to the resulting file with the voiced/unvoiced decisions
	:param time_stepF0: time step to compute the pitch, default value is 0 and Praat will use 0.75 / minf0
	:param minf0: minimum frequency for the pitch in Hz, default is 75Hz
	:param maxf0: maximum frequency for the pitch in Hz, default is 600
	:param maxVUVPeriod: maximum interval that considered part of a larger voiced interval, default 0.02
	:param averageVUVPeriod: half of this value will be taken to be the amount to which a voiced interval will extend beyond its initial and final points, default is 0.01
	:returns: nothing
	"""
	if sys.platform.find('win')>=0:
		command='/praat.exe --run '+PATH+'/vuv_praat.praat '
	else:
		command='praat --run '+PATH+'/vuv_praat.praat '
	command+=audio_filaname+' '+resultsp +' '+  resultst+' '
	command+=str(minf0)+' '+str(maxf0)+' '
	command+=str(time_stepF0)+' '+str(maxVUVPeriod)+' '+str(averageVUVPeriod)
	os.system(command)




def praat_formants(audio_filename):
	"""
	Extract F1 and F2 formants from the given audio file using Praat (via Parselmouth).
	
	:param audio_file: Path to the audio file (WAV format).
	:returns: Tuple of F1 and F2 arrays
	"""
	# Load the sound
	snd = parselmouth.Sound(audio_filename)
	
	# Extract the formant object using Burg's method
	formant = snd.to_formant_burg(time_step=0.02, max_number_of_formants=5, maximum_formant=5500)

	# Prepare arrays to store the formant frequencies
	formant_list_f1 = []
	formant_list_f2 = []
	
	# Loop over the duration of the sound to get formants for each time slice
	for t in np.arange(0, snd.duration, 0.02):  # 0.02 is the time step
		try:
			f1 = formant.get_value_at_time(1, t)  # F1 (first formant)
			f2 = formant.get_value_at_time(2, t)  # F2 (second formant)
			formant_list_f1.append(f1)
			formant_list_f2.append(f2)
		except:
			formant_list_f1.append(None)
			formant_list_f2.append(None)

	# Convert to numpy arrays, handling possible None values
	f1_array = np.array([f if f is not None else np.nan for f in formant_list_f1])
	f2_array = np.array([f if f is not None else np.nan for f in formant_list_f2])

	# Filter out NaN values in F1 and F2
	f1_filtered = f1_array[np.isfinite(f1_array)]
	f2_filtered = f2_array[np.isfinite(f2_array)]

	return f1_filtered, f2_filtered



def read_textgrid_trans(file_textgrid, data_audio, fs, win_trans=0.04):
	"""
	This function reads a text file with the text grid with voiced/unvoiced
	decisions then finds the onsets (unvoiced -> voiced) and
	offsets (voiced -> unvoiced) and then reads the audio data to returns
	lists of segments of lenght win_trans around these transitions.

	:param file_textgrid: The text file with the text grid with voicing decisions.
	:param data_audio: the audio signal.
	:param fs: sampling frequency of the audio signal.
	:param win_trans: the transition window lenght, default 0.04
	:returns segments: List with both onset and offset transition segments.
	:returns segments_onset: List with onset transition segments
	:returns segments_offset: List with offset transition segments
	"""
	segments=[]
	segments_onset=[]
	segments_offset=[]
	prev_trans=""
	prev_line=0
	with open(file_textgrid) as fp:
		for line in fp:
			line = line.strip('\n')
			if line in ('"V"', '"U"'):
				transVal=int(float(prev_line)*fs)-1
				segment=data_audio[int(transVal-win_trans*fs):int(transVal+win_trans*fs)]
				segments.append(segment)
				if prev_trans in ('"V"', ""):
					segments_onset.append(segment)
				elif prev_trans=='"U"':
					segments_offset.append(segment)
				prev_trans=line
			prev_line=line
	return segments,segments_onset,segments_offset

def decodeF0(fileTxt,len_signal=0, time_stepF0=0):
	"""
	Reads the content of a pitch file created with praat_vuv function.
	By default it will return the contents of the file in two arrays,
	one for the actual values of pitch and the other with the time stamps.
	Optionally the lenght of the signal and the time step of the pitch
	values can be provided to return an array with the full pitch contour
	for the signal, with padded zeros for unvoiced segments.

	:param fileTxt: File with the pitch, which can be generated using the function praat_vuv
	:param len_signal: Lenght of the audio signal in
	:param time_stepF0: The time step of pitch values. Optional.
	:returns pitch: Numpy array with the values of the pitch.
	:returns time_voiced: time stamp for each pitch value.
	"""
	if os.stat(fileTxt).st_size==0:
		return np.array([0]), np.array([0])
	pitch_data=np.loadtxt(fileTxt)
	if len(pitch_data.shape)>1:
		time_voiced=pitch_data[:,0] # First column is the time stamp vector
		pitch=pitch_data[:,1] # Second column
	elif len(pitch_data.shape)==1: # Only one point of data
		time_voiced=pitch_data[0] # First datum is the time stamp
		pitch=pitch_data[1] # Second datum is the pitch value
	if len_signal>0:
		n_frames=int(len_signal/time_stepF0)
		t=np.linspace(0.0,len_signal,n_frames)
		pitch_zeros=np.zeros(int(n_frames))
		if len(pitch_data.shape)>1:
			for idx,time_p in enumerate(time_voiced):
				argmin=np.argmin(np.abs(t-time_p))
				pitch_zeros[argmin]=pitch[idx]
		else:
			argmin=np.argmin(np.abs(t-time_voiced))
			pitch_zeros[argmin]=pitch
		return pitch_zeros, t
	return pitch, time_voiced
