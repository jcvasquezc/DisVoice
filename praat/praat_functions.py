
import os
import numpy as np
def multi_find(s, r):
    s_len = len(s)
    r_len = len(r)
    _complete = []
    if s_len < r_len:
        n = -1
    else:
        for i in range(s_len):
            # search for r in s until not enough characters are left
            if s[i:i + r_len] == r:
                _complete.append(i)
            else:
                i = i + 1
    return(_complete)

def praat_vuv(audio_filaname, resultsp, resultst, time_stepF0=0, minf0=75, maxf0=600, maxVUVPeriod=0.02, averageVUVPeriod=0.01):
	command='praat ../praat/vuv_praat.praat '
	command+=audio_filaname+' '+resultsp +' '+  resultst+' '
	command+=str(minf0)+' '+str(maxf0)+' '
	command+=str(time_stepF0)+' '+str(maxVUVPeriod)+' '+str(averageVUVPeriod)
	os.system(command)

def praat_formants(audio_filename, results_filename,sizeframe,step, n_formants=5, max_formant=5500):
    command='praat ../praat/FormantsPraat.praat '
    command+=audio_filename + ' '+results_filename+' '
    command+=str(n_formants)+' '+ str(max_formant) + ' '
    command+=str(float(sizeframe)/2)+' '
    command+=str(float(step))
    os.system(command) #formant extraction praat

def read_textgrid_trans(file_textgrid, data_audio, fs, win_trans=0.04):
	segments=[]
	segments_onset=[]
	segments_offset=[]
	prev_line=""
	with open(file_textgrid) as fp:
		for line in fp:
			line = line.strip('\n')
			if line=='"V"' or line == '"U"':
				prev_trans=line
				transVal=int(float(prev_line)*fs)-1
				segment=data_audio[int(transVal-win_trans*fs):int(transVal+win_trans*fs)]
				segments.append(segment)
				if prev_trans=='"U"' or prev_trans=="":
					segments_onset.append(segment)
				elif prev_trans=='"V"':
					segments_offset.append(segment)
			prev_line=line
	return segments,segments_onset,segments_offset

def decodeF0(fileTxt):
	pitch_data=np.loadtxt(fileTxt)
	time_voiced=pitch_data[:,0] # First column is the time stamp vector
	pitch=np.log(pitch_data[:,1]) # Second column
	return pitch, time_voiced

def decodeFormants(fileTxt):
    fid=open(fileTxt)
    datam=fid.read()
    end_line1=multi_find(datam, '\n')
    F1=[]
    F2=[]
    ji=10
    while (ji<len(end_line1)-1):
        line1=datam[end_line1[ji]+1:end_line1[ji+1]]
        cond=(line1=='3' or line1=='4' or line1=='5')
        if (cond):
            F1.append(float(datam[end_line1[ji+1]+1:end_line1[ji+2]]))
            F2.append(float(datam[end_line1[ji+3]+1:end_line1[ji+4]]))
        ji=ji+1
    F1=np.asarray(F1)
    F2=np.asarray(F2)
    return F1, F2
