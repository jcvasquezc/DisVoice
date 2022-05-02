
import math
import numpy as np
import pysptk

def bark(f):
    x=(f*0.00076)
    x2=(f/7500)**2
    b=[]
    for i in range (0,len(f)):
        b.append(13*( math.atan(x[i]) )+3.5*( math.atan(x2[i]))) #Bark scale values
    return (b)

def barke(x,Fs, nfft=2048, nB=25):
    """
    e: Energy in frequency bands according to the Bark scale
    x: signal
    Fs: sampling frequency
    nfft: number of points for the Fourier transform
    nB: number of bands for energy computation
    """
    eps = 1e-30
    y = fftsolp(x,nfft)
    f = (Fs/2)*(np.linspace(0,1,int(nfft/2+1)))
    barkScale = bark(f)
    barkIndices = []
    for i in range (0,len(barkScale)):
        barkIndices.append(int(barkScale[i]))

    barkIndices = np.asarray(barkIndices)

    barkEnergy=[]
    for i in range (nB):
        brk = np.nonzero(barkIndices==i)
        brk = np.asarray(brk)[0]
        sizeb=len(brk)
        if (sizeb>0):
            barkEnergy.append(sum(np.abs(y[brk]))/sizeb)
        else:
            barkEnergy.append(0)

    e = np.asarray(barkEnergy)+eps
    e = np.log(e)
    return e


def fftsolp(x,nfft):
    """
    STFT for compute the energy in Bark scale
    x: signal
    nffft: number of points of the Fourier transform
    """
    window = np.hamming(len(x)/4)
    noverlap = np.ceil(len(window)/2)

    nx = len(x)
    nwind = len(window)

    ncol = np.fix((nx-noverlap)/(nwind-noverlap))
    ncol = int(ncol)
    colindex = (np.arange(0,ncol))*(nwind-noverlap)
    colindex = colindex.astype(int)

    rowindex = np.arange(0,nwind)
    rowindex = rowindex.astype(int)
    rowindex = rowindex[np.newaxis]
    rowindex = rowindex.T
    d = np.ones((nwind,ncol),dtype=np.int)
    y = x[d*(rowindex+colindex)]
    window = window.astype(float)
    window = window[np.newaxis]
    window = window.T
    new = window*d
    y = new*y
    y = y[:,0]

    y = np.fft.fft(y,nfft)
    y = (y[0:int(nfft/2+1)])
    return y

    
def extract_transitions(segments, fs, size_frameS, size_stepS, nB=22, nMFCC=12, nfft=2048):
    frames=[]
    size_frame_full=int(2**np.ceil(np.log2(size_frameS)))
    fill=int(size_frame_full-size_frameS)
    overlap=size_stepS/size_frameS
    for j in range(len(segments)):
        if (len(segments[j])>size_frameS):
            nF=int((len(segments[j])/size_frameS)/overlap)-1
            for iF in range(nF):
                frames.append(np.hamming(size_frameS)*segments[j][int(iF*size_stepS):int(iF*size_stepS+size_frameS)])

    BarkEn=np.zeros((len(frames),nB))
    MFCC=np.zeros((len(frames),nMFCC))
    for j in range(len(frames)):
        frame_act=np.hstack((frames[j], np.zeros(fill)))
        BarkEn[j,:]=barke(frame_act,fs, nfft, nB)
        MFCC[j,:]=pysptk.sptk.mfcc(frame_act, order=nMFCC, fs=fs, alpha=0.97, num_filterbanks=32, cepslift=22, use_hamming=True)
    return BarkEn, MFCC


def get_transition_segments(F0, data_audio, fs, transition, size_tran=0.04):
    segment=[]
    time_stepF0=int(len(data_audio)/len(F0))

    for j in range(2, len(F0)):
        if transition=="onset":
            condition=F0[j-1]==0 and F0[j]!=0
        elif transition=="offset":
            condition=F0[j-1]!=0 and F0[j]==0
        if condition:
            border=j*time_stepF0
            initframe=int(border-size_tran*fs)
            endframe=int(border+size_tran*fs)
            segment.append(data_audio[initframe:endframe])
    return segment
