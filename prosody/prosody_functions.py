
import numpy as np



def V_UV(F0, data_audio, fs, type_seg, size_stepS):
    if type_seg=="Voiced":
        pitch_seg = np.where(F0!=0)[0]
    elif type_seg=="Unvoiced":
        pitch_seg = np.where(F0==0)[0]
    dchange = np.diff(pitch_seg)
    change = np.where(dchange>1)[0]
    ini = pitch_seg[0]
    init_seg= (pitch_seg[0]*size_stepS)+size_stepS
    segment=[]
    for indx in change:
        endv = pitch_seg[indx]+1
        end_seg = (pitch_seg[indx]*size_stepS)+size_stepS#To compute energy
        seg = data_audio[int(init_seg):int(end_seg)]#To compute energy
        segment.append(seg)

        ini= pitch_seg[indx+1]
        init_seg = (pitch_seg[indx+1]*size_stepS)+size_stepS#To compute energy

    return segment



def E_cont(seg,size_frameS,size_stepS,overlap):
    #Compute energy contour
    loge = []
    nF=int((len(seg)/size_frameS/overlap))-1
    for l in range(nF):
        data_frame = seg[int(l*size_stepS):int(l*size_stepS+size_frameS)]
        loge.append(logEnergy(data_frame))
    return loge

def logEnergy(sig):
    sig2=np.power(sig,2)
    sumsig2=np.sum(np.absolute(sig2))/len(sig2)

    logE=10*np.log10(sumsig2)
    if np.isnan(logE) or np.isinf(logE):
        logE=-1e30

    return logE
