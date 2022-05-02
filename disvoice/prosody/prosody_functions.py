
import numpy as np
import scipy.stats as st
from sklearn.metrics import mean_squared_error


def F0feat(F0):

    F0nz=F0[F0!=0]
    tsf0=0.01
    if len(F0nz)==0:
        return np.zeros(30)

    F0mean=np.mean(F0nz)
    F0std=np.std(F0nz)
    F0max=np.max(F0nz)
    F0min=np.min(F0nz)
    F0skew=st.skew(F0nz)
    F0kurt=st.kurtosis(F0nz)
    tilt=[]
    mse=[]
    F0nzlist=np.split(F0, np.where(F0 == 0)[0]+1)
    F0nzlist=[F0nzlist[j] for j in range(len(F0nzlist) ) if len(F0nzlist[j])>1]
    F0nzlist=[F0nzlist[j][:-1] for j in  range(len(F0nzlist) )]
    for k in range(len(F0nzlist)):
        if len(F0nzlist[k])>1:
            t=np.arange(len(F0nzlist[k]))*tsf0
            pol=np.polyfit(t, F0nzlist[k],1)
            if not np.isnan(pol[0]):
                tilt.append(pol[0])
                f0rec=t*pol[0]+pol[1]
                msef0t=mean_squared_error(F0nzlist[k],f0rec)
                mse.append(msef0t)

    if len(tilt) == 0:
        tilt.append(0)            
        mse.append(0)
        
    tiltmean=np.mean(np.asarray(tilt))
    msemean=np.mean(np.asarray(mse))
    tiltstd=np.std(np.asarray(tilt))
    msestd=np.std(np.asarray(mse))
    tiltmax=np.max(np.asarray(tilt))
    msemax=np.max(np.asarray(mse))
    tiltmin=np.min(np.asarray(tilt))
    msemin=np.min(np.asarray(mse))
    tiltskw=st.skew(np.asarray(tilt))
    mseskw=st.skew(np.asarray(mse))
    tiltku=st.kurtosis(np.asarray(tilt))
    mseku=st.kurtosis(np.asarray(mse))

    f01mean=np.mean(F0nzlist[0])
    f01std=np.std(F0nzlist[0])
    f01max=np.max(F0nzlist[0])
    f01min=np.min(F0nzlist[0])
    f01skw=st.skew(F0nzlist[0])
    f01ku=st.kurtosis(F0nzlist[0])

    f0lmean=np.mean(F0nzlist[-1])
    f0lstd=np.std(F0nzlist[-1])
    f0lmax=np.max(F0nzlist[-1])
    f0lmin=np.min(F0nzlist[-1])
    f0lskw=st.skew(F0nzlist[-1])
    f0lku=st.kurtosis(F0nzlist[-1])
    featv=np.hstack([F0mean, F0std, F0max, F0min, F0skew, F0kurt, tiltmean, msemean, tiltstd, msestd, tiltmax, msemax, tiltmin, msemin,tiltskw, mseskw, tiltku, mseku, f01mean, f01std, f01max, f01min, f01skw, f01ku, f0lmean, f0lstd, f0lmax, f0lmin, f0lskw, f0lku])
    return featv



def energy_cont_segm(segments, size_frameS, size_stepS):

    E=[]
    for j in range(len(segments)):
        logE=[]
        segment_t=segments[j]
        overlap=size_stepS/size_frameS
        nF=int((len(segment_t)/size_frameS/overlap))-1
        if nF>1:
            for l in range(nF):
                data_frame=segment_t[int(l*size_stepS):int(l*size_stepS+size_frameS)]
                logE.append(logEnergy(data_frame))
            E.append(np.asarray(logE))
    return E



def polyf0(F0):
    tsf0=0.01

    F0rec=[]
    F0v=[]
    F0u=[]
    total=0
    for j in range(len(F0)-1):
        if F0[j]>0 and F0[j+1]==0:
            offset=True
            onset=False
            voiced=False
            Unvoiced=False
        elif F0[j]>0 and F0[j+1]>0:
            offset=False
            onset=False
            voiced=True
            Unvoiced=False
        elif F0[j]==0 and F0[j+1]==0:
            offset=False
            onset=False
            voiced=False
            Unvoiced=True
        elif F0[j]==0 and F0[j+1]>0:
            offset=False
            onset=True
            voiced=False
            Unvoiced=False

        if voiced:
            F0v.append(F0[j])
        elif Unvoiced:
            F0u.append(0)

        elif onset:
            F0v=[]
            F0u.append(0)
            total+=len(F0u)
            F0rec.append(F0u)
        elif offset:
            F0v.append(F0[j])
            F0u=[]
            t=np.arange(len(F0v))*tsf0
            if len(F0v)>1:
                pol=np.polyfit(t, F0v,1)
                f0rec=t*pol[0]+pol[1]
                F0rec.append(f0rec)
            else:
                F0rec.append(np.zeros(len(F0v)))
            total+=len(F0v)
        


    F0rec=np.hstack(F0rec)

    diffend=len(F0)-len(F0rec)

    f0fill=np.zeros(diffend)
    F0rec=np.hstack((F0rec, f0fill))

    return F0rec


def energy_feat(segments, fs, size_frameS, size_stepS):

    first_segment=0
    last_segment=len(segments)-1
    E_mean=[]
    E_std=[]
    E_skw=[]
    E_ku=[]
    tilt=[]
    mseE=[]
    if len(segments)<2:
        return np.zeros(24)

    for j in range(len(segments)):
        tsE=size_stepS/fs
        
        segment_t=segments[j]
        overlap=size_stepS/size_frameS
        log_energy_contour = get_energy_segment(size_frameS, size_stepS, segment_t, overlap)

        if j==first_segment and len(log_energy_contour)==0:
            first_segment+=1

        if j==last_segment and len(log_energy_contour)==0:
            last_segment-=1

        if len(log_energy_contour)<=1:
            continue
        E_mean.append(np.mean(log_energy_contour))
        E_std.append(np.std(log_energy_contour))
        E_skw.append(st.skew(log_energy_contour))
        E_ku.append(st.kurtosis(log_energy_contour))

        t=np.arange(len(log_energy_contour))*tsE
        pol=np.polyfit(t, log_energy_contour,1)
        if not np.isnan(pol[0]):
            tilt.append(pol[0])
            Erec=t*pol[0]+pol[1]
            mseE.append(mean_squared_error(log_energy_contour,Erec))
    
    E_mean=np.asarray(E_mean)
    E_std=np.asarray(E_std)
    E_skw=np.asarray(E_skw)
    E_ku=np.asarray(E_ku)
    tilt=np.asarray(tilt)
    mseE=np.asarray(mseE)
    if first_segment==len(segments):
        E0mean=0
        E0std=0
        E0max=0
        E0min=0
        E0skw=0
        E0ku=0
    else:
        segment_t=segments[first_segment]
        log_energy_contour = get_energy_segment(size_frameS, size_stepS, segment_t, overlap)
        E0mean=np.mean(log_energy_contour)
        E0std=np.std(log_energy_contour)
        E0max=np.max(log_energy_contour)
        E0min=np.min(log_energy_contour)
        E0skw=st.skew(log_energy_contour)
        E0ku=st.kurtosis(log_energy_contour)

    if last_segment==0:
        Elmean=0
        Elstd=0
        Elmax=0
        Elmin=0
        Elskw=0
        Elku=0
    else:
        segment_t=segments[last_segment]
        log_energy_contour = get_energy_segment(size_frameS, size_stepS, segment_t, overlap)

        if len(log_energy_contour)>0:
            Elmean=np.mean(log_energy_contour)
            Elstd=np.std(log_energy_contour)
            Elmax=np.max(log_energy_contour)
            Elmin=np.min(log_energy_contour)
            Elskw=st.skew(log_energy_contour)
            Elku=st.kurtosis(log_energy_contour)
        else:
            Elmean=0
            Elstd=0
            Elmax=0
            Elmin=0
            Elskw=0
            Elku=0

    feat_vec=np.hstack([np.mean(E_mean), np.mean(E_std), np.mean(E_skw), np.mean(E_ku), np.mean(tilt), np.std(tilt), st.skew(tilt), st.kurtosis(tilt), np.mean(mseE), np.std(mseE), st.skew(mseE), st.kurtosis(mseE), E0mean, E0std, E0max, E0min, E0skw, E0ku, Elmean, Elstd, Elmax, Elmin, Elskw, Elku])

    return feat_vec

def get_energy_segment(size_frameS, size_stepS, segment_t, overlap):
    nF=int((len(segment_t)/size_frameS/overlap))-1
    logE=[]
    for l in range(nF):
        data_frame=segment_t[int(l*size_stepS):int(l*size_stepS+size_frameS)]
        logE.append(logEnergy(data_frame))
    E=np.asarray(logE)
    return E



def dur_seg(segments, fs):
    dur=[]
    for j in range(len(segments)):
        dur.append(len(segments[j])/fs)
    dur=np.asarray(dur)
    if len(dur)>1:
        return np.asarray([np.mean(dur), np.std(dur), st.skew(dur), st.kurtosis(dur), np.max(dur), np.min(dur)])
    return np.zeros(6)




def duration_feat(segV, segU, segPause, signal, fs):
    Vrate=fs*float(len(segV))/len(signal)
    
    durV=dur_seg(segV, fs)
    durU=dur_seg(segU, fs)
    durP=dur_seg(segPause, fs)

    if durP[0]!=0 and durU[0]!=0 and durV[0]!=0: 
        PVU=durP[0]/(durV[0]+durU[0])
        PU=durP[0]/durU[0]
        UVU=durU[0]/(durV[0]+durU[0])
        VVU=durV[0]/(durV[0]+durU[0])
        VP=durV[0]/durP[0]
        UP=durU[0]/durP[0]
        dur_ratio=np.hstack([PVU, PU, UVU, VVU, VP, UP])
    else:
        dur_ratio=np.zeros(6)

    feat_vec=np.hstack((Vrate, durV, durU, durP, dur_ratio))

    return feat_vec





def V_UV(F0, data_audio, type_seg, size_stepS):
    if type_seg=="Voiced":
        pitch_seg = np.where(F0!=0)[0]
    elif type_seg=="Unvoiced":
        pitch_seg = np.where(F0==0)[0]
    dchange = np.diff(pitch_seg)
    change = np.where(dchange>1)[0]
    if len(pitch_seg)==0:
        return []
    init_seg= (pitch_seg[0]*size_stepS)+size_stepS
    segment=[]
    for indx in change:
        end_seg = (pitch_seg[indx]*size_stepS)+size_stepS#To compute energy
        seg = data_audio[int(init_seg):int(end_seg)]#To compute energy
        segment.append(seg)
        init_seg = (pitch_seg[indx+1]*size_stepS)+size_stepS#To compute energy

    return segment




def logEnergy(sig):
    if len(sig)==0:
        return -1e30
    sig2=np.power(sig,2)
    sumsig2=np.sum(np.abs(sig2))/len(sig2)

    logE=10*np.log10(sumsig2)
    if np.isnan(logE) or np.isinf(logE):
        logE=-1e30

    return logE



