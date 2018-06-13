

import sys
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import numpy as np
import os

sys.path.append('../praat')
import praat_functions

import utils_gci
import pysptk

from scipy.io.wavfile import write
from scipy.integrate import cumtrapz
from peakdetect import peakdetect


def getF0(audio_file, size_step=0.02, minF0=60, maxF0=500):
    name_audio=audio_file.split('/')
    temp_uuid='phon'+name_audio[-1][0:-4]
    temp_filename_vuv='../tempfiles/tempVUV'+temp_uuid+'.txt'
    temp_filename_f0='../tempfiles/tempF0'+temp_uuid+'.txt'
    praat_functions.praat_vuv(audio, temp_filename_f0, temp_filename_vuv, time_stepF0=size_step, minf0=minF0, maxf0=maxF0)
    F0,_=praat_functions.decodeF0(temp_filename_f0,len(data_audio)/float(fs),size_step)
    os.remove(temp_filename_vuv)
    os.remove(temp_filename_f0)
    return F0


# GCI detection - SQ-VQ algorithm
def SE_VQ_varF0(x,fs, pitch_method="RAPT", f0=[]):

    # Function to extract GCIs using an adapted version of the SEDREAMS
    # algorithm which is optimised for non-modal voice qualities (SE-VQ). Ncand maximum
    # peaks are selected from the LP-residual signal in the interval defined by
    # the mean-based signal. A dynamic programming algorithm is then used to
    # select the optimal path of GCI locations. Then a post-processing method,
    # using the output of a resonator applied to the residual signal, is
    # carried out to remove false positives occurring in creaky speech regions
    # Note that this method is slightly different from the standard SE-VQ
    # algorithm as the mean based signal is calculated using a variable window
    # length. This is set using an f0 contour interpolated over unvoiced
    # regions and heavily smoothed. This is particularly useful for speech
    # involving large f0 excursions (i.e. very expressive speech). Note also
    # that this algorithm has not yet been fully formally tested.
    #
    # Usage:
    #
    # Input:
    #           x            - speech signal (in samples) if pitch_method=="RAPT", or audio file (*.wav) if pitch_method=="PRAAT"
    #           fs           - sampling frequency (Hz)
    #           pitch_method - RAPT (uses the RAPT algorithm to estimate F0)
    #                        - PRAAT (uses de praat based autocorrelation fucntion to estimate f0)
    #
    #
    # REFERENCE:
    #       Kane, J., Gobl, C., (2013) `Evaluation of glottal closure instant
    #       detection in a range of voice qualities', Speech Communication
    #       55(2), pp. 295-314.
    #
    #
    ###########################################################################
    ## ORIGINAL FUNCTION WAS CODED BY JOHN KANE AT THE PHONETICS AND SPEECH LAB IN #####
    ## TRINITY COLLEGE DUBLIN ON 2013.
    ## THE SEDREAMS FUNCTION WAS
    ## CODED BY THOMAS DRUGMAN OF THE UNIVERSITY OF MONS ######################
    ##
    ## THE CODE WAS TRANSLATED TO PYTHON AND ADAPTED BY J. C. Vasquez-Correa
    ## AT PATTERN RECOGNITION LAB UNIVERSITY OF ERLANGEN NUREMBER- GERMANY
    ## AND UNIVERSTY OF ANTIOQUIA, COLOMBIA
    ## JCAMILO.VASQUEZ@UDEA.EDU.CO
    ## jcvasquezc.wix.com/home
    ###########################################################################

    F0min=20
    F0max=500
    if len(f0)==0 or sum(f0)==0:
        if pitch_method=="RAPT":
            size_stepS=0.01*fs
            voice_bias=-0.2
            x=x-np.mean(x)
            x=x/np.max(np.abs(x))
            data_audiof=np.asarray(x*(2**15), dtype=np.float32)
            f0=pysptk.sptk.rapt(data_audiof, fs, int(size_stepS), min=F0min, max=F0max, voice_bias=voice_bias, otype='f0')
        elif pitch_method=="PRAAT":
            f0=getF0(x, minF0=F0min, maxF0=F0max)
            fs, x=read(x)
            x=x-np.mean(x)
            x=x/np.max(np.abs(x))


    F0nz=np.where(f0>0)[0]
    F0mean=np.median(f0[F0nz])
    VUV=np.zeros(len(f0))
    VUV[F0nz]=1
    if F0mean<70:
        print('Utterance likely to contain creak')
        F0mean=80



    # Interpolate f0 over unvoiced regions and heavily smooth the contour

    ptos=np.linspace(0,len(x),len(VUV))
    VUV_inter=np.interp(np.arange(len(x)), ptos, VUV)

    VUV_inter[np.where(VUV_inter>0.5)[0]]=1
    VUV_inter[np.where(VUV_inter<=0.5)[0]]=0

    f0_int, f0_samp=utils_gci.create_continuous_smooth_f0(f0,VUV,x)

    T0mean = fs/f0_samp
    winLen = 25; # window length in ms
    winShift = 5; # window shift in ms
    LPC_ord = int((fs/1000)+2) # LPC order
    Ncand=5; # Number of candidate GCI residual peaks to be considered in the dynamic programming
    trans_wgt=1 # Transition cost weight
    relAmp_wgt=0.3 # Local cost weight

    repNum=2 # Number of iterations for post-processing
    removeThresh=0.4 # Threshold for removing false GCIs
    search_reg=1.3/1000*fs # Search region for post processing


    #Calculate LP-residual and extract N maxima per mean-based signal determined intervals

    res = utils_gci.GetLPCresidual(x,winLen*fs/1000,winShift*fs/1000,LPC_ord, VUV_inter); # Get LP residual
    rep = utils_gci.RCVD_reson_GCI(res,fs,F0mean) #Get resonator output

    MBS = utils_gci.get_MBS(x,fs,T0mean) # Extract mean based signal


    interval = utils_gci.get_MBS_GCI_intervals(MBS,fs,T0mean,F0max) # Define search intervals

    [GCI_N,GCI_relAmp] = utils_gci.search_res_interval_peaks(res,interval,Ncand, VUV_inter) # Find residual peaks

    GCI = utils_gci.RESON_dyProg_mat(GCI_relAmp,GCI_N,F0mean,x,fs,trans_wgt,relAmp_wgt, plots=False) # Do dynamic programming
    # plt.plot(x)
    # plt.plot(res)
    # plt.plot(GCI, 0.2*np.ones(len(GCI)), 'ro')
    #
    # plt.show()
    return GCI
    #plt.plot(MBS)
    #plt.show()



def IAIF(x,fs,GCI):


    # Function to carry out iterative and adaptive inverse filtering (Alku et
    # al 1992).

    # USAGE:
    #       Input:
    #             x   : speech signal (in samples)
    #             fs  : sampling frequency (in Hz)
    #             GCI : Glottal closure instants (in samples)
    #
    #       Output:
    #             g_iaif : glottal flow derivative estimate
    #
    #
    #########################################################################
    ## Function Coded by John Kane @ The Phonetics and Speech Lab ###########
    ## Trinity College Dublin, August 2012 ##################################
    #########################################################################


    ## THE CODE WAS TRANSLATED TO PYTHON AND ADAPTED BY J. C. Vasquez-Correa
    ## AT PATTERN RECOGNITION LAB UNIVERSITY OF ERLANGEN NUREMBER- GERMANY
    ## AND UNIVERSTY OF ANTIOQUIA, COLOMBIA
    ## JCAMILO.VASQUEZ@UDEA.EDU.CO
    ## jcvasquezc.wix.com/home


    p=int(fs/1000)+2 # LPC order

    x_filt=x-np.mean(x)
    x_filt=x_filt/max(abs(x_filt))

    # ------------------------------------------------
    # emphasise high-frequencies of speech signal (LPC order 1) - PART 2 & 3
    ord_lpc1=1;
    x_emph=utils_gci.calc_residual(x_filt,x_filt,ord_lpc1,GCI);

    # ------------------------------------------------
    # first estimation of the glottal source derivative - PART 4 & 5
    ord_lpc2=p;
    residual1=utils_gci.calc_residual(x_filt,x_filt,ord_lpc2,GCI);

    # integration of the glottal source derivative to calculate the glottal
    # source pulse - PART 6 (cancelling lip radiation)
    ug1=cumtrapz(residual1)
    # ------------------------------------------------
    # elimination of the source effect from the speech spectrum - PART 7 & 8

    ord_lpc3=4;
    vt_signal=utils_gci.calc_residual(x_filt,ug1,ord_lpc3,GCI);

    # ------------------------------------------------
    # second estimation of the glottal source signal - PART 9 & 10
    ug2=cumtrapz(vt_signal)

    ord_lpc4=p;
    residual2=utils_gci.calc_residual(x_filt,vt_signal,ord_lpc4,GCI);
    return residual2



def get_vq_params(gf, gfd, fs, GCI):
    # Function to estimate the glottal parameters: NAQ, QOQ, H1-H2, and HRF

    #  This function can be used to estimate a range of conventional glottal
    #  source parameters often used in the literature. This includes: the
    #  normalized amplitude quotient (NAQ), the quasi-open quotient (QOQ), the
    #  difference in amplitude of the first two harmonics of the differentiated
    #  glottal source spectrum (H1-H2), and the harmonic richness factor (HRF)
    # Inputs
    #  gf       : [samples] [N] Glottal flow estimation
    #  gfd      : [samples] [N] Glottal flow derivative estimation
    #  fs       : [Hz]      [1] sampling frequency
    #  GCI      : [samples]       [M] Glottal closure instants
    #
    # Outputs
    #  NAQ      : [s,samples] [Mx2] Normalised amplitude quotient
    #  QOQ      : [s,samples] [Mx2] Quasi-open quotient
    #  H1H2     : [s,dB]      [Mx2] Difference in glottal harmonic amplitude
    #  HRF      : [s,samples] [Mx2] Harmonic richness factor
    #
    # References
    #  [1] Alku, P., B ackstrom, T., and Vilkman, E. Normalized amplitude quotient
    #     for parameterization of the glottal flow. Journal of the Acoustical
    #     Society of America, 112(2):701-710, 2002.
    #  [2] Hacki, T. Klassifizierung von glottisdysfunktionen mit hilfe der
    #     elektroglottographie. Folia Phoniatrica, pages 43-48, 1989.
    #  [3] Alku, P., Strik, H., and Vilkman, E. Parabolic spectral parameter -
    #     A new method for quantification of the glottal flow. Speech
    #     Communication, 22(1):67-79, 1997.
    #  [4] Hanson, H. M. Glottal characteristics of female speakers: Acoustic
    #     correlates. Journal of the Acoustical Society of America,
    #     10(1):466-481, 1997.
    #  [5] Childers, D. G. and Lee, C. K. Voice quality factors: Analysis,
    #     synthesis and perception. Journal of the Acoustical Society of
    #     America, 90(5):2394-2410, 1991.
    #
    # Author
    #########################################################################
    ## Function Coded by John Kane @ The Phonetics and Speech Lab ###########
    ## Trinity College Dublin, August 2012 ##################################
    #########################################################################

    ## THE CODE WAS TRANSLATED TO PYTHON AND ADAPTED BY J. C. Vasquez-Correa
    ## AT PATTERN RECOGNITION LAB UNIVERSITY OF ERLANGEN NUREMBERGER- GERMANY
    ## AND UNIVERSTY OF ANTIOQUIA, COLOMBIA
    ## JCAMILO.VASQUEZ@UDEA.EDU.CO
    ## jcvasquezc.wix.com/home

    F0min=20
    F0max=500
    NAQ=np.zeros(len(GCI))
    QOQ=np.zeros(len(GCI))
    H1H2=np.zeros(len(GCI))
    HRF=np.zeros(len(GCI))
    T1=np.zeros(len(GCI))
    T2=np.zeros(len(GCI))
    glot_shift=np.round(0.5/1000*fs)
    qoq_level=0.5 # threhols for QOQ estimation
    T0_num=3 #Number of local glottal pulses to be used for harmonic spectrum
    min_harm_num=5
    HRF_freq_max=5000 # Maximum frequency used for harmonic measurement

    for n in range(len(GCI)):

        # Get glottal pulse compensated for zero-line drift
        if n==0:
            start=0;
            stop=int(GCI[n])
            T0=GCI[n+1]-GCI[n]
        else:
            start=int(GCI[n-1])
            stop=int(GCI[n])
            T0=GCI[n]-GCI[n-1]
            if T0==0 and n>2:
                T0=GCI[n]-GCI[n-2]
                start=int(GCI[n-2])
        F0=fs/T0

        if T0>0 and F0>F0min and F0<F0max:
            gf_comb=[gf[start], gf[stop]]
            if start!=stop:
                if len(gf_comb)>1:
                    line=np.interp(np.arange(stop-start), np.linspace(0,stop-start,2), gf_comb)
                else:
                    line=gf_comb
            else:
                line=0
            gf_seg=gf[start:stop]
            gf_seg_comp=gf_seg-line
            if stop+glot_shift<=len(gfd):
                stop2=int(stop+glot_shift)
            else:
                stop2=stop
            gfd_seg=gfd[start:stop2]

            # get NAQ and QOQ
            d_peak=np.max(np.abs(gfd_seg))
            f_ac=np.max(gf_seg_comp)
            max_idx=np.argmax(gf_seg_comp)
            Amid=f_ac*qoq_level

            T1[n],T2[n] = findAmid_t(gf_seg_comp,Amid,max_idx)
            NAQ[n]=(f_ac/d_peak)/T0
            QOQ[n]=(T2[n]-T1[n])/(fs/F0)

            #Get frame positions for H1-H2 parameter
            if GCI[n]-int((T0*T0_num)/2)>0:
                f_start=int(GCI[n]-int((T0*T0_num)/2))
            else:
                f_start=0
            if GCI[n]+int((T0*T0_num)/2)<=len(gfd):
                f_stop=int(GCI[n]+int((T0*T0_num)/2))
            else:
                f_stop=len(gfd)
            f_frame=gfd[f_start:f_stop]
            f_win=f_frame*np.hamming(len(f_frame))
            f_spec=20*np.log10(np.abs(np.fft.fft(f_win, fs)))

            f_spec=f_spec[0:int(len(f_spec)/2)]
            # get H1-H2 and HRF
            [max_peaks, min_peaks]=peakdetect(f_spec,lookahead = int(T0))
            if len(max_peaks)==0:
                H1H2[n]=0
                HRF[n]=0
                continue
            h_idx, h_amp=zip(*max_peaks)


            HRF_harm_num=np.fix(HRF_freq_max/F0)

            if len(h_idx)>=min_harm_num:
                temp1=np.arange(HRF_harm_num)*F0
                f0_idx=np.zeros(len(h_idx))
                for mp in range(len(h_idx)):

                    temp2=h_idx[mp]-temp1
                    temp2=np.abs(temp2)
                    posmin=np.where(temp2==min(temp2))[0]
                    if len(posmin)>1:
                        posmin=posmin[0]

                    if posmin<len(h_idx):
                        f0_idx[mp]=posmin
                    else:
                        f0_idx[mp]=len(h_idx)-1

                f0_idx=[int(mm) for mm in f0_idx]

                H1H2[n]=h_amp[f0_idx[0]]-h_amp[f0_idx[1]]
                harms=[h_amp[mm] for mm in f0_idx[1:]]
                HRF[n]=sum(harms)/h_amp[f0_idx[0]]

    return NAQ, QOQ, T1, T2, H1H2, HRF

def findAmid_t(glot_adj, Amid, Tz):
    #Function to find the start and stop positions of the quasi-open phase.
    T1=0
    T2=0
    if Tz!=0:
        n=Tz
        while glot_adj[n]>Amid and n>2:
            n=n-1
        T1=n
        n=Tz
        while glot_adj[n] > Amid and n < len(glot_adj)-1:
            n=n+1;
        T2=n
    return T1, T2

if __name__=="__main__":
    if len(sys.argv)!=2:
        print("python GCI.py <audio_file.wav>")
        sys.exit()
    else:
        audio=sys.argv[1]

    fs, data_audio=read(audio)
    GCI=SE_VQ_varF0(data_audio,fs, pitch_method="RAPT")
    print('Glottal inverse filtering using IAIF algorithm (Alku et al. 1992)')
    g_iaif=IAIF(data_audio,fs,GCI)
    g_iaif=g_iaif-np.mean(g_iaif)
    g_iaif=g_iaif/max(abs(g_iaif))

    print(data_audio.dtype)
    data_audio=data_audio/max(abs(data_audio))
    plt.figure()
    plt.plot(data_audio)
    plt.plot(g_iaif)

    plt.show()

    glottal=cumtrapz(g_iaif)
    glottal=glottal-np.mean(glottal)
    glottal=glottal/max(abs(glottal))

    plt.figure()
    plt.plot(glottal)
    plt.show()
