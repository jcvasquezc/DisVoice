
# -*- coding: utf-8 -*-
"""
Created on Jul 21 2017

@author: J. C. Vasquez-Correa
"""


import numpy as np



def jitter_env(vPPS, iNumPuntos):

    iLongSec=len(vPPS)

    if (iLongSec < 2):
        print( 'Pitch sequence is too short' );
        vJitta=np.zeros(iNumPuntos)
        return vJitta

    vJitta=np.zeros(iNumPuntos)
    iIndiceIni=0

    iDesplazamiento=iLongSec/iNumPuntos

# average f0 of signal
    rFoMed=np.max(vPPS)

    for n in range(iNumPuntos-1):
        indice=int( iIndiceIni+n*iDesplazamiento )
        if ( n>0 ) and (indice==int( iIndiceIni+(n-1)*iDesplazamiento )):
            vJitta[ n ]=vJitta[ n-1 ]
        else:
            if (indice+1 < iLongSec):
                    vJitta[n]=np.abs(vPPS[indice+1]-vPPS[indice])
            else:
                vJitta[n]=0;
        vJitta[n]=100*vJitta[n]/rFoMed

    return vJitta

def logEnergy(sig):
    sig2=np.power(sig,2)
    sumsig2=np.sum(np.absolute(sig2))/len(sig2)
    logE=np.log10(sumsig2)
    return logE


def Power(sig):
    sig2=np.square(sig)
    sumsig2=np.sum(sig2)/len(sig)
    return sumsig2


def shimmer_env(vPPS, iNumPuntos):

    iLongSec=len(vPPS)

    if (iLongSec < 2):
        print( 'Pitch sequence is too short' );
        vShimm=np.zeros(iNumPuntos)
        return vShimm

    vShimm=np.zeros(iNumPuntos)
    iIndiceIni=0

    iDesplazamiento=iLongSec/iNumPuntos

# average f0 of signal
    rFoMed=np.max(vPPS)

    for n in range(iNumPuntos-1):
        indice=int( iIndiceIni+n*iDesplazamiento )
        if ( n>0 ) and (indice==int( iIndiceIni+(n-1)*iDesplazamiento )):
            vShimm[ n ]=vShimm[ n-1 ]
        else:
            if (indice+1 < iLongSec):
                    vShimm[n]=np.abs(vPPS[indice+1]-vPPS[indice])
            else:
                vShimm[n]=0;
        vShimm[n]=100*vShimm[n]/rFoMed

    return vShimm



def PQ(x,k):
    """
    Perturbation Quotient in percentage of the signal x
    input: x--> input sequence: F0 values or Amplitude values
    k--> average factor (must be an odd number)
    """
    N=len(x)
    if N<k or k%2==0:
        return 0
    m=int(0.5*(k-1))
    summ=0
    for n in range(N-k):
        dif=0
        for r in range(k):
            dif=dif+x[n+r]-x[n+m]
        dif=np.abs(dif/float(k))
        summ=summ+dif
    num=summ/(N-k)
    den=np.mean(np.abs(x))
    c=100*num/den
    if np.sum(np.isnan(c))>0:
        print(x)
    return c

def APQ(PAS):
    """
    Amplitude perturbation quotient (APQ)
    input:-->PAS: secuence of peak amplitudes of a signal
    """
    return PQ(PAS,11)

def PPQ(PPS):
    """
    Period perturbation quotient (APQ)
    input:-->PAS: secuence of pitch periods of a signal
    """
    return PQ(PPS,5)
