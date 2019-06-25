## Author: Ankita Pasad
# Created on: 22.09.2016
# Implementation of zero frequency filtering of speech signal for epoch detection
# Reference: http://speech.iiit.ac.in/svlpubs/article/MurtyK.S.R.Yegna2008.pdf

import numpy as np
import scipy.signal as sigpy
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import scipy.io.wavfile as wav
import sys
import pandas.tools.plotting
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
from mic_test import mic_remov


def findZffs(signal,f0):

    numerator = np.array([1,-1])
    denominator = np.array([1,-3,3,-1])
    signal = signal/np.float(np.max(signal))
# filtering the speech signal
    zffsSigRaw = sigpy.lfilter(numerator,denominator,signal)
    # zffsSigRaw = zffsSigRaw/np.float(np.max(zffsSigRaw)) # normalizing
# removing the trend
    # 2N+1 = number of samples in one pitch period                  
    N = int(math.ceil(((fs/f0)-1)/2)) 
    noSamples = int((2*N)+1)

    # appending zeros for mean computation
    before = np.zeros(N)
    after = np.zeros(N)
    zffsSig_appended = np.concatenate([before,zffsSigRaw])
    zffsSig_appended = np.concatenate([zffsSig_appended,after])
    size = len(zffsSig_appended)

    # array to be subtracted from the original array
    # moving mean, considering the window of size (2N+1) around each sample
    ma_zffsSig = pd.rolling_mean(zffsSig_appended,noSamples,center=True) 
    # extracting out the relevant part
    ma_zffsSig = ma_zffsSig[N:size-N] 
    # subtracting the mean
    zffsSig = zffsSigRaw - ma_zffsSig
    length = len(zffsSig)
    zffsSig = zffsSig[0:length-N]
    # taking signal to be of same length
    signal = signal[0:length-N]
    # zffsSig = zffsSig/np.float(np.max(zffsSig))

    return zffsSig,signal,zffsSigRaw[0:length-N]

def findExcitation(zffs):

    # Excitation: defined as slope at positive zero crossings for zff signal
    # finding positive zero crossings
    indices = np.where(np.diff(np.sign(zffs))>0) 
    # converting to 1-dim array
    indices = indices[0]
    excitation = np.zeros(len(zffs)) 
    for i in range (0,len(indices)):
        # approximating the slope at positive zero crossings
        excitation[indices[i]]=(zffs[indices[i]+1]-zffs[indices[i]]) 

    return excitation

def findCorrCoef(corr):
    peakLoc = np.where(np.diff(np.sign(np.diff(corr)))<0)[0] # positions of peak locations
    if(len(peakLoc)==0):
        corrCoef=0
    else:
        peakLoc = peakLoc + np.ones(len(peakLoc))
        peakAmplitude = np.zeros(len(peakLoc))
        for i in range(0,len(peakLoc)):
            peakAmplitude[i] = corr[peakLoc[i]]
        peakAmplitudeSort = np.sort(peakAmplitude)
        # peakAmplitudeSortArg =np.argsort(peakAmplitude)
        corrCoef = peakAmplitudeSort[-1]

    return corrCoef

def findFeatures(zffsSig,winSize,fs): # When the whole signal is provided
    winLen = winSize*fs # winSize in seconds
    # noWindows = math.floor((len(zffsSig)/(winLen/2)) - 1)
    noWindows = int((len(zffsSig)/np.ceil(float(winLen)/2)) - 1)
    # noWindows = int(noWindows)
    excitation  = findExcitation(zffsSig)
    zffsSig = zffsSig/30000.0
    corrCoef = np.zeros(noWindows)
    for i in range (0,noWindows):
#        print i    
        windowedSig = zffsSig[i*winLen/2: int(i*winLen/2 + winLen)]
#        print len(windowedSig)
        corr = np.correlate(windowedSig,windowedSig,mode='same')
#        print len(corr)
        corr = corr/np.float(np.max(corr))
        if(len(np.where(corr==1)[0])!=0):
            index = np.where(corr==1)[0][0]
            index = index+1
            corrCoef[i] = findCorrCoef(corr[0:index]) 
    
    return zffsSig, excitation, corrCoef

def lpfilter(sig,order):
    # order: N
    # winSize = 2N + 1
    winSize = 2*order+1
    before = np.zeros(order)
    after = np.zeros(order)
    sig_appended = np.concatenate([before,sig])
    sig_appended = np.concatenate([sig_appended,after])
    size = len(sig_appended)
    ma_sig = pd.rolling_mean(sig_appended,winSize,center=True) 
    # extracting out the relevant part
    ma_sig = ma_sig[order:size-order] 

    return ma_sig

if __name__ == '__main__':

    noise1 = sys.argv[1]
    sigName2 = noise1
#    noise1 = 'test3'
    # txtName = "data/zffs/"+noise1+"_zffs.txt"
    txtName = "/home/swar/swar/shreeharsha/VAD/Ankita_VAD_results/zffs/" + noise1.split('s/')[1] + "_zffs.txt"
    # sigName2 = "data/"+noise1+".wav"    
    zffcoeffName="/home/swar/swar/shreeharsha/VAD/Ankita_VAD_results/zffs/"+noise1.split('s/')[1]+"_zffcoeff.txt"
    [fs, signal1] = wav.read(sigName2)  # read wav file
    signal1 = signal1/(0.0+np.max(np.abs(signal1)))
    signal1 = mic_remov(signal1,fs)*signal1
    #signal1 = signal1/30000.0 # normalizing in order to deal with over-flow issue
    [zffsSig1,signal1,zffsSigRaw1] = findZffs(signal1,300.0)
    # index = np.where(np.max(np.abs(signal1))/np.abs(signal1)>=10**1.5)
    # signal1[index] = 0
    winsize=0.02
    [zffsSig1,excitation1,corrCoef1] = findFeatures(zffsSig1,winsize,fs)
#    corrCoeff = corrCoef1[0::2]
#    corrCoeff = np.repeat(corrCoeff,nrep)
    np.savetxt(zffcoeffName,corrCoef1,fmt='%5.4f')
#    
    thresh = np.max(corrCoef1)*0.75
### Dynamic threshold
#    # Emax = np.float(np.max(corrCoef1))
#    # Emin = np.float(np.min(corrCoef1))
#    # I1 = 0.03*(Emax - Emin) + Emin
#    # I2 = 4*Emin
#    # Thl = np.min([I1,I2])
#    # Thu = 2*Thl
#    # Thl = 2*I1
#    # thresh = Thl
#    # print thresh
#
    indices = np.where(corrCoef1<=thresh)[0]
    # corrCoef1[indices] = np.zeros(len(indices))
    decision = np.ones(len(corrCoef1))
    decision[indices] = np.zeros(len(indices))
    zffsSig1 = np.abs(zffsSig1)/np.max(zffsSig1)
#
    # converting 50 ms window to 10 ms window
    decision = decision[0::2] # 25 to 50
    nrep = 5
    decision = np.repeat(decision,nrep) # 50 to 10
    np.savetxt(txtName,decision,fmt='%.0f')
#    
#
#    
#      # for plotting
#    nrep = np.int(fs*0.025)
#    index1 = len(corrCoef1)*nrep
#    xaxis = np.linspace(0,index1-1,index1)/16000.0
##    decision = np.repeat(decision,nrep)
#        
## Plotting the speech signal and zero frequency filtered signal
#    # length = len(signal) # xaxis in seconds
#    # xaxis = np.arange(0,np.float(length/float(fs)),(1/float(fs)))
#    
#    plt.figure()
#    plt.subplot(2,1,1)
#    plt.plot(xaxis, signal1[0:index1],xaxis,(np.repeat(corrCoef1,nrep)))
#    plt.title('Audio signal '+noise1)
##    plt.subplot(2,1,2)
##    plt.plot(xaxis,np.repeat(corrCoef1,nrep))
##    plt.title('Normalized Correlation Coefficient',fontsize=11)
#    # # plt.subplot(3,1,3)
#    # # plt.plot(decision)
#    # # plt.ylim([-0.5,1.5])
#    # # # plt.title('Correlation Coefficient',fontsize=11)
#    # plt.xlabel('Number of samples')
##############################plots for dual y-axis######################################
##    A = signal1[0:index1]
##    B = (np.repeat(corrCoef1,nrep))
##
##    host = host_subplot(111, axes_class=AA.Axes)
##    # plt.subplots_adjust(right=0.75)
##
##    par1 = host.twinx()
##    # par2 = host.twinx()
##
##    # offset = 60
##    # new_fixed_axis = par2.get_grid_helper().new_fixed_axis
##    # par2.axis["right"] = new_fixed_axis(loc="right",axes=par2,offset=(offset, 0))
##
##    # par2.axis["right"].toggle(all=True)
##
##    # host.set_xlim(0, 2)
##    host.set_ylim(-1.5,3)
##
##    host.set_xlabel("Duration (seconds)",fontsize=14)
##    host.set_ylabel("Normalized Audio Signal",fontsize=14)
##    par1.set_ylabel("Normalized Correlation Coefficient",fontsize=14)
##    # par2.set_ylabel("Velocity")
##
##    p1, = host.plot(xaxis, A, label="Audio Signal")
##    p2, = par1.plot(xaxis, B, label="Correlation Coefficient")
##    # p3, = par2.plot([0, 1, 2], [50, 30, 15], label="Velocity")
##
##    par1.set_ylim(-2.2,1.8)
##    # par2.set_ylim(1, 65)
##
##    host.legend()
##
##    host.axis["left"].label.set_color(p1.get_color())
##    par1.axis["right"].label.set_color(p2.get_color())
##    # par2.axis["right"].label.set_color(p3.get_color())
##
##    plt.title("Normalized Correlation Coefficient of ZFF signal")
##    plt.draw()
##    plt.show()
##############################plots for dual y-axis######################################
#    # C = np.zeros([index1,2])
#    # C[:,0] = A
#    # C[:,1] = B
#    # df = pd.DataFrame(C, index=xaxis, columns=list('AB'))
#    # df.A.plot(label="Audio Signal", legend=True, ylim=(-1.5,3))
#    # df.B.plot(secondary_y=True, label="Correlation Coefficient", legend=True)
#
#    # plt.figure()
#    # plt.plot(xaxis, (signal1[0:index1]-1), label = "Audio Signal")
#    # plt.plot(xaxis,(np.repeat(corrCoef1,nrep)+1),label = "Correlation Coefficient")
#    # plt.legend(loc="best")
#    # plt.title("Correlation Coefficient of ZFF signal")
#    # plt.xlabel('Sample Number')
#    # plt.ylabel('')
#
#    # plt.show()