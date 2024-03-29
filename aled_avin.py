import numpy as np
from numpy import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import scipy.io.wavfile as wav
import sys
from scipy.fftpack import fft, ifft
from scipy.stats.mstats import gmean
from mic_test import mic_remov
# import obspy
# from obspy.signal.filter import envelope

def findEnergy(sig,winSize):
# frame wise energy plot
    # 50% overlap
    window = np.hamming(winSize)
    # noWindows = math.ceil((len(signal)/(winSize/2)) - 1)
    # noWindows = int(noWindows)
    noWindows = int((len(signal)/np.ceil(float(winSize)/2)) - 1)
    # noWindows.astype(int)
    stEnergy = np.zeros(noWindows)
    spFlat = np.zeros(noWindows)
    for i in range(0,noWindows):
        # print i
        windowedSig = signal[i*winSize/2 : i*winSize/2 + winSize]
        dft = fft(windowedSig)
        hammwindowedSig = np.multiply(windowedSig,window)
        stEnergy[i] = np.sqrt(np.sum(np.multiply(hammwindowedSig,hammwindowedSig)))/winSize
        power = np.abs(dft)**2 # power spectrum
        din = np.mean(power)     
        # print din
        num = np.exp(np.mean(np.log(power)))
        # print num
        if (din!=0):
            spFlat[i] = float(num)/float(din)

    return stEnergy,spFlat

def setThresh(stEnergy):
    # temp1 = len(stEnergy)
    start = np.mean(stEnergy[0:np.ceil(fs*4*2/winSize)]) # 2 seconds
    # stEnergy1 = np.insert(stEnergy[0:(temp1-1)],0,1)
    temp = np.where(stEnergy>(2*start))
    # temp = where((np.add(stEnergy,np.ones(temp1))/np.add(stEnergy1,np.ones(temp1)))<1)
    # thresh = mean(stEnergy[0:temp[0][0]])
    return temp

def variance(x,size,m):
    if(size>=m):
        var_x = np.var(x)
    else:
        index = m-size
        trunc_x = x[index:m]
        var_x = np.var(trunc_x)
    return var_x

def getp(ratio):
    if (ratio>=1.25):
        p = 0.25
    elif (ratio>=1.1):
        p = 0.2
    elif (ratio>=1):
        p = 0.15
    else: # ratio < 1
        p = 0.1
    # p = p+0.5

    return p


if __name__ == '__main__':

    childName = sys.argv[1]
    # childName = test4
    #childName = "train1"
    wavName = childName
    datName = "/home/swar/swar/shreeharsha/VAD/Ankita_VAD_results/aled/" + childName.split('s/')[1] + "_aled.txt"
    confName = "/home/swar/swar/shreeharsha/VAD/Ankita_VAD_results/aled/" + childName.split('s/')[1] + "_conf.txt"
    stEnergyName = "/home/swar/swar/shreeharsha/VAD/Ankita_VAD_results/aled/" + childName.split('s/')[1] + "_stEnergy.txt"
    fid = open("/home/swar/swar/shreeharsha/VAD/Ankita_VAD_results/aled/"+childName.split('s/')[1]+"_EnergyThresholded.csv","w") ## path names for confidence, energy...
    [fs, signal] = wav.read(wavName)  # read wav file
    signal = signal/(0.0+np.max(np.abs(signal)))
    #plt.figure()
    #plt.plot(signal)
    signal = mic_remov(signal,fs)*signal
    #plt.figure()
    #plt.plot(signal)
    #plt.show()
    winSize = fs/50#320 for 16Khz
    m = (fs/16000)*50  #50 for 16Khz# buffer of most recent 'm' silence frames -> 0.8 seconds

    buffer = np.zeros(m).tolist()
    hardThresh = 0 # it has to be silence if it is below this, this will ensure that the algo continues to go in the loop

    for sec in range(0,2):
        # wavName = "data/" + childName + ".wav"
        
        [stEnergy,spFlat] = findEnergy(signal,winSize)
        np.savetxt(stEnergyName,stEnergy)
        noFrames = len(stEnergy)
        out = np.ones(noFrames) # tagged as silence by default
        temp = setThresh(stEnergy)
        # print temp
        if sec == 0:
            thresh = np.mean(stEnergy[0:np.ceil(fs*2*0.1*2/winSize)]) # Initial first iteration
        else:
            thresh = float(np.loadtxt('/home/swar/swar/shreeharsha/VAD/Ankita_VAD_results/aled/'+childName.split('s/')[1]+'_asilen.txt'))
        # thresh = np.abs(stEnergy[0])
        # print stEnergy[0]
        threshold = np.ones(noFrames)
        var_plot = np.zeros(noFrames)
        pPlot = np.zeros(noFrames)
        ratioPlot = np.zeros(noFrames)
        confidence = np.zeros(noFrames)
        #confidence = confidence*0.0
        k = 1.4 # thresholding factor
        no_sil = 1 # count of number of sil frames
        var = 0
        speechFrame = 0
        p = 0
        ratio = 0

        # if(thresh<0.001):
        #     thresh = 0.001

        # print stEnergy[0:10]
        for i in range (0,noFrames):
            threshold[i] = thresh
            prev_var = var
            fid.write(str(stEnergy[i]-k*thresh)+"\n")
            if (stEnergy[i]>k*thresh):
                confidence[i] = stEnergy[i] - (k*thresh)
                out[i] = 0  # speech
                speechFrame=speechFrame+1
                if(speechFrame==1): 
                    tap = i-1 # record the index of 1st speech frame in speech segment
                if(speechFrame>250): # too long speech segment
                    i = tap # reinitializing the index
                    thresh = hardThresh

        # updating threshold
            if (out[i]==1): # add E-silence to buffer and update threshold
                speechFrame=0
                buffer.append(stEnergy[i]) # adds element at the end
                buffer.pop(0) # removes element from the beginning
                # print len(buffer)
                if(no_sil >= m):
                    var = variance(buffer,no_sil,m)
                no_sil = no_sil+1
                if(prev_var != 0):
                    ratio = var/prev_var
                    p = getp(ratio)
                    thresh = (1-p)*thresh + p*stEnergy[i]
                if(thresh>hardThresh):
                    hardThresh = thresh
            pPlot[i] = p
            ratioPlot[i] = ratio
            var_plot[i] = var
        #confidence = confidence/np.max(confidence)
        # indices = np.where(confidence>0.01)[0]
        # confidence = np.zeros(noFrames)
        # confidence[indices] = np.ones(len(indices))
        print(sec)
        out = np.abs(out-1) # 1: speech, 0: silence
        np.savetxt(datName, out, fmt='%.0f')
        if sec == 0:
            ene = np.array([line.rstrip('\n') for line in open('/home/swar/swar/shreeharsha/VAD/Ankita_VAD_results/aled/'+childName.split('s/')[1]+'_stEnergy.txt')]).astype(float)
            vad = np.array([line.rstrip('\n') for line in open('/home/swar/swar/shreeharsha/VAD/Ankita_VAD_results/aled/'+childName.split('s/')[1]+'_aled.txt')]).astype(float)
            asen = sum(ene*vad)/len(ene)
            with open('/home/swar/swar/shreeharsha/VAD/Ankita_VAD_results/aled/'+childName.split('s/')[1]+'_asilen.txt', 'w') as f:
                f.write('%f'%asen)
        #np.savetxt(confName, confidence, fmt='%.0f')
    
    fid.close()