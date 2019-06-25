import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import sys
from scipy.signal import gaussian
from scipy.stats import kurtosis, skew

def chunks(l, k):
      for i in range(0, len(l), int(k/2)):
            yield l[i:i+k]
# inp = sys.argv[1]
# wavName =inp  #'/home/swar/swar/shreeharsha/BNB/'+
# [fs, signal] = wav.read(wavName)  # read wav file
# signal = signal/(0.0+np.max(np.abs(signal)))
def mic_remov(signal,fs):
    win_size = int(fs*0.0005)
    signal2 = np.copy(signal)

    # for i in range(1,len(signal2)):
    #   signal[i] = signal2[i]-0.95*signal2[i-1]
    #signal = signal/np.max(np.abs(signal))
    #signal2 = signal2/np.max(np.abs(signal2))
    chunkeddata = list(chunks(signal, win_size))
    chunkeddata2 = chunkeddata[0:-2]
    del chunkeddata
    # chunkeddata = np.copy(chunkeddata2)
    # for i in range(int(np.floor(len(chunkeddata2)/win_size))):
    #   chunkeddata[i] =  chunkeddata2[i]*gaussian(win_size,win_size/3)
    ps = np.square(np.abs(np.fft.fft(chunkeddata2,128)))
    # for i in range(len(ps)):
    #   if (np.max(ps[i] != 0)):
    #       ps[i] = ps[i]/(np.max(ps[i]))
    kmeas = np.std(ps,axis = 1)
    # kmeas = kmeas/np.max(kmeas)
    kmeas2 = np.repeat(kmeas,len(signal)/(0.0+len(kmeas)))
    tim = np.linspace(0,int(np.ceil(len(signal)/16000)),len(signal))

    r1 = len(signal)/len(ps)

    skewmeas = np.hstack((kmeas2,np.zeros(len(signal)-len(kmeas2))))
    #skewmeas = 10000*np.log10(skewmeas+10**-5)
    m3 = np.hstack((np.diff(np.hstack((np.diff(np.hstack((np.diff(skewmeas),0))),0))),0)) # consec difference
    m3 = np.abs(m3)
    m3m = np.max(m3)
    mask = np.ones(len(m3))

    r2 = np.mean(m3)*100/(np.max(m3)-np.min(m3))
    print(r2)
    flag = 0

    for i in range(len(ps)):
        if (np.max(ps[i] != 0)):
            ps[i] = ps[i]/(np.max(ps[i]))
    kmeas = np.std(ps,axis = 1)
    del ps
    del chunkeddata2
    # kmeas = kmeas/np.max(kmeas)
    kmeas2 = np.repeat(kmeas,len(signal)/(0.0+len(kmeas)))

    skewmeas = np.hstack((kmeas2,np.zeros(len(signal)-len(kmeas2))))

    if r2<0.15:
        th = 0.1
        flag =1
    elif r2>=0.15 and r2<0.25:
        th = 0.18
        flag =1
    else:
        th = 1.0

    for j in range(len(m3)):
        if m3[j]>th*m3m:
            if sum(m3[j-50:j+50])<10*np.max(m3[j-50:j+50]): ## This ensures that only infrequent mic bursts are detected.
                mask[j-150:j+150] = 0 ## If there are too many frequent mic bursts then speech (which may be present) also gets 
                                      ## cut so unless the local sum is < 10% of local max i.e. there is an infrequent burst in m3
                                      ## then only the mask is made 0. 
                                      
    #indices = np.where(m1>0.75*np.max(m1))
    
    data = np.copy(mask)
    datasig=np.append(data,1-data[-1])
        #datasig = np.logical_not(datasig).astype(int)
    if flag ==1:
        sil=0
        sp=0
        sildurthresh=np.ceil(10)
        initialsil=0
        initialsp=0
        spdurthresh=np.ceil(100)
        for i in range(len(datasig)-1):
            if datasig[i]==0:
                sil=sil+1
            elif datasig[i]==1:
                sp=sp+1
            if datasig[i]==1 and datasig[i+1]==0:
                if sp<=spdurthresh:
                    datasig[initialsp:i+1]=0    # post processing the mask
                    sil=sp+sil
                    sp=0
                else:
                    sil=0
                    finalsp=i
                    initialsil=i+1
            elif datasig[i]==0 and datasig[i+1]==1:
                if sil<=sildurthresh:
                    # datasig[initialsil:i+1]=1   # post processing the mask
                    sp=sp+sil
                    sil=0
                else:
                    sp=0
                    finalsil=i
                    initialsp=i+1

    data_new=np.delete(datasig,-1)
    #flag=1
    return data_new
#wav.write(wavName[:-4]+'_mic.wav',fs,signal2*data_new)
#data_new = np.logical_not(data_new).astype(int)

# f,ax = plt.subplots(4,sharex=True)
# ax[0].plot(tim,m3,label = '3rd consec diff of std coeffs')
# #ax[1].plot(tim,signal2*mask,label= 'mic-free signal',color = 'r')
# ax[2].plot(tim,signal,label= 'signal',color='k')
# ax[1].plot(tim,data_new,label= 'Mask',color='g')
# #ax[3].plot(tim,data_new,label= 'm2',color='g')
# if(flag == 1):
#   ax[3].plot(tim,signal2*data_new,label= 'mic-free signal',color = 'r')

# ax[0].legend()
# ax[1].legend()
# ax[2].legend()
# ax[3].legend()
# plt.show()