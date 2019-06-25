import numpy as np
from numpy import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import scipy.io.wavfile as wav
import sys


if __name__ == '__main__':
    
    datName=sys.argv[1]
    saveName = datName+'_pp.txt'

    #childName='TheTalkativeTortoise.wav_vad_13.wav'
    
#    flag = sys.argv[2]
    
    #wavName = dirName+childName
#    if (flag == '0'):
#        datName = "data/aled/" + childName + "_aled.txt"  # aled
#        saveName = "data/aled/" + childName + "_aledmod.txt" # aled
#    elif (flag == '1'):
#        datName = "data/zffs/" + childName + "_zffs.txt"  # zffs
#        saveName = "data/zffs/" + childName + "_zffsmod.txt" ## zffs
#    elif (flag == '2'):
#        datName = "data/twopass/"+ childName +".txt"  # 2pass
#        saveName = "data/twopass/"+ childName +"_mod.txt" ## 2pass
#    elif (flag == '3'):
#        datName = "data/dvad/test"+ childName +".txt"  # 2pass
#        saveName = "data/dvad/test"+ childName +"_dvad.txt" ## 2pass
#    else:
#        print("Please enter a valid option")
    #datName=dirName+"aled/"+childName + '_aled.txt'
    #saveName=dirName+"aled/"+childName + '_aledmod.txt'
    # print "Python script is running ..."
    #[fs, signal] = wav.read(wavName)
    #winSize = 160
    #signal = signal/30000.0
    data = loadtxt(datName)
    datasig=np.append(data,1-data[-1])
    #print datasig
    #datasig = np.logical_not(datasig).astype(int)
    sil=0
    sp=0
    framelen= float(sys.argv[2]) ## in ms
    sildurthresh=np.ceil(200/framelen)
    initialsil=0
    initialsp=0
    spdurthresh=np.ceil(100/framelen)######## Left to right
    datasig2 = np.copy(np.flipud(datasig)) ####### Right to left
    for i in range(len(datasig)-1):
        if datasig[i]==0:
            sil=sil+1
        elif datasig[i]==1:
            sp=sp+1
        if datasig[i]==1 and datasig[i+1]==0:
            if sp<=spdurthresh:
                datasig[initialsp:i+1]=0    # small speech to silence
                sil=sp+sil
                sp=0
            else:
                sil=0
                finalsp=i
                initialsil=i+1
        elif datasig[i]==0 and datasig[i+1]==1:
            if sil<=sildurthresh:
                datasig[initialsil:i+1]=1   # small silence to speech
                sp=sp+sil
                sil=0
            else:
                sp=0
                finalsil=i
                initialsp=i+1
    #print datasig
    #
    for i in range(len(datasig2)-1):
        if datasig2[i]==0:
            sil=sil+1
        elif datasig2[i]==1:
            sp=sp+1
        if datasig2[i]==1 and datasig2[i+1]==0:
            if sp<=spdurthresh:
                datasig2[initialsp:i+1]=0    # small speech to silence
                sil=sp+sil
                sp=0
            else:
                sil=0
                finalsp=i
                initialsil=i+1
        elif datasig2[i]==0 and datasig2[i+1]==1:
            if sil<=sildurthresh:
                datasig2[initialsil:i+1]=1   # small silence to speech
                sp=sp+sil
                sil=0
            else:
                sp=0
                finalsil=i
                initialsp=i+1
    datasig = np.logical_or(datasig,np.flipud(datasig2))
    data_new=np.delete(datasig,-1)
    np.savetxt(saveName,data_new,fmt='%.0f')
