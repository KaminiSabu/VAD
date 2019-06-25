import numpy as np
import csv
from numpy import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import scipy.io.wavfile as wav
import sys
import zffs

fileName = sys.argv[1]
flag = sys.argv[2]
back_noise = sys.argv[3] #Background noise flag
if back_noise == '0': ## normal background
	bn = 0.01
elif back_noise == '1': ## Loud background talkers & loud speaker voice
	bn = 0.25
elif back_noise == '2': ## Speaker's voice too soft
	bn = 0
fileName = fileName.split('s/')[1]
startDir = "/home/swar/swar/shreeharsha/VAD/"
if(flag=='1'):
	wavName = "Ankita_VAD_new/audios/" + fileName
	txtName = "data/twopass/"+fileName+".txt"
	aled = open("data/aled/"+fileName+"_aledmod.txt","rw+")
	zffs = open("data/zffs/"+fileName+"_zffsmod.txt","rw+")
elif(flag=='2'):
	wavName = startDir + "Ankita_VAD_new/audios/" + fileName
	txtName = startDir + "Ankita_VAD_results/twopass/"+fileName+".txt"
	aled = open(startDir+"Ankita_VAD_results/aled/"+fileName+"_aled.txt","rw+")
	zffs = open(startDir+"Ankita_VAD_results/zffs/"+fileName+"_zffho.txt","rw+")
with open('/home/swar/swar/shreeharsha/VAD/Ankita_VAD_results/aled/'+fileName+'_EnergyThresholded.csv') as csvfile:
    confidence2 = list(csv.reader(csvfile))
    #reader = csv.reader(x.replace('\0', '') for x in confidence2)
confidence = np.zeros(len(confidence2))
j = 0
for i in confidence2:
	confidence[j] = float(i[0])
	j+=1
aledLabel =array(map(int,aled.readlines()))
zffsLabel = array(map(int,zffs.readlines()))
#confidence = array(map(float,conf.readlines()))

aled.close()
zffs.close()
confidence = confidence/np.max(np.abs(confidence))
# equal length
length = min(len(aledLabel),len(zffsLabel))
aledLabel = aledLabel[:length]
zffsLabel = zffsLabel[:length]
# confidence = confidence[0:length]
out = np.zeros(length)
l2 = 15
la = l2-2
lz = (2/3*l2)-1
lc = l2*bn
# if(back_noise==2):
# 	la = l2/5
for j in range(l2,length-l2):
	if (sum(aledLabel[j-l2:j+l2]) < la and sum(zffsLabel[j-l2:j+l2]) > lz):
		out[j] = np.logical_and(aledLabel[j],zffsLabel[j])
	if (sum(aledLabel[j-l2:j+l2]) > la and sum(confidence[j-l2:j+l2]) > lc):
		out[j] = np.logical_or(aledLabel[j],zffsLabel[j])
out[-l2:] = out[-2*l2:-l2]

# [fs, signal1] = wav.read(wavName)
# signal1 = signal1/30000.0

# aledr = np.repeat(aledLabel,np.ceil(float(len(signal1))/float(len(aledLabel))))
# outr = np.repeat(out,np.ceil(float(len(signal1))/float(len(out))))
# zffsr = np.repeat(zffsLabel,np.ceil(float(len(signal1))/float(len(zffsLabel))))
# confidencer = np.repeat(confidence,np.ceil(float(len(signal1))/float(len(zffsLabel))))

# len1 = max(len(signal1),len(aledr),len(outr),len(zffsr),len(confidence))
# signal1 = np.pad(signal1, (0,len1-len(signal1)), 'constant')
# tim = np.linspace(0,len1/16000.0,len1)
# plt.plot(tim,outr,color = 'k',label = 'modified 2pass before PP')
# plt.plot(tim,aledr/1.5,color = 'r',label = 'ALED')
# plt.plot(tim,zffsr/1.2,color = 'g',label = 'ZFF')
# #plt.plot(tim,np.abs(confidencer),color = 'b', label = 'energy')
# plt.plot(tim,confidencer,color = 'b',label = 'confidence')
# plt.legend()
# plt.xlabel('Time (sec)')
# plt.show()
#print(out).
#plt.plot(aledLabel/1.2,color = 'r',label = 'aled');plt.plot(out/3,color = 'b',label = '2pass')
#plt.plot(zffsLabel/1.5,color = 'g',label = 'zffs')
#plt.show()

## if silence segment in zffsLabel>=time second, retain it
# time = 1 # in seconds
# noFrames = time*100
# backTrace = np.concatenate(([0],((np.where(np.diff(zffsLabel!=0))[0])+1),[len(zffsLabel)]))
# segLen = np.diff(backTrace)
# # total number of frames in non-speech segments
# if(zffsLabel[0]==0):
# 	noZeros = segLen[0::2]
# 	indices3 = np.where(noZeros>=noFrames)[0] # extract bigger length non-speech segments from zffs+ts+ho
# 	for i in indices3:
# 		start = backTrace[i*2]
# 		stop = backTrace[(i*2) + 1]
# 		out[start:stop] = np.zeros(stop-start)

# if(zffsLabel[0]==1):
# 	noZeros = segLen[1::2]
# 	indices3 = np.where(noZeros>=noFrames)[0] 
# 	for i in indices3:
# 		start = backTrace[(i*2) + 1]
# 		stop = backTrace[(i*2) + 2]
# 		out[start:stop] = np.zeros(stop-start)



# zffsLabel = zffsLabel1
# indices = np.where(aledLabel==0)[0] # we are confident about aled-labelled non-speech
# zffsLabel[indices] = np.zeros(len(indices)) # 1st pass through aled
# temp = aledLabel + zffsLabel
# indices1 = np.where(temp==2)[0] # 2nd pass through zffs
# out = np.zeros(length)
# out[indices1] = np.ones(len(indices1))

np.savetxt(txtName,out,fmt='%.0f')

# plt.figure()
# plt.subplot(3,1,1)
# plt.plot(aledLabel)
# plt.ylim([-0.5,1.5])

# plt.subplot(3,1,2)
# plt.plot(zffsLabel1)
# plt.ylim([-0.5,1.5])

# plt.subplot(3,1,3)
# plt.plot(out)
# plt.ylim([-0.5,1.5])

# plt.show()