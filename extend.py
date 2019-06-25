import numpy as np
from numpy import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import scipy.io.wavfile as wav
import sys

# extend zffs detected speech segments by 250 ms on each side, after temporal smoothing

fileName = sys.argv[1]
flag = sys.argv[2]
fileName = fileName.split('s/')[1]
startDir = "/home/swar/swar/shreeharsha/VAD/Ankita_VAD_results/"
if(flag=='1'):
	zffs = open("data/zffs/"+fileName+"_zffsmod.txt","rw+")
	saveName = "data/zffs/"+fileName+"_zffsmod.txt"
elif(flag=='2'):
	zffs = open(startDir + "zffs/"+fileName+"_zffs.txt","rw+")
	saveName = startDir+"zffs/"+fileName+"_zffho.txt"
zffsLabel =array(map(int,zffs.readlines()))
xtime = 0.1 # multiple of 0.01 preferably
xFrames = xtime/0.01

zerotoone = np.where(np.diff(zffsLabel)==1)[0]
onetozero = np.where(np.diff(zffsLabel)==-1)[0]

for i in zerotoone:
	j = i+1
	if (j-xFrames>=0):
		zffsLabel[j-xFrames:j] = np.ones(xFrames)
	else:
		zffsLabel[0:j] = np.ones(j)

for i in onetozero:
	j = i+1
	if (j+xFrames<=len(zffsLabel)):
		zffsLabel[j:j+xFrames] = np.ones(xFrames)
	else:
		zffsLabel[j:len(zffsLabel)] = np.ones(len(zffsLabel)-j)

np.savetxt(saveName,zffsLabel,fmt='%.0f')