# ! /bin /bash

for file in /home/swar/swar/shreeharsha/VAD/Ankita_VAD_new/aser_audios/*.wav; do 
# ipython aled.py "BAB${i}" 2< warnings.txt;
# ipython post_proc.py "BAB${i}" 0 2< warnings.txt
# ipython aledHangover.py "BAB${i}" 2< warnings.txt
# ipython zffs_1011.py "BAB${i}" 2< warnings.txt;
# ipython post_proc.py "BAB${i}" 1 2< warnings.txt;
# ipython extend.py "BAB${i}" 2< warnings.txt;
# ipython twoPass.py "BAB${i}" 2< warnings.txt; 
# ipython post_proc.py "BAB${i}" 2 2< warnings.txt;
# ipython validationFramework.py "BAB${i}" 2< warnings.txt; 
# done

#i=$1
echo "$file"
echo "ALED"
python aled_avin.py "$file" #2< warnings.txt;
echo "ALED post processing"
python post_proc.py "${file}_aled.txt" 10 aled ## 10 is parameter for frame size i.e 10 ms

echo "ZFFS"
python zffs.py "$file" #2< warnings.txt;
echo "ZFFS post processing"
python post_proc.py "${file}_zffs.txt" 10 zffs ## 10 is parameter for frame size i.e. 10 ms
echo "ZFFS extend"
python extend.py "$file" 2 #2< warnings.txt;
echo "Two pass system"
python twoPass_mod2.py "${file}" 2 0; ### 3rd argument (default 0) is to set the background noise level
									 ### back_noise == '0': ##normal background
									 ### back_noise == '1': ## Loud background talkers & loud speaker voice
									 ### back_noise == '2': ## Speaker's voice too soft
#echo "making textGrid"
#ipython toCTM.py "BAB${i}" 4 2<warnings.txt;
## write combine 1s aqnd 0s and onvert ottext grid
#cd VAD4LETS/aledOut/exp
#paste ones.txt ones.txt detected.txt id.txt -d " " > combine.ctm
done

for filen in /home/swar/swar/shreeharsha/VAD/Ankita_VAD_results/twopass/*.txt;do

echo "Final post_processing"
python post_proc_final.py $filen 10;
done