# VAD
All scripts and ReadMe for Voice Activity Detection(VAD)

./twoPass.sh -- bash script to run the VAD on all wav files in a folder and get the final VAD output (Binary decisions 1 = speech and 0 = silence) for each 10 ms frame in an audio wav file.

Path names have to be changed appropriately in each file for local use (Note the use of splits and Start Directories while changing path names).

The complete VAD algorithm with the pre-processing, post-processing stages and the results on a sample dataset are explained here: 
https://docs.google.com/document/d/1DzAEFAMArOZe_eZ5yWhffIqZguJlCn9gye3XXxO9Tdw/edit?usp=sharing

The final VAD decisions are text files of the form "audio_filename.wav.txt_pp.txt". VAD has been tested on 16kHz and 8kHz wav files and 10ms frame size.
