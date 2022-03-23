
"""

"""

import pickle
import numpy 
import librosa


def calculate_logmels0 (wavfile , number_of_mel_bands , window_len_in_ms , window_hop_in_ms , sr_target):
    
    win_len_sample = int (sr_target * window_len_in_ms)
    win_hop_sample = int (sr_target * window_hop_in_ms)
    n_fft = 512    
    y, sr = librosa.load(wavfile)
    y = librosa.core.resample(y, sr, sr_target) 
      
    mel_feature = librosa.feature.melspectrogram(y, sr=sr_target, n_fft=n_fft, window='hann',
                                                 win_length = win_len_sample, hop_length=win_hop_sample, n_mels=number_of_mel_bands, power = 2)
    # replacing zeros with small/min number
    zeros_mel = mel_feature[mel_feature==0]          
    if numpy.size(zeros_mel)!= 0:
        
        mel_flat = mel_feature.flatten('F')
        mel_temp =[value for counter, value in enumerate(mel_flat) if value!=0]
    
        if numpy.size(mel_temp)!=0:
            min_mel = numpy.min(numpy.abs(mel_temp))
        else:
            min_mel = 1e-12 
           
        mel_feature[mel_feature==0] = min_mel           
    logmel_feature = numpy.transpose(10*numpy.log10(mel_feature))       
    return logmel_feature

def calculate_logmels1 (wavfile , number_of_mel_bands , window_len_in_ms , window_hop_in_ms , sr_target):
    
    win_len_sample = int (sr_target * window_len_in_ms)
    win_hop_sample = int (sr_target * window_hop_in_ms)
    n_fft = 512    
    y, sr = librosa.load(wavfile)
    y = librosa.core.resample(y, sr, sr_target) 
    
    stft = librosa.stft(y, n_fft = n_fft, win_length = win_len_sample, hop_length = win_hop_sample, window='hann')
    esd = (abs(stft))**2

    m = librosa.filters.mel(sr=sr,n_fft = n_fft, n_mels = number_of_mel_bands)
    mel_feature = numpy.dot(m,esd)
    
    # replacing zeros with small/min number
    zeros_mel = mel_feature[mel_feature==0]          
    if numpy.size(zeros_mel)!= 0:
        
        mel_flat = mel_feature.flatten('F')
        mel_temp =[value for counter, value in enumerate(mel_flat) if value!=0]
    
        if numpy.size(mel_temp)!=0:
            min_mel = numpy.min(numpy.abs(mel_temp))
        else:
            min_mel = 1e-12 
           
        mel_feature[mel_feature==0] = min_mel           
    logmel_feature = numpy.transpose(10*numpy.log10(mel_feature))       
    return logmel_feature

def serialize_features (input_file, filename):
    outfile = open(filename,'wb')
    pickle.dump(input_file ,outfile , protocol=pickle.HIGHEST_PROTOCOL)
    outfile.close()

wavfile = '/tuni/groups/3101050_Specog/corpora/SPOKEN-COCO/wavs/val/1/m3zbehjp85if0v-3DUZQ9U6SMOQX7F2O8QQOO22HN5VSO_128372_441816.wav'

logmel0 = calculate_logmels0 (wavfile , 40 , 0.025 , 0.010 , 16000)
logmel1 = calculate_logmels1 (wavfile , 40 , 0.025 , 0.010 , 16000)

from matplotlib import pyplot as plt
plt.subplot(1,2,1)
plt.imshow(logmel0)
plt.subplot(1,2,2)
plt.imshow(logmel1)