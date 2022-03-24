
"""

"""
import config as cfg
from utils import calculate_logmels1, serialize_features

import os
import pathlib

class Features:
    
    def __init__(self):
    
        # self.paths = paths
        # self.audio_feature_parameters = audio_feature_parameters
        # self.visual_feature_parameters = visual_feature_parameters
        # self.action_parameters = action_parameters
        

        # paths
        self.audio_data_path = cfg.paths['audio_data_path']
        self.audio_feature_path = cfg.paths['audio_feature_path']
        self.dataset_name = cfg.paths['dataset_name']
        self.feature_name = cfg.paths['feature_name']

        # Audio features parameters
        self.number_of_mel_bands = cfg.audio_feature_parameters['number_of_mel_bands']
        self.window_len_in_seconds = cfg.audio_feature_parameters['window_len_in_seconds']
        self.window_hop_in_seconds = cfg.audio_feature_parameters['window_hop_in_seconds']
        self.sr_target = cfg.audio_feature_parameters['sr_target']

        # Visual features parameters
        self.vgg_layer_name = cfg.visual_feature_parameters['vgg_layer_name']
        
        # action parameters
        self.extracting_audio_features = cfg.action_parameters['extracting_audio_features']   
        self.extracting_visual_features = cfg.action_parameters ['extracting_visual_features']
        self.processing_train_data = cfg.action_parameters ['processing_train_data']
        self.processing_validation_data = cfg.action_parameters ['processing_validation_data']
        
    def read_file_paths (self, dataset_name):
       
       if dataset_name == "SPOKEN-COCO":
            
           self.json_path_train = os.path.join( self.audio_data_path , 'SpokenCOCO_train.json' )
           self.json_path_val = os.path.join( self.audio_data_path , 'SpokenCOCO_val.json' )
           
           self.audio_path_train = os.path.join( self.audio_data_path , 'wavs' , 'train')  
           self.audio_path_val = os.path.join( self.audio_data_path , 'wavs' , 'val')
           
           if self.processing_train_data:
               self.audio_path = self.audio_path_train
               self.feature_path =  os.path.join(self.audio_feature_path , "SPOKEN-COCO" , "train")
               os.makedirs(self.feature_path , exist_ok= True)
           elif self.processing_validation_data:
               self.audio_path = self.audio_path_val
               self.feature_path = os.path.join( self.audio_feature_path , "SPOKEN-COCO" , "val")
               os.makedirs(self.feature_path , exist_ok= True)
               
  
    
    def find_logmel_features(self, wavfile):
        logmel_feature = calculate_logmels1 (wavfile , self.number_of_mel_bands , self.window_len_in_seconds , self.window_hop_in_seconds , self.sr_target)
        return logmel_feature
    
    def save_logmel_features (self, input_file , feature_fullpath , save_name):
        filename = os.path.join(feature_fullpath, save_name)
        serialize_features (input_file, filename)
        pass
       
    def extract_audio_features (self, dataset_name):
        self.read_file_paths (dataset_name)
        
        
        if dataset_name == "SPOKEN-COCO":
            folders = os.listdir(self.audio_path)
            for folder_name in folders:
                print(folder_name)
                feature_fullpath = os.path.join(self.feature_path, folder_name)  
                os.makedirs(feature_fullpath, exist_ok= True)
                
                files = os.listdir(os.path.join(self.audio_path,folder_name))
                for file_name in files:
                    wavfile = os.path.join(self.audio_path, folder_name , file_name)
                    logmel = self.find_logmel_features(wavfile)
                    save_name = file_name[0:-4] 
                    self.save_logmel_features(logmel, feature_fullpath , save_name)
    
    def __call__(self):
        pass