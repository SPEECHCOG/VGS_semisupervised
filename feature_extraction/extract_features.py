
"""

"""
import config as cfg
from utils import calculate_logmels1, calculate_vgg_b5, serialize_features
from data_preprocessing import get_SPOKENCOCO_imagenames
import os
import pathlib

class Features:
    
    def __init__(self):
    
        # paths
        self.path_SPOKENCOCO = cfg.paths['path_SPOKENCOCO']
        self.path_MSCOCO = cfg.paths['path_MSCOCO']
        self.feature_path_SPOKENCOCO = cfg.paths['feature_path_SPOKENCOCO']
        self.feature_path_MSCOCO = cfg.paths['feature_path_MSCOCO']
        self.dataset_name = cfg.paths['dataset_name']

        # Audio features parameters
        self.number_of_mel_bands = cfg.audio_feature_parameters['number_of_mel_bands']
        self.window_len_in_seconds = cfg.audio_feature_parameters['window_len_in_seconds']
        self.window_hop_in_seconds = cfg.audio_feature_parameters['window_hop_in_seconds']
        self.sr_target = cfg.audio_feature_parameters['sr_target']

        # Visual features parameters
        self.visual_feature_name = cfg.visual_feature_parameters['visual_feature_name']
        self.visual_feature_subname = cfg.visual_feature_parameters['visual_feature_subname']
        
        # action parameters
        self.extracting_audio_features = cfg.action_parameters['extracting_audio_features']   
        self.extracting_visual_features = cfg.action_parameters ['extracting_visual_features']
        self.processing_train_data = cfg.action_parameters ['processing_train_data']
        self.processing_validation_data = cfg.action_parameters ['processing_validation_data']

    def save_features (self, input_file , feature_fullpath , save_name):
        filename = os.path.join(feature_fullpath, save_name)
        serialize_features (input_file, filename)
        
    def read_file_paths (self, dataset_name):
       
       if dataset_name == "SPOKEN-COCO":
           
           if self.processing_train_data:
               self.json_file = os.path.join( self.path_SPOKENCOCO , 'SpokenCOCO_train.json' )
               self.audio_path = os.path.join( self.path_SPOKENCOCO , 'wavs' , 'train') 
               self.feature_path_audio =  os.path.join(self.feature_path_SPOKENCOCO , "train")
               self.feature_path_visual =  os.path.join(self.feature_path_MSCOCO , "train")
               
           elif self.processing_validation_data:
               self.json_file = os.path.join( self.path_SPOKENCOCO , 'SpokenCOCO_val.json' )
               self.audio_path = os.path.join( self.path_SPOKENCOCO , 'wavs' , 'val')
               self.feature_path_audio = os.path.join(self.feature_path_SPOKENCOCO , "val")
               self.feature_path_visual = os.path.join(self.feature_path_MSCOCO , "val")
       
       
               
    def find_visual_features (self, image_fullpath):
        if self.visual_feature_name == 'vgg' and self.visual_feature_subname == 'block5_conv3':
            vf_output = calculate_vgg_b5 (image_fullpath)
        return vf_output

        
    def extract_visual_features(self, dataset_name):
        self.read_file_paths (dataset_name)
        os.makedirs(self.feature_path_visual , exist_ok= True)
        
        if dataset_name == "SPOKEN-COCO":
            image_fullnames = get_SPOKENCOCO_imagenames (self.json_file)
            for image_fullname in image_fullnames:
                
                image_fullpath = os.path.join(self.path_MSCOCO, image_fullname)
                vf_output = self.find_visual_features(image_fullpath)
                
                image_name = image_fullname.split('/')[1]
                # remove ".jpg" from name
                save_name = image_name[:-4] 
                self.save_features(vf_output, self.feature_path_visual , save_name)
    
    def find_logmel_features(self, wavfile):
        logmel_feature = calculate_logmels1 (wavfile , self.number_of_mel_bands , self.window_len_in_seconds , self.window_hop_in_seconds , self.sr_target)
        return logmel_feature
    

       
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
                    self.save_features(logmel, feature_fullpath , save_name)
    
    def __call__(self):
        pass