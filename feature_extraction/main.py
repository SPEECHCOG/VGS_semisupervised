from extract_features import Features
########################## initial configuration ##############################

# import config as cfg

# paths
# audio_data_path = cfg.paths['audio_data_path']
# audio_feature_path = cfg.paths['audio_feature_path']
# dataset_name = cfg.paths['dataset_name']
# feature_name = cfg.paths['feature_name']

# # Audio features parameters
# number_of_mel_bands = cfg.audio_feature_parameters['number_of_mel_bands']
# window_len_in_seconds = cfg.audio_feature_parameters['window_len_in_seconds']
# window_hop_in_seconds = cfg.audio_feature_parameters['window_hop_in_seconds']
# sr_target = cfg.audio_feature_parameters['sr_target']

# # Visual features parameters
# vgg_layer_name = cfg.visual_feature_parameters['vgg_layer_name']

# #.............................


# extracting_audio_features = cfg.action_parameters['extracting_audio_features']   
# extracting_visual_features = cfg.action_parameters ['extracting_visual_features']
# processing_train_data = cfg.action_parameters ['processing_train_data']
# processing_validation_data = cfg.action_parameters ['processing_validation_data']

###############################################################################

obj = Features()
#
obj.extract_visual_features("SPOKEN-COCO")


