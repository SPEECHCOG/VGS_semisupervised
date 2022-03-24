
###################### initial configuration  #################################


paths = {
  "audio_data_path": "/tuni/groups/3101050_Specog/corpora/SPOKEN-COCO/",
  "audio_feature_path": "../../features/",
  "dataset_name" : "SPOKEN-COCO",
  "feature_name" : "logmel40_"
}



audio_feature_parameters = {
    "number_of_mel_bands" : 40,
    "window_len_in_seconds" : 0.025,
    "window_hop_in_seconds" : 0.01,
    "sr_target" : 16000
    }

visual_feature_parameters = {
    "vgg_layer_name" : 'block5_conv3'
    }

action_parameters = {
    "extracting_audio_features" : False,
    "extracting_visual_features" : False,
    "processing_train_data" : True,
    "processing_validation_data" : False
    }