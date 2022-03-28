
###################### initial configuration  #################################


paths = {
  "path_SPOKENCOCO": "/tuni/groups/3101050_Specog/corpora/SPOKEN-COCO/",
  "path_MSCOCO" : "/tuni/groups/3101050_Specog/corpora/Image_COCO",
  "feature_path_SPOKENCOCO": "../../features/SPOKENCOCO/",
  "feature_path_MSCOCO": "../../features/MSCOCO/",
  "dataset_name" : "SPOKEN-COCO",
}



audio_feature_parameters = {
    "number_of_mel_bands" : 40,
    "window_len_in_seconds" : 0.025,
    "window_hop_in_seconds" : 0.01,
    "sr_target" : 16000
    }

visual_feature_parameters = {
    "visual_feature_name" : "vgg",
    "visual_feature_subname" : 'block5_conv3'
    }

action_parameters = {
    "extracting_audio_features" : False,
    "extracting_visual_features" : False,
    "processing_train_data" : True,
    "processing_validation_data" : False
    }