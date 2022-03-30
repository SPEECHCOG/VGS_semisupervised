
###################### initial configuration  #################################


paths = {
  "feature_path_SPOKENCOCO": "../../features/SPOKENCOCO/",
  "feature_path_MSCOCO": "../../features/MSCOCO/",
  "dataset_name" : "SPOKEN-COCO",
  "modeldir": "../../model/model4/",
  "featuredir": "/run/media/hxkhkh/khazar_data_1/khazar/features/coco/SPOKEN-COCO/",
  #"featuredir": "/run/media/hxkhkh/b756dee3-de7e-4cdd-883b-ac95a8d00407/features/coco/SPOKEN-COCO/train/",
}


action_parameters = {
  "use_pretrained": False,
  "training_mode": True,
  "evaluating_mode": True,
  "save_model":False,
  "save_best_recall" : False,
  "save_best_loss" : True,
  "find_recall" : False,
  "number_of_epochs" : 10,
  "n_caps_per_image":5,
  "chunk_length":10000
}

feature_settings = {
    "model_name": "CNNatt",
    "model_subname": "v0",
    "length_sequence" : 512,
    "Xshape" : (512,40),
    "Yshape" : (14,14,512)
    }