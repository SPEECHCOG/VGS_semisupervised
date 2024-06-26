
###################### initial configuration  #################################


paths = {
  "feature_path_SPOKENCOCO": "../../features/SPOKEN-COCO/",
  "feature_path_MSCOCO": "../../features/MSCOCO/",
  "json_path_SPOKENCOCO" : "../../data/SPOKEN-COCO/",
  "dataset_name" : "SPOKEN-COCO",
  "modeldir": "../../model/model4/",
}


action_parameters = {
  "use_pretrained": False,
  "training_mode": True,
  "evaluating_mode": True,
  "save_model":True,
  "save_best_recall" : True,
  "save_best_loss" : False,
  "find_recall" : True,
  "number_of_epochs" : 50,
  "chunk_length":10000
}

feature_settings = {
    "model_name": "CNN0",
    "model_subname": "v0",
    "length_sequence" : 512,
    "Xshape" : (512,40),
    "Yshape" : (14,14,512)
    }
