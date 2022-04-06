
###################### initial configuration  #################################


paths = {
  "feature_path_SPOKENCOCO": "../../features/SPOKEN-COCO/",
  "feature_path_MSCOCO": "../../features/MSCOCO/",
  "json_path_SPOKENCOCO" : "/tuni/groups/3101050_Specog/corpora/SPOKEN-COCO/",
  "dataset_name" : "SPOKEN-COCO",
  "modeldir": "../../model/model4/",
}


action_parameters = {
  "use_pretrained": True,
  "training_mode": True,
  "evaluating_mode": True,
  "save_model":True,
  "save_best_recall" : True,
  "save_best_loss" : False,
  "find_recall" : True,
  "number_of_epochs" : 100,
  "chunk_length":2000
}

feature_settings = {
    "model_name": "CNNatt",
    "model_subname": "v0",
    "length_sequence" : 512,
    "Xshape" : (512,40),
    "Yshape" : (14,14,512)
    }
