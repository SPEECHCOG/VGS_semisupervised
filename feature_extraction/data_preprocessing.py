
"""
"""

import json
import os

import numpy
import pickle

# file = "/tuni/groups/3101050_Specog/corpora/SPOKEN-COCO/SpokenCOCO_train.json"
# infile = open(file, 'rb')
# data = json.load(infile)
# infile.close()

# train_data = data['data']

# train_image_paths = []
# train_image_captions = []
# for item in train_data:
#     img_fullname = item['image']
#     img_data = item['captions']
    
#     train_image_paths.append(img_fullname)
#     train_image_captions.append(img_data)
    
    
    
    # img_name = img_fullname.split('/')[1]
    # outfile_name = img_name[:-4]
    
    
def get_SPOKENCOCO_imagenames (json_file):
        
    infile = open(json_file, 'rb')
    data = json.load(infile)
    infile.close()

    train_data = data['data']

    train_image_paths = []
    train_image_captions = []
    for item in train_data:
        img_fullname = item['image']
        img_data = item['captions']
        
        train_image_paths.append(img_fullname)
        train_image_captions.append(img_data)
        
    return train_image_paths 



json_file = "/tuni/groups/3101050_Specog/corpora/SPOKEN-COCO/SpokenCOCO_train.json"
output = get_SPOKENCOCO_imagenames (json_file)

