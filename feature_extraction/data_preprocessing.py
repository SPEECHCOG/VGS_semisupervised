
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

def loadXdata (filename, len_of_longest_sequence , i_cap):
    infile = open(filename ,'rb')
    logmel = pickle.load(infile)
    infile.close()
    logmel_i = [item[i_cap] for item in logmel]
    Xdata = preparX (logmel_i, len_of_longest_sequence ,)
    del logmel
    return Xdata
    
def loadYdata (filename):
    infile = open(filename ,'rb')
    vgg = pickle.load(infile)
    infile.close()
    Ydata = preparY(vgg)
    del vgg 
    return Ydata

    
def preparX (dict_logmel, len_of_longest_sequence):
    number_of_audios = numpy.shape(dict_logmel)[0]
    number_of_audio_features = numpy.shape(dict_logmel[0])[1]
    X = numpy.zeros((number_of_audios ,len_of_longest_sequence, number_of_audio_features),dtype ='float32')
    for k in numpy.arange(number_of_audios):
       logmel_item = dict_logmel[k]
       logmel_item = logmel_item[0:len_of_longest_sequence]
       X[k,len_of_longest_sequence-len(logmel_item):, :] = logmel_item
    return X


def preparY (dict_vgg):
    Y = numpy.array(dict_vgg)    
    return Y


def shuffle_file_names (Ydata_all_initial, Xdata_all_initial):
    pass

def read_file_names (feature_path_audio, feature_path_image ):
    pass


def prepare_XY (Ydata_names, Xdata_names):
     #...................................................................... Y 
    filename = ''
    Ydata = loadYdata(filename)
    #.................................................................. X
    filename = ''
    Xdata = loadXdata(filename) 

    return Ydata, Xdata 
