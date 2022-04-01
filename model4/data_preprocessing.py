
"""
"""

import json
import os
import numpy
import pickle
from sklearn.utils import shuffle

    
    
def get_SPOKENCOCO_data (json_file , split):
        
    infile = open(json_file, 'rb')
    content = json.load(infile)
    infile.close()

    json_data = content['data']
    
    data = []
    #iterating over images
    for json_item in json_data:
        item_dict = {}
        img_fullname = json_item['image']
        img_data = json_item['captions']
        
        item_dict ['image_name'] = img_fullname     
        img_feature_name = img_fullname.split('/')[1][:-4]     
        item_dict ['visual_feature'] = os.path.join(split,img_feature_name )
        
        item_dict['audio_data'] = []
        # iterating over captions of each image
        for caption_item in img_data:
            audio_info = {}
            
            wav_name = caption_item ['wav']
            audio_info ['wav_name'] = wav_name
            audio_info ['audio_feature'] = os.path.join(wav_name.split('/')[1],
                                                        wav_name.split('/')[2],
                                                        wav_name.split('/')[3][:-4])
                                                        
            audio_info ['text_caption'] = caption_item ['text']
            item_dict['audio_data'].append(audio_info)
        data.append(item_dict)
    return data 





feature_path_SPOKENCOCO = "../../features/SPOKENCOCO/"
feature_path_MSCOCO = "../../features/MSCOCO/"
    

def read_file_names (path_json, split, shuffle = True ):
    
    if split == 'train':
        json_name = 'SpokenCOCO_train.json'
    elif split == 'val':
        json_name = 'SpokenCOCO_val.json'
    json_file = os.path.join(path_json , json_name)
    
    data = get_SPOKENCOCO_data (json_file , split)
    
    Ydata_all_initial = []
    Xdata_all_initial = []
    Zdata_all_initial = []
     
    #iterating over images
    for element in data:
        
        audio_data = element['audio_data']
        #iterating over captions per image
        for caption_item in audio_data:
            Ydata_all_initial.append(element['visual_feature'])
            Xdata_all_initial.append(caption_item['audio_feature'])
            Zdata_all_initial.append(caption_item['text_caption'])
            
    if shuffle:
        inds_shuffled = shuffle(numpy.arange(len(data)))       
    else:
        inds_shuffled = numpy.arange(len(data))
        
    # Ydata_all_initial = numpy.array(Ydata_all_initial)
    # Xdata_all_initial = numpy.array(Xdata_all_initial)
    # Zdata_all_initial = numpy.array(Zdata_all_initial) 
    
    # Ydata_all = Ydata_all_initial[inds_shuffled]
    # Xdata_all = Xdata_all_initial[inds_shuffled]
    # Zdata_all = Zdata_all_initial[inds_shuffled]
        
    return Ydata_all_initial, Xdata_all_initial , Zdata_all_initial    

def chunk_file_names (Ydata_all, Xdata_all , Zdata_all , chunk_length):
    
    data_length = len(Ydata_all)
    data_chunked = []
    for start_chunk in range(0, data_length , chunk_length):
        Ydata_chunk = Ydata_all [start_chunk:start_chunk +chunk_length ]
        Xdata_chunk = Xdata_all [start_chunk:start_chunk +chunk_length ]
        Zdata_chunk = Zdata_all [start_chunk:start_chunk +chunk_length ]
        
        chunk = {}
        chunk['Ydata'] = Ydata_chunk
        chunk['Xdata'] = Xdata_chunk
        chunk['Zdata'] = Zdata_chunk
        data_chunked.append(chunk)
        return data_chunked
    
    
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





def prepare_XY (Ydata_names, Xdata_names):
     #...................................................................... Y 
    filename = ''
    Ydata = loadYdata(filename)
    #.................................................................. X
    filename = ''
    Xdata = loadXdata(filename) 

    return Ydata, Xdata 
