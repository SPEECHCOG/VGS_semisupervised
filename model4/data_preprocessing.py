
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



json_file = "/tuni/groups/3101050_Specog/corpora/SPOKEN-COCO/SpokenCOCO_train.json"
split = 'train'
output = get_SPOKENCOCO_data (json_file , 'train')

feature_path_SPOKENCOCO = "../../features/SPOKENCOCO/"
feature_path_MSCOCO = "../../features/MSCOCO/"

def prepare_chunked_names (self):
    # 1. read all data names
    # 2. shuffle data names
    # 3. divide into chunks
    if self.dataset_name == "SPOKEN-COCO":

        feature_path_audio = os.path.join(self.feature_path_SPOKENCOCO , self.split)
        feature_path_image = os.path.join(self.feature_path_MSCOCO , self.split)
        data_path_json = self.json_path_SPOKENCOCO
        Ydata_all_initial, Xdata_all_initial = read_file_names (feature_path_audio, feature_path_image, data_path_json ) 
        Ydata_all, Xdata_all = shuffle_file_names (Ydata_all_initial, Xdata_all_initial)
        data_length = len(Ydata_all)
        data_all = []
        for start_chunk in range(0, data_length ,self.chunk_length):
            Ydata_chunk = Ydata_all [start_chunk:start_chunk +self.chunk_length ]
            # Xdata contains 5 captions per item
            Xdata_chunk = Xdata_all [start_chunk:start_chunk +self.chunk_length ]
            chunk = {}
            chunk['Ydata'] = Ydata_chunk
            chunk['Xdata'] = Xdata_chunk
            data_all.append(chunk)
    return data_all
    
    
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
