import numpy
import scipy.spatial as ss
import tensorflow as tf
from tensorflow.keras import backend as K

###############################################################################
                            # recall #
###############################################################################

def calculate_recallat10( embedding_1,embedding_2, sampling_times, number_of_all_audios, pool):   
    recall_all = []
    recallat = 10  
    for trial_number in range(sampling_times):      
        data_ind = numpy.random.randint(0, high=number_of_all_audios, size=pool)       
        vec_1 = [embedding_1[item] for item in data_ind]
        vec_2 = [embedding_2[item] for item in data_ind]           
        distance_utterance = ss.distance.cdist( vec_1 , vec_2 ,  'cosine') # 1-cosine
       
        r = 0
        for n in range(pool):
            ind_1 = n #random.randrange(0,number_of_audios)                   
            distance_utterance_n = distance_utterance[n]            
            sort_index = numpy.argsort(distance_utterance_n)[0:recallat]
            r += numpy.sum((sort_index==ind_1)*1)   
        recall_all.append(r)
        del distance_utterance  
        
    return recall_all


###############################################################################
                    # weak supervision ( triplet loss) #
###############################################################################

def prepare_triplet_data (Ydata, Xdata):    
    #..........................................................Triplet
    n_samples = len(Ydata)
    orderX,orderY = randOrder(n_samples)
    
    bin_triplet = numpy.array(make_bin_target(n_samples)) 
    Ydata_triplet = Ydata[orderY]
    Xdata_triplet = Xdata[orderX]
    return Ydata_triplet, Xdata_triplet, bin_triplet

  
def make_bin_target (n_sample):
    target = []
    for group_number in range(n_sample):    
        target.append(1)
        target.append(0)
        target.append(0)
        
    return target        

def randOrder(n_t):
    random_order = numpy.random.permutation(int(n_t))
    random_order_X = numpy.random.permutation(int(n_t))
    random_order_Y = numpy.random.permutation(int(n_t))
    
    data_orderX = []
    data_orderY = []     
    for group_number in random_order:
        
        data_orderX.append(group_number)
        data_orderY.append(group_number)
        
        data_orderX.append(group_number)
        data_orderY.append(random_order_Y[group_number])
        
        data_orderX.append(random_order_X[group_number])
        data_orderY.append(group_number)
        
    return data_orderX,data_orderY
    
 

def triplet_loss(y_true,y_pred):    
    margin = 0.1
    Sp = y_pred[0::3]
    Si = y_pred[1::3]
    Sc = y_pred[2::3]      
    return K.sum(K.maximum(0.0,(Sc-Sp + margin )) + K.maximum(0.0,(Si-Sp + margin )),  axis=0) 

###############################################################################
                                   # MMS #
############################################################################### 


            
def prepare_MMS_data (Ydata, Xdata ,shuffle_data = False):
    n_samples = len(Xdata)     
    target = numpy.ones( n_samples)
    if shuffle_data:           
        random_order = numpy.random.permutation(int(n_samples)) 
        Ydata_MMS = Ydata[random_order ]
        Xdata_MMS = Xdata[random_order ]                        
    else:
        Ydata_MMS = Ydata
        Xdata_MMS = Xdata
    
    return [Ydata_MMS, Xdata_MMS] , target

def mms_loss(y_true , y_pred ): 
    
    out_visual = y_pred [:,0,:]
    out_audio = y_pred [:,1,:]

    out_visual = K.expand_dims(out_visual, 0)
    out_audio = K.expand_dims(out_audio, 0)

    S = K.squeeze( K.batch_dot(out_audio, out_visual,axes=[-1,-1]) , axis = 0)
    # ...................................................... method 1
    P1 = K.softmax(S, axis = 0) #row-wise softmax
    P2 = K.softmax(S, axis = 1) #column-wise softmax
    
    P1 = P1 + 0.05
    P2 = P2 + 0.05
    
    Y_hat1 = tf.linalg.diag_part (P1)
    Y_hat2 = tf.linalg.diag_part (P2)   
    # ......................................................
    
    I2C_loss = K.sum ( - (K.log(Y_hat1)) , axis = 0)
    C2I_loss = K.sum ( -  (K.log(Y_hat2)) , axis = 0) 
    
    loss = I2C_loss + C2I_loss
    return loss

###############################################################################
                    # supervised similarity (bert) #
############################################################################### 

from similarity_analysis import bert_similarity


def find_y_true (pairs, similarity_type):
    pass
            
def prepare_supervised_data (Ydata, Xdata, Znames):
    
    n_samples = len(Ydata)
    orderX,orderY = randOrder(n_samples)
    pairX = [Znames [i] for i in orderX]
    pairY = [Znames [i] for i in orderY]
    cosine_similarity_all = bert_similarity (pairX, pairY)
    target_triplet = numpy.array(cosine_similarity_all) 
    target_triplet = target_triplet **10
    Ydata_triplet = Ydata[orderY]
    Xdata_triplet = Xdata[orderX]
    return Ydata_triplet, Xdata_triplet, target_triplet