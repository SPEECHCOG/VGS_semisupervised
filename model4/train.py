import os
import numpy
import scipy.io
from matplotlib import pyplot as plt

from tensorflow import keras
from data_preprocessing import prepare_XY , read_file_names, shuffle_file_names
from utils import  prepare_triplet ,  triplet_loss , calculate_recallat10
from model import VGS
import config as cfg
    
class train_validate (VGS):
    
    def __init__(self):
        VGS.__init__(self)
        
        # paths
        self.feature_path_SPOKENCOCO = cfg.paths['feature_path_SPOKENCOCO']
        self.feature_path_MSCOCO = cfg.paths['feature_path_MSCOCO']
        self.dataset_name = cfg.paths['dataset_name']
        self.model_dir = cfg.paths['modeldir']

        # action parameters
        self.use_pretrained = cfg.action_parameters['use_pretrained']
        self.training_mode = cfg.action_parameters['training_mode']
        self.evaluating_mode = cfg.action_parameters['evaluating_mode']
        self.saveing_mode = cfg.action_parameters['save_model']
        self.save_best_recall = cfg.action_parameters['save_best_recall']
        self.save_best_loss = cfg.action_parameters['save_best_loss']
        self.find_recall = cfg.action_parameters['find_recall']
        self.number_of_epochs = cfg.action_parameters['number_of_epochs']
        self.number_of_captions_per_image = cfg.action_parameters['n_caps_per_image']
        self.chunk_length = cfg.action_parameters['chunk_length']
        
        # model setting
        self.model_name = cfg.model_settings ['model_name']
        self.model_subname = cfg.model_settings ['model_subname']
        self.length_sequence = cfg.model_settings['length_sequence']
        self.Xshape = cfg.model_settings['Xshape']
        self.Yshape = cfg.model_settings['Yshape']
        self.input_dim = [self.Xshape,self.Yshape] 
        
        self.length_sequence = self.Xshape[0]
        
        super().__init__() 
        
        
    def initialize_model_parameters(self):
        
        if self.use_pretrained:
            data = scipy.io.loadmat(self.model_dir + 'valtrainloss.mat', variable_names=['allepochs_valloss','allepochs_trainloss','all_avRecalls', 'all_vaRecalls'])
            allepochs_valloss = data['allepochs_valloss'][0]
            allepochs_trainloss = data['allepochs_trainloss'][0]
            all_avRecalls = data['all_avRecalls'][0]
            all_vaRecalls = data['all_vaRecalls'][0]
            
            allepochs_valloss = numpy.ndarray.tolist(allepochs_valloss)
            allepochs_trainloss = numpy.ndarray.tolist(allepochs_trainloss)
            all_avRecalls = numpy.ndarray.tolist(all_avRecalls)
            all_vaRecalls = numpy.ndarray.tolist(all_vaRecalls)
            recall_indicator = numpy.max(allepochs_valloss)
            val_indicator = numpy.min(allepochs_valloss)
        else:
            allepochs_valloss = []
            allepochs_trainloss = []
            all_avRecalls = []
            all_vaRecalls = []
            recall_indicator = 0
            val_indicator = 1000
            
        saving_params = [allepochs_valloss, allepochs_trainloss, all_avRecalls, all_vaRecalls, val_indicator , recall_indicator ]       
        return saving_params 
    
    def prepare_chunked_names (self):
        # 1. read all data names
        # 2. shuffle data names
        # 3. divide into chunks
        if self.dataset_name == "SPOKEN-COCO":

            feature_path_audio = os.path.join(self.feature_path_SPOKENCOCO , self.split)
            feature_path_image = os.path.join(self.feature_path_MSCOCO , self.split)
            Ydata_all_initial, Xdata_all_initial = read_file_names (feature_path_audio, feature_path_image ) 
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
    
    def train_model(self, vgs_model): 
        self.split = 'train'
        if self.use_pretrained:
            vgs_model.load_weights(self.model_dir + 'model_weights.h5')
        self.chunk_length       
        data = self.prepare_chunked_names()
        
        for counter,chunk in enumerate(data):
            Ydata_names = chunk['Ydata']
            Xdata_names_all_captions = chunk['Xdata']
            
            for Xdata_names in Xdata_names_all_captions:
                Ydata, Xdata = prepare_XY (Ydata_names, Xdata_names)
                Ydata_triplet, Xdata_triplet, bin_triplet = prepare_triplet (Ydata, Xdata) 
                vgs_model.fit([Ydata_triplet, Xdata_triplet ], bin_triplet, shuffle=False, epochs=1,batch_size=160)                 
        
                del Xdata_triplet, Ydata_triplet

    def evaluate_model (self, vgs_model,  visual_embedding_model, audio_embedding_model ) :
        self.split = 'val'        
        epoch_cum_val = 0
        epoch_cum_recall_av = 0
        epoch_cum_recall_va = 0
        i_caption = numpy.random.randint(0, 5)
        set_of_input_chunks = self.validation_chunks

        for chunk_counter, chunk_name in enumerate(set_of_input_chunks):
            print('.......... validation chunk ..........' + str(chunk_counter))
            Ydata, Xdata = prepare_XY (self.feature_dir , self.visual_feature_name ,  self.audio_feature_name , chunk_name , i_caption , self.length_sequence)
            #..................................................................... Recall
            if self.find_recall:

                number_of_samples = len(Ydata)
                visual_embeddings = visual_embedding_model.predict(Ydata)
                visual_embeddings_mean = numpy.mean(visual_embeddings, axis = 1) 

                audio_embeddings = audio_embedding_model.predict(Xdata)
                audio_embeddings_mean = numpy.mean(audio_embeddings, axis = 1)                 
                
                
                ########### calculating Recall@10                    
                poolsize =  1000
                number_of_trials = 100
                recall_av_vec = calculate_recallat10( audio_embeddings_mean, visual_embeddings_mean, number_of_trials,  number_of_samples , poolsize )          
                recall_va_vec = calculate_recallat10( visual_embeddings_mean , audio_embeddings_mean, number_of_trials,  number_of_samples , poolsize ) 
                recall10_av = numpy.mean(recall_av_vec)/(poolsize)
                recall10_va = numpy.mean(recall_va_vec)/(poolsize)
                epoch_cum_recall_av += recall10_av
                epoch_cum_recall_va += recall10_va               
                del Xdata, audio_embeddings
                del Ydata, visual_embeddings            
            #del Xdata_triplet,Ydata_triplet
            
        final_recall_av = epoch_cum_recall_av / (chunk_counter + 1 ) 
        final_recall_va = epoch_cum_recall_va / (chunk_counter + 1 ) 
        final_valloss = epoch_cum_val/ len (set_of_input_chunks) 
        
        validation_output = [final_recall_av, final_recall_va , final_valloss]
        print(validation_output)
        return validation_output
    
    def save_model(self, vgs_model, initialized_output , training_output, validation_output):
        
        os.makedirs(self.model_dir, exist_ok=1)
        [allepochs_valloss, allepochs_trainloss, all_avRecalls, all_vaRecalls, val_indicator , recall_indicator ] = initialized_output
        [final_recall_av, final_recall_va , final_valloss] = validation_output 
        [epoch_recall_av, epoch_recall_va , epoch_valloss] = validation_output
               
            
        if self.save_best_recall:
            epoch_recall = ( epoch_recall_av + epoch_recall_va ) / 2
            if epoch_recall >= recall_indicator: 
                recall_indicator = epoch_recall
                # weights = vgs_model.get_weights()
                # vgs_model.set_weights(weights)
                vgs_model.save_weights('%smodel_weights.h5' % self.model_dir)
        else :
            if epoch_valloss <= val_indicator: 
                val_indicator = epoch_valloss
                # weights = vgs_model.get_weights()
                # vgs_model.set_weights(weights)
                vgs_model.save_weights('%smodel_weights.h5' % self.model_dir)
                      
        allepochs_trainloss.append(training_output)  
        allepochs_valloss.append(epoch_valloss)
        if self.find_recall: 
            all_avRecalls.append(epoch_recall_av)
            all_vaRecalls.append(epoch_recall_va)
        save_file = self.model_dir + 'valtrainloss.mat'
        scipy.io.savemat(save_file, 
                          {'allepochs_valloss':allepochs_valloss,'allepochs_trainloss':allepochs_trainloss,'all_avRecalls':all_avRecalls,'all_vaRecalls':all_vaRecalls })  
        
        self.make_plot( [allepochs_trainloss, allepochs_valloss , all_avRecalls, all_vaRecalls ])

        
    def make_plot (self, plot_lists):
        
        plt.figure()
        plot_names = ['training loss','validation loss','speech_to_image recall','image_to_speech recall']
        for plot_counter, plot_value in enumerate(plot_lists):
            plt.subplot(2,2,plot_counter+1)
            plt.plot(plot_value)
            plt.ylabel(plot_names[plot_counter])
            plt.grid()

        plt.savefig(self.model_dir + 'evaluation_plot.pdf', format = 'pdf')
 
    
    def __call__(self):
    
        vgs_model, visual_embedding_model, audio_embedding_model = self.build_model(self.model_name, self.model_subname, self.input_dim)
        vgs_model.compile(loss=triplet_loss, optimizer= keras.optimizers.Adam(lr=1e-04))
        print(vgs_model.summary())
  
        initialized_output = self.initialize_model_parameters()
        
  
        if self.use_pretrained:
            vgs_model.load_weights(self.model_dir + 'model_weights.h5')

        for epoch_counter in numpy.arange(self.number_of_epochs):
            
            print('......... epoch ...........' , str(epoch_counter))
            
            if self.training_mode:
                #training_output = self.train_model(vgs_model)
                training_output = self.train_model(vgs_model,  visual_embedding_model, audio_embedding_model)
            else:
                training_output = 0
                
            if self.evaluating_mode:
                
                validation_output = self.evaluate_model(vgs_model,  visual_embedding_model, audio_embedding_model )
            else: 
                validation_output = [0, 0 , 0 ]
                
            if self.saving_mode:
                self.save_model(vgs_model,initialized_output, training_output, validation_output)
                

