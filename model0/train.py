import os
import numpy
import scipy.io
from matplotlib import pyplot as plt

from tensorflow import keras
from data_preprocessing import prepare_XY , read_feature_filenames,  expand_feature_filenames
from utils import  prepare_triplet_data ,  triplet_loss , calculate_recallat10
from model import VGS
import config as cfg
    
class train_validate (VGS):
    
    def __init__(self):
        VGS.__init__(self)
        
        # paths
        self.feature_path_SPOKENCOCO = cfg.paths['feature_path_SPOKENCOCO']
        self.feature_path_MSCOCO = cfg.paths['feature_path_MSCOCO']
        self.json_path_SPOKENCOCO = cfg.paths["json_path_SPOKENCOCO"]
        self.dataset_name = cfg.paths['dataset_name']
        self.model_dir = cfg.paths['modeldir']

        # action parameters
        self.use_pretrained = cfg.action_parameters['use_pretrained']
        self.training_mode = cfg.action_parameters['training_mode']
        self.evaluating_mode = cfg.action_parameters['evaluating_mode']
        self.saving_mode = cfg.action_parameters['save_model']
        self.save_best_recall = cfg.action_parameters['save_best_recall']
        self.save_best_loss = cfg.action_parameters['save_best_loss']
        self.find_recall = cfg.action_parameters['find_recall']
        self.number_of_epochs = cfg.action_parameters['number_of_epochs']
        self.chunk_length = cfg.action_parameters['chunk_length']
        
        # model setting
        self.model_name = cfg.feature_settings ['model_name']
        self.model_subname = cfg.feature_settings ['model_subname']
        self.length_sequence = cfg.feature_settings['length_sequence']
        self.Xshape = cfg.feature_settings['Xshape']
        self.Yshape = cfg.feature_settings['Yshape']
        self.input_dim = [self.Xshape,self.Yshape] 
        
        self.length_sequence = self.Xshape[0]
        
        
        
        self.split = 'train'
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
        split = self.split
        if self.dataset_name == "SPOKEN-COCO":
            feature_names = read_feature_filenames (self.json_path_SPOKENCOCO, split , shuffle_data = True )  
            #feature_names_chunked = chunk_feature_filenames (feature_names, self.chunk_length)
            n = len(feature_names)
            Ynames_all, Xnames_all , Znames_all = [],[],[]

            for start_chunk in range(0, n , self.chunk_length):
                end_chunk = start_chunk + self.chunk_length
                chunk = feature_names [start_chunk:end_chunk ]            
                Ynames, Xnames , Znames = expand_feature_filenames(chunk )
                Ynames_all.append(Ynames)
                Xnames_all.append(Xnames)
                Znames_all.append(Znames)
        return Ynames_all, Xnames_all , Znames_all
    
    def set_feature_paths (self):
        if self.dataset_name == "SPOKEN-COCO":
            self.feature_path_audio = self.feature_path_SPOKENCOCO 
            self.feature_path_image = self.feature_path_MSCOCO 
        
    def train_model(self): 
        self.split = 'train'
        self.set_feature_paths()
               
        Ynames_all, Xnames_all , Znames_all = self.prepare_chunked_names()
        number_of_chunks = len(Ynames_all)
        for counter in range(number_of_chunks):
            Ynames = Ynames_all [counter]
            Xnames = Xnames_all [counter]
            Znames = Znames_all [counter]
            
            Ydata, Xdata = prepare_XY (Ynames, Xnames , self.feature_path_audio , self.feature_path_image , self.length_sequence )
                
            Ydata_triplet, Xdata_triplet, bin_triplet = prepare_triplet_data (Ydata, Xdata) 
            history = self.vgs_model.fit([Ydata_triplet, Xdata_triplet ], bin_triplet, shuffle=False, epochs=1,batch_size=120)                      
            del Ydata, Xdata, Xdata_triplet, Ydata_triplet
        training_output = history.history['loss'][0]
        return training_output

    def evaluate_model (self) :
        self.split = 'val' 
        self.set_feature_paths()
        epoch_cum_val = 0
        epoch_cum_recall_av = 0
        epoch_cum_recall_va = 0
        Ynames_all, Xnames_all , Znames_all = self.prepare_chunked_names()
        number_of_chunks = len(Ynames_all)
        for counter in range(number_of_chunks):
            Ynames = Ynames_all [counter]
            Xnames = Xnames_all [counter]
            Znames = Znames_all [counter]
            
            Ydata, Xdata = prepare_XY (Ynames, Xnames , self.feature_path_audio , self.feature_path_image , self.length_sequence )
            Ydata_triplet, Xdata_triplet, bin_triplet = prepare_triplet_data (Ydata, Xdata) 
            loss = self.vgs_model.evaluate([Ydata_triplet, Xdata_triplet ], bin_triplet, batch_size=120) 
            epoch_cum_val += loss
            #............................................................. Recall
            if self.find_recall:

                number_of_samples = len(Ydata)
                visual_embeddings = self.visual_embedding_model.predict(Ydata)
                visual_embeddings_mean = numpy.mean(visual_embeddings, axis = 1) 

                audio_embeddings = self.audio_embedding_model.predict(Xdata)
                audio_embeddings_mean = numpy.mean(audio_embeddings, axis = 1)                 
  
                ########### calculating Recall@10                    
                poolsize =  1000
                number_of_trials = 100
                recall_av_vec = calculate_recallat10( audio_embeddings_mean, visual_embeddings_mean, number_of_trials,  number_of_samples , poolsize )          
                recall_va_vec = calculate_recallat10( visual_embeddings_mean , audio_embeddings_mean, number_of_trials,  number_of_samples , poolsize ) 
                recall10_av = numpy.mean(recall_av_vec)/(poolsize)
                recall10_va = numpy.mean(recall_va_vec)/(poolsize)
                epoch_cum_recall_av += recall10_av
                print(epoch_cum_recall_av)
                epoch_cum_recall_va += recall10_va               
                del Xdata, audio_embeddings
                del Ydata, visual_embeddings            
            del Xdata_triplet, Ydata_triplet
            
        final_recall_av = epoch_cum_recall_av / (number_of_chunks ) 
        final_recall_va = epoch_cum_recall_va / (number_of_chunks ) 
        final_valloss = epoch_cum_val/ number_of_chunks
        
        validation_output = [final_recall_av, final_recall_va , final_valloss]
        print(validation_output)
        return validation_output
    
    def save_model(self, initialized_output , training_output, validation_output):
        
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
                self.vgs_model.save_weights('%smodel_weights.h5' % self.model_dir)
        else :
            if epoch_valloss <= val_indicator: 
                val_indicator = epoch_valloss
                # weights = vgs_model.get_weights()
                # vgs_model.set_weights(weights)
                self.vgs_model.save_weights('%smodel_weights.h5' % self.model_dir)
                      
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
 
    def define_and_compile_models (self):
        self.vgs_model, self.visual_embedding_model, self.audio_embedding_model = self.build_model(self.model_name, self.model_subname, self.input_dim)
        self.vgs_model.compile(loss=triplet_loss, optimizer= keras.optimizers.Adam(lr=1e-04))
        print(self.vgs_model.summary())
        
    def __call__(self):
    
        self.define_and_compile_models()
  
        initialized_output = self.initialize_model_parameters()
        
        if self.use_pretrained:
            self.vgs_model.load_weights(self.model_dir + 'model_weights.h5')

        for epoch_counter in numpy.arange(self.number_of_epochs):
            
            print('......... epoch ...........' , str(epoch_counter))
            
            if self.training_mode:
                #training_output = self.train_model(vgs_model)
                training_output = self.train_model()
            else:
                training_output = 0
                
            if self.evaluating_mode:
                
                validation_output = self.evaluate_model()
            else: 
                validation_output = [0, 0 , 0 ]
                
            if self.saving_mode:
                self.save_model(initialized_output, training_output, validation_output)
                

