###############################################################################

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
config.gpu_options.per_process_gpu_memory_fraction=0.7
sess = tf.Session(config=config)

 

###############################################################################

from train import train_validate



run_training_and_validation = train_validate ()
run_training_and_validation()

    