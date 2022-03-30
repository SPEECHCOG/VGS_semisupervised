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

# method 1
# number_of_max_pairs = 10
# for counter_row , row in enumerate(sim):
#     thresh = row[counter_row]
#     result = numpy.where(row>thresh)
#     if numpy.size(result[0]) > number_of_max_pairs:
#         result = numpy.where(row>thresh + 0.05)
#         if numpy.size(result[0]) > number_of_max_pairs:
#             result = numpy.where(row>thresh + 0.1) 
#     print(numpy.size(result[0]))
    