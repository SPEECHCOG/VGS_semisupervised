
"""
"""

import scipy.spatial as ss
from scipy import spatial
import numpy
from sent2vec.vectorizer import Vectorizer

vectorizer_obj = Vectorizer()
#vectorizer = Vectorizer( pretrained_weights='bert-base-uncased')


#..............................................................................
#            use this function on the new version of sent2ve                  #
#..............................................................................

# def find_similar_pairs (list_of_sentences):
#     vectors = vectorizer.vectors
#     vectors = vectorizer.run(list_of_sentences)
#     vectors = numpy.array(vectors)
#     #dist = spatial.distance.cosine(vectors[0], vectors[1])
#     dist = ss.distance.cdist( vectors,vectors ,  'cosine')
#     best = {}
#     best['best_pairs'] = []
#     best['similarity'] = []
#     for row in dist:

#         print(row)
#         best_similarity = numpy.sort(row)[1]
#         best_pair = numpy.argsort(row)[1]
#         best['similarity'].append(best_similarity)
#         best['best_pairs'].append(best_pair)      
#     return best    




#..............................................................................
#            use this function on the old version of sent2ve                  #
#..............................................................................


def bert_similarity (pairX, pairY):
    
    vectorizer_obj.bert(pairX)    
    vectors_bert_X = vectorizer_obj.vectors    
    vectorizer_obj.vectors = []

    vectorizer_obj.bert(pairY)    
    vectors_bert_Y = vectorizer_obj.vectors    
    vectorizer_obj.vectors = []
    
    cosine_similarity_all = []
    for i in range(len(vectors_bert_X)):
        vector1 = vectors_bert_X[i]
        vector2 = vectors_bert_Y[i]
        cosine_similarity = 1 - spatial.distance.cosine(vector1, vector2)
        cosine_similarity_all.append(cosine_similarity)
    #dist = ss.distance.cdist( vectors_bert, vectors_bert ,  'cosine')
    #max_similarities = 1 - spatial.distance.cosine(vectors_bert[0],vectors_bert[1])
    return cosine_similarity_all

# def find_similar_pairs (list_of_sentences):
#     dist = bert_distance (list_of_sentences)
#     # vectors = vectorizer.vectors
#     # vectors = vectorizer.bert(list_of_sentences[0],list_of_sentences[1])
#     # vectors = numpy.array(vectors)
#     #dist = spatial.distance.cosine(vectors[0], vectors[1])
#     #dist = ss.distance.cdist( vectors,vectors ,  'cosine')
#     best = {}
#     best['best_pairs'] = []
#     best['similarity'] = []
#     for row in dist:

#         #print(row)
#         best_similarity = numpy.sort(row)[1]
#         best_pair = numpy.argsort(row)[1]
#         best['similarity'].append(best_similarity)
#         best['best_pairs'].append(best_pair)      
#     return best    


pairX = ['a blue sky','A man is passing by car', 'educational courses are there ready to be tought at this semester']

pairY = ['a blue sky','educational courses are there ready to be tought at this semester' , 'A man is passing by car']

cosine_similarity_all = bert_similarity (pairX, pairY)