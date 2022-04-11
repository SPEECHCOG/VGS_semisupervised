
"""
"""

import scipy.spatial as ss
from scipy import spatial
import numpy
from sent2vec.vectorizer import Vectorizer

#vectorizer = Vectorizer()
vectorizer = Vectorizer() # pretrained_weights='bert-base-uncased'


def calculate_distance_matrix (list_of_sentences):
    vectors = vectorizer.vectors
    vectorizer.run(list_of_sentences)
    vectors = numpy.array(vectors)
    #dist = spatial.distance.cosine(vectors[0], vectors[1])
    dist = ss.distance.cdist( vectors,vectors ,  'cosine')
    return dist


def find_similar_pairs (list_of_sentences):
    sim = calculate_distance_matrix(list_of_sentences)
    best = {}
    best['best_pairs'] = []
    best['similarity'] = []
    for row in sim:

        print(row)
        best_similarity = numpy.sort(row)[1]
        best_pair = numpy.argsort(row)[1]
        best['similarity'].append(best_similarity)
        best['best_pairs'].append(best_pair)      
    return best    

list_of_sentences = ['a blue sky','A man is passing by car', 'educational courses are there ready to be tought at this semester','I love dogs']
#sim = calculate_similarity_matrix (list_of_sentences)
best = find_similar_pairs(list_of_sentences)