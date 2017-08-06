'''
Created on Aug 6, 2017

@author: alon
'''

class Evaluator:
    
    def __init__(self, model, fileName):
        self.model = model
        
    
    def basic_normalized_eval(self, similarityVector, labels):
        '''
        return the distribution mass on the correct samples when using regular normalization 
        '''
        return 0
        
    
    def softmax_eval(self, similarityVector, labels):
        '''
        return the distribution mass on the correct samples when using softmax to normalized 
        '''
        return 0
    
    def elimination_eval(self, similarityVector, labels, size):
        '''
        eliminate the minimum size of the pred_dist and return #of correct eliminated / size - 
        this is the fraction of incorrect vectors that were eliminated  
        '''
        
        return 0
        
        