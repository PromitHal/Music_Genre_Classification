import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import keras
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from tensorflow.keras.callbacks import Callback
from keras.models import Model
from Data import DataLoader
from DataLoader import Data
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
from sklearn.datasets import load_iris
from sklearn.feature_selection import mutual_info_classif 

class MMR_Selection:

    def __init__(self,trainx,trainy,alpha):
        self.trainx=trainx
        self.alpha=alpha
        self.trainy=trainy
        

        
    @staticmethod
    def euclidean_similarity_matrix(data):
        """
        Compute the similarity matrix using Euclidean similarity for a dataset.

        Parameters:
            - data (numpy array): A 2D numpy array where each row represents a sample and each column represents a feature.

        Returns:
            - similarity_matrix (numpy array): A square matrix containing the similarity scores between features.
        """
        transposed_data = np.transpose(data)
        num_features = transposed_data.shape[0]
        similarity_matrix = np.zeros((num_features, num_features))

        for i in range(num_features):
            for j in range(num_features):
                # Calculate the Euclidean distance between feature vectors i and j
                euclidean_distance = np.linalg.norm(transposed_data[i] - transposed_data[j])
                # Convert distance to similarity score (use 1/distance to represent similarity)
                similarity_matrix[i, j] = 1.0 / (1.0 + euclidean_distance)

        return similarity_matrix
    
    @staticmethod
    def MMR(relevance_scores,similarity_matrix,alpha):
    
        """
        Maximum Marginal Relevance (MMR) feature selection using Euclidean similarity.

        Parameters:
                - relevance_scores (array-like): Relevance scores of features with respect to the target variable.
                - similarity_matrix (array-like): Matrix containing similarity scores between features.
                - alpha (float): Trade-off parameter between relevance and redundancy (0 <= alpha <= 1).

        Returns:
            - selected_indices (list): List of indices of the selected features.
        """
        num_features = len(relevance_scores)
        selected_indices = []
        remaining_indices = list(range(num_features))

        while len(selected_indices) < num_features:
            max_mmr_score = -float('inf')
            best_index = None

            for idx in remaining_indices:
                # Calculate MMR score for each feature
                relevance_score = relevance_scores[idx]
                if len(selected_indices) > 0:
                    similarity_with_selected = np.mean(similarity_matrix[idx, selected_indices])
                else:
                    similarity_with_selected = 0.0  # No similarity if no features are selected
                mmr_score = alpha * relevance_score - (1 - alpha) * similarity_with_selected

                if mmr_score > max_mmr_score:
                    max_mmr_score = mmr_score
                    best_index = idx

            selected_indices.append(best_index)
            remaining_indices.remove(best_index)

        return selected_indices
    
    def get_features(self):
        
        sim_matrix=self.euclidean_similarity_matrix(self.trainx)
        # Calculate the information gain for each feature with respect to the target variable
        feature_importance = mutual_info_classif(self.trainx,self.trainy)
        selected_features = self.MMR(relevance_scores=feature_importance, similarity_matrix=sim_matrix, alpha=self.alpha)
        return selected_features
        
   
    def get_cols(self,matrix,num_features):
        """
        Extract only the columns from a 2D matrix that are specified in the index list.

        Parameters:
            - matrix (numpy array): The 2D matrix from which columns will be extracted.
            - index_list (list): The list of indices of columns to extract.

        Returns:
            - extracted_matrix (numpy array): The extracted submatrix containing only the specified columns.
        """
        index_list=self.get_features()
        extracted_matrix = self.trainx[:, index_list[:num_features]]
        return index_list[:num_features],self.trainx