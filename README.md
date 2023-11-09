
![image](https://github.com/PromitHal/Music_Genre_Classification/assets/83832850/d3b0bdfc-4f1a-4954-a9a7-11bb0389e37d)


# Music Genre Classification
Music Genre Classification via K Nearest Neighbors, Artificial Neural Networks , XGBoost Classifiers. 

**Dataset**: GTZAN dataset(https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification)

Description  of the dataset: 
    GTZAN dataset provides us with 10 different genres of songs with 100 samples each . Each of the audio sample is of length 30 seconds.
    The genres are:
    
      1. Rock  2. Classical 3. Country 4. Disco 5. Hiphop  6. Jazz  7. Metal  8. Pop  9. Reggae 10. Rock 

The steps involved in this project are :

1. **Splitting audio files**: The audio files were splitted into 10 parts of 3 seconds ,thus creating 10,000 files.

2. **Feature Extraction** : This involves extracting various acoustic features from audio files.The following features were extracted which involved windowing of the signals and subsequent  overlapping of frames to compute STFT(Short Time Fourier Transform):
(i)Chroma features (ii) RMS value (iii)Spectral Centroid    (iv)Spectral Bandwidth   (v)Spectral Rolloff  (vi)Zero crossing rate  (vii)    Harmonics (viii)Perceptual   (ix)Spectral contrast (x)Tonnetz
(xi)Mel Frequency Cepstral Coefficients (xii)    Mel Spectrogram  (xiii)   Tempo
                                 
3. **Feature engineering**:  It involed standardizing data via feature scaling.This is an essential part of Machine Learning.

5. **Building Models**:
   The following models were built :
   
       1. Artifical Neural Networks (ANN)- Greedy Souped.
       
       2. Decision Tree Classifier
        
       3. K-Nearest_Neighbors Classifier
        
       4. Random Forest Classifier
       
       5. One Dimensional Convolutional Neural Networks ( This was just experimental, suited mainly for temporal data. I applied on the frequencey features which is not a good idea!).
   
6.**Dimensionality Reduction** :

Combined Maximum Margin Relevance algorithm with Mutual Informationn between features and target variable for feature selection.
Proposed method outperformed PCA for both XGBoost classifier and Artificial Neural Network in terms of f1-score(4.6% and 1.83%) respectively.

![image](https://github.com/PromitHal/Music_Genre_Classification/assets/83832850/ad05b11b-dc43-4262-8550-c10074ca64f3)   ![image](https://github.com/PromitHal/Music_Genre_Classification/assets/83832850/b34334e9-d22b-41d4-b4a9-ed6ead558f96)


![image](https://github.com/PromitHal/Music_Genre_Classification/assets/83832850/87cc9d78-c673-4d93-b633-b69e87d160ce)

                              
