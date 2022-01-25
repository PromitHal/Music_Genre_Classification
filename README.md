# Music Genre Classification
Music Genre Classification via K Nearest Neighbors, Artificial Neural Networks , One dimensioncal Convolutional Neural Networks.
Dataset: GTZAN dataset(https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification)
Description  of the dataset: 
    GTZAN dataset provides us with 10 different genres of songs with 100 samples each . Each of the audio sample is of length 30 seconds.
    The genres are:
      1. Rock 
      
      2. Classical 
      
      3. Country
      
      4. Disco
      
      5. Hiphop
      
      6. Jazz 
      
      7. Metal 
      
      8. Pop 
      
      9. Reggae 
      
      10. Rock 

The steps involved in this project are :

1.Splitting audio files: The audio files were splitted into 10 parts of 3 seconds ,thus creating 10,000 files.

2. Feature Extraction : This involves extracting various acoustic features from audio files.The following features were extracted which involved windowing of the signals and subsequent  overlapping of frames to compute STFT(Short Time Fourier Transform):
3. 
                                 (i)      Chroma features 
                                 
                                 (ii)     RMS value
                                 
                                 (iii)    Spectral Centroid
                                 
                                 (iv)     Spectral Bandwidth
                                 
                                 (v)      Spectral Rolloff
                                 
                                 (vi)     Zero crossing rate
                                 
                                 (vii)    Harmonics
                                 
                                 (viii)   Perceptual
                                 
                                 (ix)     Spectral contrast
                                 
                                 (x)      Tonnetz
                                 
                                 (xi)     Mel Frequency Cepstral Coefficients
                                 
                                 (xii)    Mel Spectrogram
                                 
                                 (xiii)   Tempo
                                 
3. Feature engineering:  It involed standardizing data via feature scaling.This is an essential part of Machine Learning.

5. Building Models:
   The following models were built :
   
       1. Artifical Neural Networks (ANN)
       
       2. Decision Tree Classifier
        
       3. K-Nearest_Neighbors Classifier
        
       4. Random Forest Classifier
       
       5. One Dimensional Convolutional Neural Networks 
 
                              
                              
